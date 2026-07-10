"""HumanEval benchmark: cada brazo (modelo/router) contra los 164 problemas.

Diseñado para planes de suscripción con cuotas chicas — nunca "queda a mitad":
- Cada resultado se appendea de inmediato a bench_results/humaneval.jsonl
  (crash-safe); re-correr salta lo ya hecho y completa los huecos.
- Un brazo que devuelve error de cuota/rate-limit (o 3 errores seguidos) se
  desactiva por el resto de la corrida; los demás siguen. Cuando la ventana
  del plan se renueva, volver a correr el mismo comando reanuda ese brazo.
- Los brazos corren en paralelo por problema (1 thread por brazo), así una
  interrupción deja cobertura pareja en todos los brazos, no uno completo
  y el resto vacío.

Uso:
    python -m fable_router.bench                       # todos los brazos, 164
    python -m fable_router.bench --arms claude --limit 40
    python -m fable_router.bench --report              # tabla pass@1
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from .adapters import claude_cli, codex_cli, copilot, dashscope, opencode_cli, vertex
from .adapters.base import Result
from . import router

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPO_ROOT / "bench_data" / "HumanEval.jsonl.gz"
RESULTS_FILE = REPO_ROOT / "bench_results" / "humaneval.jsonl"
DATASET_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"

ARMS: dict[str, callable] = {
    "codex": lambda p: codex_cli.complete(p),
    "claude": lambda p: claude_cli.complete(p),
    "gemini-pro": lambda p: vertex.complete("gemini-pro", p),
    "gemini-flash": lambda p: vertex.complete("gemini-flash", p),
    "glm": lambda p: opencode_cli.complete("glm", p),
    "qwen-oc": lambda p: opencode_cli.complete("qwen", p),
    "qwen-max": lambda p: dashscope.complete("qwen-max", p),
    "copilot-sonnet": lambda p: copilot.complete("sonnet", p),
    "copilot-terra": lambda p: copilot.complete("terra", p),
    "copilot-luna": lambda p: copilot.complete("luna", p),
    "copilot-kimi": lambda p: copilot.complete("kimi", p),
    "router": lambda p: router.ask(p, task_type="code"),
}

QUOTA_RE = re.compile(
    r"429|rate.?limit|quota|resource.?exhausted|too many requests|usage limit"
    r"|credit|l[ií]mite",
    re.IGNORECASE,
)
CONSECUTIVE_ERRORS_TO_DISABLE = 3

PROMPT_TEMPLATE = """\
Implement the following Python function. Respond with a single ```python code \
block containing the complete function (including the exact signature shown) \
plus any imports or helpers it needs. No explanations, no tests.

```python
{prompt}
```"""


def load_dataset() -> list[dict]:
    if not DATA_FILE.exists():
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando HumanEval -> {DATA_FILE}")
        urllib.request.urlretrieve(DATASET_URL, DATA_FILE)
    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else text


def run_candidate(code: str, task: dict) -> tuple[bool, str]:
    """Ejecuta candidato + tests oficiales en un subproceso con timeout."""
    if f"def {task['entry_point']}" not in code:
        code = task["prompt"] + "\n" + code  # completion-style: le faltó la firma
    program = f"{code}\n\n{task['test']}\ncheck({task['entry_point']})\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, text=True, timeout=15,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return False, "exec timeout"
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or "").strip()[-200:]


def load_done() -> set[tuple[str, str]]:
    done = set()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["arm"], r["task_id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def run(arm_names: list[str], limit: int | None) -> None:
    tasks = load_dataset()
    if limit:
        tasks = tasks[:limit]
    done = load_done()
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    disabled: dict[str, str] = {}
    consecutive_errors: dict[str, int] = {a: 0 for a in arm_names}
    write_lock = threading.Lock()

    def save(row: dict) -> None:
        with write_lock:
            with open(RESULTS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def run_arm(arm: str, task: dict) -> None:
        result: Result = ARMS[arm](PROMPT_TEMPLATE.format(prompt=task["prompt"]))
        if result.ok:
            consecutive_errors[arm] = 0
            passed, fail_reason = run_candidate(extract_code(result.text), task)
            save({
                "arm": arm, "task_id": task["task_id"], "pass": passed,
                "error": None, "fail_reason": fail_reason or None,
                "provider": result.provider, "model": result.model,
                "latency_s": round(result.latency_s, 2),
                "in_tokens": result.input_tokens, "out_tokens": result.output_tokens,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            print(f"  {arm}: {'PASS' if passed else 'fail'} ({result.latency_s:.0f}s)")
            return
        # Error del proveedor: cuota => desactivar ya; si no, tolerar hasta 3 seguidos.
        # No se guarda checkpoint (el re-run debe reintentar este task).
        if QUOTA_RE.search(result.error or ""):
            disabled[arm] = f"cuota/rate-limit: {result.error[:120]}"
        else:
            consecutive_errors[arm] += 1
            if consecutive_errors[arm] >= CONSECUTIVE_ERRORS_TO_DISABLE:
                disabled[arm] = f"{CONSECUTIVE_ERRORS_TO_DISABLE} errores seguidos: {result.error[:120]}"
        print(f"  {arm}: ERROR {result.error[:100]}")

    for i, task in enumerate(tasks):
        live = [a for a in arm_names if a not in disabled
                and (a, task["task_id"]) not in done]
        if not live:
            continue
        print(f"[{i + 1}/{len(tasks)}] {task['task_id']} -> {', '.join(live)}")
        threads = [threading.Thread(target=run_arm, args=(a, task)) for a in live]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if len(disabled) == len(arm_names):
            break

    print("\n--- Fin de corrida ---")
    for arm, reason in disabled.items():
        print(f"DESACTIVADO {arm}: {reason}")
    if disabled:
        print("Re-corré el mismo comando cuando se renueve la cuota; retoma donde quedó.")
    report(arm_names)


def report(arm_names: list[str] | None = None) -> None:
    latest: dict[tuple[str, str], dict] = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    latest[(r["arm"], r["task_id"])] = r
                except (json.JSONDecodeError, KeyError):
                    continue
    stats: dict[str, list[dict]] = {}
    for (arm, _tid), r in latest.items():
        stats.setdefault(arm, []).append(r)

    print(f"\n{'brazo':<14}{'pass@1':>8}{'n':>6}{'lat p50':>9}")
    for arm in sorted(stats, key=lambda a: -sum(r["pass"] for r in stats[a]) / len(stats[a])):
        if arm_names and arm not in arm_names:
            continue
        rows = stats[arm]
        n = len(rows)
        rate = sum(r["pass"] for r in rows) / n
        lats = sorted(r["latency_s"] for r in rows)
        print(f"{arm:<14}{rate:>7.1%}{n:>6}{lats[n // 2]:>8.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default=",".join(ARMS), help="brazos separados por coma")
    ap.add_argument("--limit", type=int, default=None, help="primeros N problemas")
    ap.add_argument("--report", action="store_true", help="solo mostrar tabla")
    args = ap.parse_args()

    if args.report:
        report()
        return
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arm_names if a not in ARMS]
    if unknown:
        ap.error(f"brazos desconocidos: {unknown}. Válidos: {list(ARMS)}")
    run(arm_names, args.limit)


if __name__ == "__main__":
    main()
