"""AIME 2026 benchmark: 30 problemas de matemática olímpica, respuesta entera.

Mismo diseño anti-cortes que bench.py (checkpoint inmediato, brazos que se
desactivan por cuota, resume al re-correr). Scoring: se extrae el último
\\boxed{N} (o el último entero) de la respuesta y se compara exacto contra
la respuesta oficial (dataset MathArena/aime_2026, HuggingFace).

A diferencia de HumanEval, aquí no hay ejecución que verifique: es el
terreno donde el camino "judge" de fable6 (Fable 5 sintetizando) trabaja.

Uso:
    uv run python -m fable_router.bench_aime
    uv run python -m fable_router.bench_aime --arms fable6 --limit 5
    uv run python -m fable_router.bench_aime --report
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
import urllib.request
from pathlib import Path

from .bench import ARMS, CONSECUTIVE_ERRORS_TO_DISABLE, QUOTA_RE

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPO_ROOT / "bench_data" / "aime2026.json"
RESULTS_FILE = REPO_ROOT / "bench_results" / "aime2026.jsonl"
DATASET_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=MathArena/aime_2026&config=default&split=train&limit=100"
)

PROMPT_TEMPLATE = """\
Solve the following AIME problem. The answer is an integer between 0 and 999. \
Think carefully, then end your response with the final answer in the form \
\\boxed{{N}}.

{problem}"""


def load_dataset() -> list[dict]:
    if not DATA_FILE.exists():
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando AIME 2026 -> {DATA_FILE}")
        with urllib.request.urlopen(DATASET_URL, timeout=60) as resp:
            data = json.load(resp)
        rows = [r["row"] for r in data["rows"]]
        DATA_FILE.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def extract_answer(text: str) -> int | None:
    boxed = re.findall(r"\\boxed\{\s*(\d+)\s*\}", text)
    if boxed:
        return int(boxed[-1])
    ints = re.findall(r"\b\d{1,3}\b", text[-500:])
    return int(ints[-1]) if ints else None


def load_done() -> set[tuple[str, int]]:
    done = set()
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["arm"], r["problem_idx"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def run(arm_names: list[str], limit: int | None) -> None:
    problems = load_dataset()
    if limit:
        problems = problems[:limit]
    done = load_done()
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    disabled: dict[str, str] = {}
    consecutive_errors: dict[str, int] = {a: 0 for a in arm_names}
    write_lock = threading.Lock()

    def run_arm(arm: str, prob: dict) -> None:
        result = ARMS[arm](PROMPT_TEMPLATE.format(problem=prob["problem"]))
        if result.ok:
            consecutive_errors[arm] = 0
            got = extract_answer(result.text)
            passed = got == prob["answer"]
            with write_lock:
                with open(RESULTS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "arm": arm, "problem_idx": prob["problem_idx"],
                        "pass": passed, "got": got, "expected": prob["answer"],
                        "provider": result.provider, "model": result.model,
                        "latency_s": round(result.latency_s, 2),
                        "in_tokens": result.input_tokens,
                        "out_tokens": result.output_tokens,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }, ensure_ascii=False) + "\n")
            print(f"  {arm}: {'PASS' if passed else f'fail (dio {got}, era {prob['answer']})'} "
                  f"({result.latency_s:.0f}s)")
            return
        if QUOTA_RE.search(result.error or ""):
            disabled[arm] = f"cuota/rate-limit: {result.error[:120]}"
        else:
            consecutive_errors[arm] += 1
            if consecutive_errors[arm] >= CONSECUTIVE_ERRORS_TO_DISABLE:
                disabled[arm] = f"{CONSECUTIVE_ERRORS_TO_DISABLE} errores seguidos: {result.error[:120]}"
        print(f"  {arm}: ERROR {result.error[:100]}")

    for i, prob in enumerate(problems):
        live = [a for a in arm_names if a not in disabled
                and (a, prob["problem_idx"]) not in done]
        if not live:
            continue
        print(f"[{i + 1}/{len(problems)}] AIME {prob['problem_idx']} -> {', '.join(live)}")
        threads = [threading.Thread(target=run_arm, args=(a, prob)) for a in live]
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
    report()


def report() -> None:
    latest: dict[tuple[str, int], dict] = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    latest[(r["arm"], r["problem_idx"])] = r
                except (json.JSONDecodeError, KeyError):
                    continue
    stats: dict[str, list[dict]] = {}
    for (arm, _), r in latest.items():
        stats.setdefault(arm, []).append(r)

    print(f"\n{'brazo':<16}{'acierto':>9}{'n':>5}{'lat p50':>9}")
    for arm in sorted(stats, key=lambda a: -sum(r["pass"] for r in stats[a]) / len(stats[a])):
        rows = stats[arm]
        n = len(rows)
        rate = sum(r["pass"] for r in rows) / n
        lats = sorted(r["latency_s"] for r in rows)
        print(f"{arm:<16}{rate:>8.1%}{n:>5}{lats[n // 2]:>8.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Sin "router": su cadena "code" no aplica a matemática y contaminaría la lectura.
    ap.add_argument("--arms", default="copilot-sonnet,copilot-terra,gemini-pro,glm52,qwen-max,fable6")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report", action="store_true")
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
