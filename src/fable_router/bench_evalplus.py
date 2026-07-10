"""HumanEval+ (EvalPlus): mismos 164 problemas, tests 80x más rigurosos.

Dos fases:
  1. Generación (este módulo): cada brazo produce su solución por problema,
     checkpointeada en bench_results/evalplus_samples/<brazo>.jsonl con el
     formato que espera el evaluador oficial ({task_id, solution}).
  2. Evaluación (harness oficial de EvalPlus, no el nuestro — para poder
     publicar los números con credibilidad):
         uv run --with evalplus python -m evalplus.evaluate \
             --dataset humaneval --samples bench_results/evalplus_samples/<brazo>.jsonl
     Reporta pass@1 "Base" (HumanEval original) y "Base + Extra" (HumanEval+).

Mismo diseño anti-cortes: resume por (brazo, task_id), brazos se desactivan
por cuota y siguen los demás.

Uso:
    uv run python -m fable_router.bench_evalplus --arms fable6 --limit 10
    uv run python -m fable_router.bench_evalplus            # brazos por defecto
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from .bench import (
    ARMS,
    CONSECUTIVE_ERRORS_TO_DISABLE,
    PROMPT_TEMPLATE,
    QUOTA_RE,
    extract_code,
    load_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "bench_results" / "evalplus_samples"


def load_done(arm: str) -> set[str]:
    path = SAMPLES_DIR / f"{arm}.jsonl"
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["task_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def run(arm_names: list[str], limit: int | None) -> None:
    tasks = load_dataset()
    if limit:
        tasks = tasks[:limit]
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    done = {arm: load_done(arm) for arm in arm_names}

    disabled: dict[str, str] = {}
    consecutive_errors: dict[str, int] = {a: 0 for a in arm_names}
    write_lock = threading.Lock()

    def run_arm(arm: str, task: dict) -> None:
        result = ARMS[arm](PROMPT_TEMPLATE.format(prompt=task["prompt"]))
        if result.ok:
            consecutive_errors[arm] = 0
            code = extract_code(result.text)
            if f"def {task['entry_point']}" not in code:
                code = task["prompt"] + "\n" + code
            with write_lock:
                with open(SAMPLES_DIR / f"{arm}.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "task_id": task["task_id"], "solution": code,
                        "latency_s": round(result.latency_s, 2), "model": result.model,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }, ensure_ascii=False) + "\n")
            print(f"  {arm}: generado ({result.latency_s:.0f}s)")
            return
        if QUOTA_RE.search(result.error or ""):
            disabled[arm] = f"cuota/rate-limit: {result.error[:120]}"
        else:
            consecutive_errors[arm] += 1
            if consecutive_errors[arm] >= CONSECUTIVE_ERRORS_TO_DISABLE:
                disabled[arm] = f"{CONSECUTIVE_ERRORS_TO_DISABLE} errores seguidos: {result.error[:120]}"
        print(f"  {arm}: ERROR {result.error[:100]}")

    for i, task in enumerate(tasks):
        live = [a for a in arm_names if a not in disabled and task["task_id"] not in done[a]]
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

    print("\n--- Fin de generación ---")
    for arm, reason in disabled.items():
        print(f"DESACTIVADO {arm}: {reason}")
    print("Evalúa con el harness oficial:")
    for arm in arm_names:
        print(f"  uv run --with evalplus python -m evalplus.evaluate "
              f"--dataset humaneval --samples bench_results/evalplus_samples/{arm}.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="copilot-sonnet,copilot-terra,gemini-pro,glm52,qwen-max,router,fable6")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arm_names if a not in ARMS]
    if unknown:
        ap.error(f"brazos desconocidos: {unknown}. Válidos: {list(ARMS)}")
    run(arm_names, args.limit)


if __name__ == "__main__":
    main()
