"""Genera los PNG de resultados del benchmark (estilo informe de labs).

    uv run --with matplotlib python -m fable_router.bench_chart

Lee bench_results/humaneval.jsonl y escribe bench_results/*.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RESULTS = Path(__file__).resolve().parents[2] / "bench_results"

# Etiqueta bonita + grupo (define color): fable6 resaltado, modos multi-modelo
# nuestros en azul medio, modelos individuales en gris.
LABELS = {
    "fable6": ("Fable 6 Router", "hero"),
    "moa": ("MoA naive (síntesis)", "ours"),
    "router": ("Router (cadena code)", "ours"),
    "copilot-sonnet": ("Claude Sonnet 5", "single"),
    "copilot-terra": ("GPT-5.6 Terra", "single"),
    "gemini-pro": ("Gemini 3.1 Pro", "single"),
    "glm52": ("GLM 5.2", "single"),
    "qwen-max": ("Qwen 3.7 Max", "single"),
}
COLORS = {"hero": "#1e3a6e", "ours": "#5b7fc4", "single": "#e8eaed"}
EDGE = "#3c4043"


def load_stats() -> dict[str, dict]:
    latest: dict[tuple[str, str], dict] = {}
    with open(RESULTS / "humaneval.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                latest[(r["arm"], r["task_id"])] = r
            except (json.JSONDecodeError, KeyError):
                continue
    stats: dict[str, dict] = {}
    for (arm, _), r in latest.items():
        s = stats.setdefault(arm, {"pass": 0, "n": 0, "lats": []})
        s["pass"] += bool(r["pass"])
        s["n"] += 1
        s["lats"].append(r["latency_s"])
    for s in stats.values():
        s["rate"] = s["pass"] / s["n"]
        s["p50"] = sorted(s["lats"])[len(s["lats"]) // 2]
    return stats


def rounded_bar(ax, x, height, width, color):
    ax.add_patch(FancyBboxPatch(
        (x - width / 2, 0), width, height,
        boxstyle="round,pad=0,rounding_size=0.012",
        mutation_aspect=width / 0.024,
        facecolor=color, edgecolor=EDGE, linewidth=1.2,
    ))


def chart_bars(stats: dict) -> None:
    arms = [a for a in LABELS if a in stats]
    arms.sort(key=lambda a: -stats[a]["rate"])
    fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=150)
    for i, arm in enumerate(arms):
        rate = stats[arm]["rate"]
        rounded_bar(ax, i, rate, 0.62, COLORS[LABELS[arm][1]])
        ax.text(i, rate + 0.025, f"{rate:.1%}", ha="center", fontsize=12.5,
                color="#202124")
    n = min(stats[a]["n"] for a in arms)
    fig.suptitle("HumanEval pass@1", fontsize=17, fontweight="bold", x=0.065, y=0.985,
                 ha="left")
    ax.set_title(f"{n} problemas, mismo harness y prompt para todos los brazos",
                 fontsize=10, color="#5f6368", loc="left", pad=12)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([LABELS[a][0] for a in arms], rotation=30, ha="right", fontsize=11)
    ax.set_ylim(0, 1.06)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "humaneval_bars.png", bbox_inches="tight", facecolor="white")
    print(RESULTS / "humaneval_bars.png")


def chart_quality_latency(stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    for i, (arm, (label, group)) in enumerate(LABELS.items()):
        if arm not in stats:
            continue
        s = stats[arm]
        color = COLORS[group] if group != "single" else "#9aa0a6"
        ax.scatter(s["p50"], s["rate"], s=170 if group == "hero" else 110,
                   color=color, edgecolor=EDGE, linewidth=1.2, zorder=3)
        dy = 9 if i % 2 == 0 else -18  # alterna arriba/abajo: evita colisiones
        ax.annotate(label, (s["p50"], s["rate"]), textcoords="offset points",
                    xytext=(9, dy), fontsize=10.5, color="#202124")
    fig.suptitle("Calidad vs latencia", fontsize=17, fontweight="bold", x=0.065, y=0.985,
                 ha="left")
    ax.set_title("pass@1 vs latencia mediana por problema (log)",
                 fontsize=10, color="#5f6368", loc="left", pad=12)
    ax.set_yticks([0.98, 0.99, 1.0])
    ax.set_ylim(0.975, 1.005)
    ax.set_xscale("log")
    ax.set_xlabel("Latencia p50 (s, log)", fontsize=12)
    ax.set_ylabel("pass@1", fontsize=12)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(True, which="major", color="#f1f3f4", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "humaneval_quality_latency.png", bbox_inches="tight",
                facecolor="white")
    print(RESULTS / "humaneval_quality_latency.png")


if __name__ == "__main__":
    stats = load_stats()
    for arm, s in sorted(stats.items(), key=lambda kv: -kv[1]["rate"]):
        print(f"{arm:<16}{s['rate']:>7.1%}  n={s['n']:<4} p50={s['p50']:.1f}s")
    chart_bars(stats)
    chart_quality_latency(stats)
