"""Extensão (4.6): Weighted A* — compromisso entre qualidade do caminho e rapidez.

f(n) = g(n) + w * h(n), com w >= 1 e h admissível.
  * w = 1  -> A* clássico (ótimo).
  * w > 1  -> a pesquisa "confia mais" na heurística: expande menos nós, mas pode devolver
              um caminho mais caro. Garantia teórica: custo <= w * custo_ótimo.
  * w -> infinito -> pesquisa gananciosa (greedy best-first).

Uso (a partir da raiz do projeto):
    python -m extension.weighted_astar scenarios/anexo_a/B_terrenos.txt --moves 4
"""
from __future__ import annotations

import argparse
import math
import statistics

from planner.grid import Grid
from planner.heuristics import HEURISTICS
from planner.search import search

DEFAULT_WEIGHTS = (1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0)
EPS = 1e-9


def weight_sweep(grid: Grid, moves: int, heuristic_name: str,
                 weights=DEFAULT_WEIGHTS, repeats: int = 5) -> list[dict]:
    """Corre A* com cada peso e compara com o ótimo (Dijkstra).

    Devolve uma linha por peso: custo, razão custo/ótimo, nós expandidos, ganho face a w=1
    e tempo médio de `repeats` execuções.
    """
    h = HEURISTICS[heuristic_name]
    optimal = search(grid, moves)  # Dijkstra = referência ótima
    base = search(grid, moves, h, weight=1.0)
    rows = []
    for w in weights:
        res = search(grid, moves, h, weight=w)
        times = [search(grid, moves, h, weight=w).time_s for _ in range(repeats)]
        ratio = res.cost / optimal.cost if res.found and optimal.found else math.nan
        rows.append({
            "cenario": grid.name, "moves": moves, "heuristica": heuristic_name, "w": w,
            "encontrado": res.found,
            "custo": res.cost, "custo_otimo": optimal.cost,
            "razao_custo": ratio,
            "otimo": res.found and abs(res.cost - optimal.cost) < EPS,
            "expandidos": res.expanded,
            "reducao_expandidos": 1 - res.expanded / base.expanded if base.expanded else 0.0,
            "tempo_ms": statistics.mean(times) * 1000,
            "limite_teorico_ok": (not res.found) or res.cost <= w * optimal.cost + EPS,
        })
    return rows


def first_suboptimal_weight(grid: Grid, moves: int, heuristic_name: str,
                            lo: float = 1.0, hi: float = 20.0, step: float = 0.05):
    """Menor w (na malha dada) para o qual o caminho deixa de ser ótimo; None se nunca."""
    h = HEURISTICS[heuristic_name]
    optimal = search(grid, moves).cost
    w = lo + step
    while w <= hi + EPS:
        res = search(grid, moves, h, weight=w)
        if res.found and res.cost > optimal + EPS:
            return round(w, 4)
        w += step
    return None


def _print(rows: list[dict]) -> None:
    print(f"{'w':>5} {'custo':>9} {'c/ótimo':>8} {'exp.':>6} {'-exp.':>7} {'ms':>8}  ótimo")
    for r in rows:
        print(f"{r['w']:>5g} {r['custo']:>9.4f} {r['razao_custo']:>8.3f} {r['expandidos']:>6d} "
              f"{r['reducao_expandidos']:>6.0%} {r['tempo_ms']:>8.3f}  {'sim' if r['otimo'] else 'NÃO'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cenario")
    ap.add_argument("--moves", type=int, choices=(4, 8), default=4)
    ap.add_argument("--heuristica", choices=list(HEURISTICS), default=None)
    args = ap.parse_args()
    g = Grid.from_file(args.cenario)
    hn = args.heuristica or ("manhattan" if args.moves == 4 else "octile")
    _print(weight_sweep(g, args.moves, hn))
    print("primeiro w sub-ótimo:", first_suboptimal_weight(g, args.moves, hn))
