"""Verificação numérica de admissibilidade e consistência (secção 4.2 / 4.3).

Admissível:  h(n) <= h*(n)  (custo ótimo real de n até ao objetivo).
Consistente: h(n) <= c(n, n') + h(n') para cada aresta (n, n').

Os testes usam grelhas aleatórias (semente fixa) com terrenos 1..9 e obstáculos.
Executar:  python -m tests.test_heuristics
"""
import random

from planner import Grid, HEURISTICS, dijkstra, astar
from planner.heuristics import ADMISSIBLE_IN, manhattan

EPS = 1e-9


def random_grid(n, seed, obstacles=0.15, terrain=0.25):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        row = []
        for _ in range(n):
            x = rng.random()
            row.append("#" if x < obstacles else
                       str(rng.randint(2, 9)) if x < obstacles + terrain else ".")
        rows.append(row)
    rows[0][0], rows[-1][-1] = "S", "G"
    return Grid.from_text("\n".join("".join(r) for r in rows), name=f"rnd{seed}")


def test_admissibilidade():
    for seed in range(6):
        grid = random_grid(8, seed)
        goal = grid.goal
        for moves in (4, 8):
            for r in range(grid.rows):
                for c in range(grid.cols):
                    if not grid.is_free(r, c):
                        continue
                    g2 = Grid(grid.cost, (r, c), goal)
                    res = dijkstra(g2, moves)
                    if not res.found:
                        continue
                    for hn, h in HEURISTICS.items():
                        if moves in ADMISSIBLE_IN[hn]:
                            assert h((r, c), goal) <= res.cost + EPS, (hn, moves, (r, c), seed)


def test_consistencia():
    for seed in range(6):
        grid = random_grid(10, seed)
        for moves in (4, 8):
            for r in range(grid.rows):
                for c in range(grid.cols):
                    if not grid.is_free(r, c):
                        continue
                    for nb, step in grid.neighbors((r, c), moves):
                        for hn, h in HEURISTICS.items():
                            if moves in ADMISSIBLE_IN[hn]:
                                assert h((r, c), grid.goal) <= step + h(nb, grid.goal) + EPS


def test_manhattan_nao_admissivel_em_8_direcoes():
    """Contra-exemplo concreto (encontrado com semente 16): Manhattan sobrestima e A* falha."""
    g = Grid.from_text("S.....\n..#...\n......\n..#...\n......\n...#.G")
    otimo = dijkstra(g, 8).cost
    com_manhattan = astar(g, 8, manhattan).cost
    assert com_manhattan > otimo + 1e-6, (otimo, com_manhattan)
    # Com heurísticas admissíveis em 8 direções o resultado é ótimo.
    for hn in ("euclidiana", "chebyshev", "octile"):
        assert abs(astar(g, 8, HEURISTICS[hn]).cost - otimo) < EPS


def test_optimalidade_vs_dijkstra_em_grelhas_aleatorias():
    for seed in range(30):
        grid = random_grid(15, seed)
        for moves in (4, 8):
            ref = dijkstra(grid, moves)
            for hn, h in HEURISTICS.items():
                if moves in ADMISSIBLE_IN[hn]:
                    r = astar(grid, moves, h)
                    assert r.found == ref.found
                    if ref.found:
                        assert abs(r.cost - ref.cost) < EPS, (seed, moves, hn)


def test_astar_expande_menos_ou_igual_que_dijkstra_em_media():
    tot_d = tot_a = 0
    for seed in range(20):
        grid = random_grid(20, seed)
        tot_d += dijkstra(grid, 4).expanded
        tot_a += astar(grid, 4, manhattan).expanded
    assert tot_a < tot_d


def test_weighted_astar_limite_teorico():
    """Com h admissível, custo(w) <= w * custo_ótimo."""
    from planner import search
    for seed in range(20):
        grid = random_grid(15, seed)
        for moves, hn in ((4, "manhattan"), (8, "octile")):
            opt = dijkstra(grid, moves)
            for w in (1.5, 2, 5):
                r = search(grid, moves, HEURISTICS[hn], weight=w)
                assert r.found == opt.found
                if opt.found:
                    assert r.cost <= w * opt.cost + EPS


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok ", k)
