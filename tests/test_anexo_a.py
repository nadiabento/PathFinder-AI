"""Validação nos cenários do Anexo A (custos ótimos esperados e deteção de 'sem caminho').

Executar:  python -m tests.test_anexo_a    (ou  pytest  se estiver instalado)
"""
import math
from pathlib import Path

from planner import Grid, HEURISTICS, dijkstra, astar
from planner.heuristics import ADMISSIBLE_IN

ROOT = Path(__file__).resolve().parent.parent / "scenarios" / "anexo_a"
EXPECTED = {  # (4 direções, 8 direções)
    "A_labirinto": (17, 17),
    "B_terrenos": (20, 16 + 2 * math.sqrt(2)),
    "C_sem_caminho": (None, None),
}


def load(name):
    return Grid.from_file(ROOT / f"{name}.txt")


def test_dijkstra_custos_esperados():
    for name, exp in EXPECTED.items():
        g = load(name)
        for moves, e in zip((4, 8), exp):
            r = dijkstra(g, moves)
            if e is None:
                assert not r.found and r.path == [] and math.isinf(r.cost)
            else:
                assert r.found and abs(r.cost - e) < 1e-9, (name, moves, r.cost, e)
                assert abs(g.path_cost(r.path, moves) - r.cost) < 1e-9


def test_astar_optimo_com_heuristicas_admissiveis():
    for name, exp in EXPECTED.items():
        g = load(name)
        for moves, e in zip((4, 8), exp):
            for hn, h in HEURISTICS.items():
                if moves not in ADMISSIBLE_IN[hn]:
                    continue
                r = astar(g, moves, h)
                if e is None:
                    assert not r.found
                else:
                    assert abs(r.cost - e) < 1e-9, (name, moves, hn, r.cost)


def test_cenario_c_termina_e_explora_regiao_de_S():
    g = load("C_sem_caminho")
    for moves in (4, 8):
        assert dijkstra(g, moves).expanded == 6
        assert astar(g, moves).expanded == 6


def test_nos_expandidos_ordem_de_grandeza():
    a, b = load("A_labirinto"), load("B_terrenos")
    assert 30 <= dijkstra(a, 4).expanded <= 40
    assert 22 <= astar(a, 4, HEURISTICS["manhattan"]).expanded <= 32
    assert 27 <= dijkstra(b, 4).expanded <= 35
    assert 25 <= astar(b, 4, HEURISTICS["manhattan"]).expanded <= 33


def test_diagonal_nao_corta_esquinas():
    g = Grid.from_text("S#\n#G")
    assert not dijkstra(g, 8).found          # só se chegaria "cortando" a esquina
    g = Grid.from_text("S.\n#G")
    assert abs(dijkstra(g, 8).cost - 2.0) < 1e-9  # um dos lados livre não basta: tem de ser 2 passos


def test_ficheiros_com_crlf():
    assert Grid.from_text("S.G\r\n...\r\n").cols == 3


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok ", k)
