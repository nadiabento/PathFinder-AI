"""Dijkstra e A* próprios, sobre uma única implementação (secção 4.1).

Dijkstra é A* com h = 0; Weighted A* é A* com f = g + w*h. Manter um só ciclo evita
duplicar código e garante que as comparações entre algoritmos medem a mesma coisa.

Escolhas de implementação (úteis para o relatório/defesa)
---------------------------------------------------------
* Fila de prioridade: heapq com entradas (f, h, contador, nó). Desempate: menor h primeiro
  (prefere nós mais perto do objetivo) e depois ordem de inserção (determinístico).
* Remoção preguiçosa: em vez de decrease-key, empurra-se uma nova entrada e ignoram-se as
  entradas obsoletas (nós já expandidos). Complexidade O(E log V).
* Conjunto `closed`: cada nó é expandido no máximo uma vez. É isto que impede ciclos e garante
  que o Cenário C termina: quando a fronteira esvazia, não há caminho.
* "Nós expandidos" = nós retirados da fronteira e processados (o do objetivo conta, porque
  é retirado antes de se parar). Os valores exatos dependem do desempate.
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Callable

from .grid import Grid

Pos = tuple[int, int]
Heuristic = Callable[[Pos, Pos], float]


@dataclass
class SearchResult:
    found: bool
    path: list[Pos]
    cost: float
    expanded: int          # nós expandidos
    generated: int         # nós inseridos na fronteira (indicador de memória)
    max_frontier: int      # tamanho máximo da fronteira
    time_s: float          # tempo de execução (s)
    explored: list[Pos] = field(default_factory=list)  # ordem de expansão (para visualização)
    algorithm: str = ""
    moves: int = 4
    weight: float = 1.0

    def __str__(self) -> str:
        if not self.found:
            return f"{self.algorithm}: sem caminho (expandidos={self.expanded})"
        return (f"{self.algorithm}: custo={self.cost:.4f} passos={len(self.path) - 1} "
                f"expandidos={self.expanded} tempo={self.time_s * 1000:.3f} ms")


def search(grid: Grid, moves: int = 4, heuristic: Heuristic | None = None,
           weight: float = 1.0, name: str | None = None) -> SearchResult:
    """Pesquisa de custo mínimo de grid.start a grid.goal.

    heuristic=None -> Dijkstra. Com heurística -> A* (f = g + weight*h).
    weight == 1 com heurística admissível e consistente garante optimalidade.
    """
    if weight < 1:
        raise ValueError("weight tem de ser >= 1")
    h = heuristic if heuristic is not None else (lambda a, b: 0.0)
    start, goal = grid.start, grid.goal
    if name is None:
        name = "Dijkstra" if heuristic is None else (
            "A*" if weight == 1 else f"A* (w={weight:g})")

    counter = itertools.count()
    t0 = time.perf_counter()

    g_best: dict[Pos, float] = {start: 0.0}
    parent: dict[Pos, Pos | None] = {start: None}
    h0 = weight * h(start, goal)
    frontier: list[tuple[float, float, int, Pos]] = [(h0, h0, next(counter), start)]
    closed: set[Pos] = set()
    explored: list[Pos] = []
    generated = 1
    max_frontier = 1
    found = False

    while frontier:
        f, hn, _, node = heapq.heappop(frontier)
        if node in closed:          # entrada obsoleta (já expandido com custo menor)
            continue
        closed.add(node)
        explored.append(node)
        if node == goal:
            found = True
            break
        g_node = g_best[node]
        for nb, step in grid.neighbors(node, moves):
            if nb in closed:
                continue
            g_new = g_node + step
            if g_new < g_best.get(nb, math.inf):
                g_best[nb] = g_new
                parent[nb] = node
                h_nb = weight * h(nb, goal)
                heapq.heappush(frontier, (g_new + h_nb, h_nb, next(counter), nb))
                generated += 1
        max_frontier = max(max_frontier, len(frontier))

    elapsed = time.perf_counter() - t0

    path: list[Pos] = []
    cost = math.inf
    if found:
        cost = g_best[goal]
        node: Pos | None = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()

    return SearchResult(found=found, path=path, cost=cost, expanded=len(closed),
                        generated=generated, max_frontier=max_frontier, time_s=elapsed,
                        explored=explored, algorithm=name, moves=moves, weight=weight)


def dijkstra(grid: Grid, moves: int = 4) -> SearchResult:
    return search(grid, moves, heuristic=None)


def astar(grid: Grid, moves: int = 4, heuristic: Heuristic | None = None,
          weight: float = 1.0, heuristic_name: str | None = None) -> SearchResult:
    """A*. Por omissão usa Manhattan (4 dir.) ou octile (8 dir.), as mais informadas admissíveis."""
    from .heuristics import manhattan, octile
    if heuristic is None:
        heuristic = manhattan if moves == 4 else octile
    res = search(grid, moves, heuristic=heuristic, weight=weight)
    if heuristic_name:
        res.algorithm += f" [{heuristic_name}]"
    return res
