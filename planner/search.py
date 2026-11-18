"""Dijkstra e A* próprios, sobre uma única implementação."""
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


# CLASSE SearchResult
# O que guarda: tudo o que uma pesquisa devolve (encontrou?, caminho, custo,
#               nós expandidos, nós gerados, pico da fronteira, tempo, ordem de expansão).
# Para quê: o correr() e os gráficos leem estes campos.
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

    # __str__
    # O que faz: define o texto que aparece com print(resultado).
    def __str__(self) -> str:
        if not self.found:
            return f"{self.algorithm}: sem caminho (expandidos={self.expanded})"
        return (f"{self.algorithm}: custo={self.cost:.4f} passos={len(self.path) - 1} "
                f"expandidos={self.expanded} tempo={self.time_s * 1000:.3f} ms")


# search  (o coração do projeto: um só ciclo para Dijkstra, A* e Weighted A*)
# O que faz: encontra o caminho de menor custo de grid.start a grid.goal.
# Recebe:    a grelha, 4 ou 8 direções, a heurística (None = Dijkstra) e o peso w.
# Devolve:   um SearchResult.
# Prioridade: f = g + w*h (com h = 0, f = g e é o Dijkstra).
# Estruturas: frontier = heap de (f, h, contador, nó) -> desempata pelo menor h
#             g_best   = melhor custo conhecido até cada nó
#             parent   = de onde veio cada nó (para reconstruir o caminho)
#             closed   = nós já expandidos (nunca se expande um nó duas vezes)
# Passos do ciclo:
#   1) tira da fronteira o nó com menor f;
#   2) se já foi expandido (entrada obsoleta), ignora-o;
#   3) marca-o como expandido; se é o objetivo, para;
#   4) para cada vizinho, se encontrou um caminho mais barato, atualiza e põe na fronteira;
#   5) se a fronteira esvaziar, não há caminho (é assim que o cenário C termina).
def search(grid: Grid, moves: int = 4, heuristic: Heuristic | None = None,
           weight: float = 1.0, name: str | None = None) -> SearchResult:
    """Pesquisa de custo mínimo de grid.start a grid.goal."""
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


# dijkstra
# O que faz: Dijkstra = a mesma pesquisa sem heurística.
# Recebe:    a grelha e 4 ou 8 direções.  Devolve: um SearchResult.
def dijkstra(grid: Grid, moves: int = 4) -> SearchResult:
    return search(grid, moves, heuristic=None)


# astar
# O que faz: A* = a mesma pesquisa com heurística (e peso opcional, para o Weighted A*).
#            Se não indicares heurística, usa Manhattan em 4 direções e Octile em 8.
# Recebe:    a grelha, 4 ou 8 direções, a heurística, o peso e um nome opcional.
# Devolve:   um SearchResult.
def astar(grid: Grid, moves: int = 4, heuristic: Heuristic | None = None,
          weight: float = 1.0, heuristic_name: str | None = None) -> SearchResult:
    from .heuristics import manhattan, octile
    if heuristic is None:
        heuristic = manhattan if moves == 4 else octile
    res = search(grid, moves, heuristic=heuristic, weight=weight)
    if heuristic_name:
        res.algorithm += f" [{heuristic_name}]"
    return res