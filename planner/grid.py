"""Modelo da grelha 2D: leitura de cenários, custos de terreno e movimentos (secções 3.1, 3.2, 4.1).

Convenções
----------
* Uma posição é um tuplo (linha, coluna).
* Custo de um movimento = custo de entrada da célula de destino.
* 4 direções: cima, baixo, esquerda, direita.
* 8 direções: acrescenta diagonais, com custo = custo_destino * sqrt(2); a diagonal só é
  permitida se as DUAS células ortogonais adjacentes estiverem livres (não se corta esquinas).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

SQRT2 = math.sqrt(2)

OBSTACLE = "#"

# (dr, dc)
ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass
class Grid:
    """Grelha de custos. `cost[r][c]` é o custo de entrada, ou None se a célula for obstáculo."""

    cost: list[list[int | None]]
    start: tuple[int, int]
    goal: tuple[int, int]
    name: str = ""
    rows: int = field(init=False)
    cols: int = field(init=False)

    def __post_init__(self) -> None:
        self.rows = len(self.cost)
        self.cols = len(self.cost[0]) if self.rows else 0

    # ------------------------------------------------------------------ leitura
    @classmethod
    def from_text(cls, text: str, name: str = "") -> "Grid":
        """Constrói a grelha a partir do texto do cenário (legenda da secção 3.2)."""
        # splitlines() trata de \n, \r\n (ficheiros criados no Windows) e \r.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("cenário vazio")
        width = len(lines[0])
        start = goal = None
        cost: list[list[int | None]] = []
        for r, line in enumerate(lines):
            if len(line) != width:
                raise ValueError(f"linha {r + 1} tem {len(line)} colunas; esperado {width}")
            row: list[int | None] = []
            for c, ch in enumerate(line):
                if ch == "S":
                    if start is not None:
                        raise ValueError("mais do que uma célula inicial 'S'")
                    start = (r, c)
                    row.append(1)
                elif ch == "G":
                    if goal is not None:
                        raise ValueError("mais do que uma célula objetivo 'G'")
                    goal = (r, c)
                    row.append(1)
                elif ch == ".":
                    row.append(1)
                elif ch in "23456789":
                    row.append(int(ch))
                elif ch == OBSTACLE:
                    row.append(None)
                else:
                    raise ValueError(f"símbolo inválido {ch!r} em ({r}, {c})")
            cost.append(row)
        if start is None or goal is None:
            raise ValueError("o cenário tem de ter exatamente um 'S' e um 'G'")
        return cls(cost, start, goal, name=name)

    @classmethod
    def from_file(cls, path: str | Path) -> "Grid":
        path = Path(path)
        return cls.from_text(path.read_text(encoding="utf-8"), name=path.stem)

    def to_text(self) -> str:
        """Inverso de from_text (útil para guardar cenários gerados)."""
        out = []
        for r in range(self.rows):
            chars = []
            for c in range(self.cols):
                pos = (r, c)
                v = self.cost[r][c]
                if pos == self.start:
                    chars.append("S")
                elif pos == self.goal:
                    chars.append("G")
                elif v is None:
                    chars.append(OBSTACLE)
                elif v == 1:
                    chars.append(".")
                else:
                    chars.append(str(v))
            out.append("".join(chars))
        return "\n".join(out) + "\n"

    # ------------------------------------------------------------------ consultas
    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.cost[r][c] is not None

    def neighbors(self, pos: tuple[int, int], moves: int = 4):
        """Gera (vizinho, custo_do_movimento) para o modelo de movimento `moves` (4 ou 8)."""
        if moves not in (4, 8):
            raise ValueError("moves tem de ser 4 ou 8")
        r, c = pos
        for dr, dc in ORTHOGONAL:
            nr, nc = r + dr, c + dc
            if self.is_free(nr, nc):
                yield (nr, nc), float(self.cost[nr][nc])
        if moves == 8:
            for dr, dc in DIAGONAL:
                nr, nc = r + dr, c + dc
                if not self.is_free(nr, nc):
                    continue
                # Sem cortar esquinas: as duas células ortogonais têm de estar livres.
                if self.is_free(r + dr, c) and self.is_free(r, c + dc):
                    yield (nr, nc), self.cost[nr][nc] * SQRT2

    def path_cost(self, path: list[tuple[int, int]], moves: int = 8) -> float:
        """Recalcula o custo de um caminho, validando cada passo (útil para testes)."""
        total = 0.0
        for a, b in zip(path, path[1:]):
            for nb, step in self.neighbors(a, moves):
                if nb == b:
                    total += step
                    break
            else:
                raise ValueError(f"passo inválido {a} -> {b} no modelo de {moves} direções")
        return total
