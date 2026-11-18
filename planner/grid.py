"""Modelo da grelha 2D: leitura de cenários, custos de terreno e movimentos."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

# Constantes: raiz de 2 (custo das diagonais) e o símbolo de obstáculo nos ficheiros
SQRT2 = math.sqrt(2)

OBSTACLE = "#"

# Movimentos como (mudança na linha, mudança na coluna)
ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


# CLASSE Grid
# O que guarda: a matriz de custos (None = obstáculo), a posição de S (start),
#               a posição de G (goal) e o nome do cenário.
# O @dataclass cria o __init__ sozinho a partir dos campos.
@dataclass
class Grid:
    """Grelha de custos. `cost[r][c]` é o custo de entrada, ou None se a célula for obstáculo."""

    cost: list[list[int | None]]
    start: tuple[int, int]
    goal: tuple[int, int]
    name: str = ""
    rows: int = field(init=False)
    cols: int = field(init=False)

    # __post_init__
    # O que faz: corre logo depois de criar a Grid e calcula o nº de linhas e de colunas.
    def __post_init__(self) -> None:
        self.rows = len(self.cost)
        self.cols = len(self.cost[0]) if self.rows else 0

    # from_text
    # O que faz: transforma o texto de um cenário numa Grid e valida-o
    #            (linhas do mesmo tamanho, um só S, um só G, símbolos válidos).
    # Recebe:    o texto do cenário e, opcionalmente, um nome.
    # Devolve:   uma Grid.
    # Símbolos:  S e G custam 1, "." custa 1, 2 a 9 são custos, "#" vira None.
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

    # from_file
    # O que faz: lê um ficheiro de texto e chama o from_text.
    # Recebe:    o caminho do ficheiro.
    # Devolve:   uma Grid (o nome da Grid é o nome do ficheiro).
    @classmethod
    def from_file(cls, path: str | Path) -> "Grid":
        path = Path(path)
        return cls.from_text(path.read_text(encoding="utf-8"), name=path.stem)

    # to_text
    # O que faz: o inverso do from_text (Grid para texto).
    # Devolve:   o texto do cenário, com S, G, ".", dígitos e "#".
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

    # in_bounds
    # O que faz: diz se a posição está dentro da grelha (não sai pelas margens).
    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    # is_free
    # O que faz: diz se dá para pisar a célula (dentro da grelha e não é obstáculo).
    def is_free(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.cost[r][c] is not None

    # neighbors  (a função mais importante: é o que a pesquisa chama a cada expansão)
    # O que faz: gera os vizinhos de uma célula e o custo de ir até lá.
    # Recebe:    a posição e o modelo de movimento (4 ou 8 direções).
    # Devolve:   pares (vizinho, custo), um de cada vez (yield).
    # Passos:    1) ortogonais: custo = custo da célula de DESTINO.
    #            2) só em 8 direções, diagonais: custo = custo de destino x raiz de 2,
    #               e só se as DUAS células ortogonais adjacentes estiverem livres
    #               (não se corta esquinas).
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

    # path_cost
    # O que faz: recalcula o custo de um caminho, validando cada passo.
    #            Se algum passo não for um movimento possível, dá erro.
    # Recebe:    a lista de posições do caminho e o modelo de movimento.
    # Devolve:   o custo total.
    # Para quê:  confirmar nos testes e na célula 18 que o custo do caminho está certo.
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
        """Recalcula o custo de um caminho, validando cada passo (útil para testes)."""
        total = 0.0
        for a, b in zip(path, path[1:]):             # pares de células consecutivas: (a, b)
            for nb, step in self.neighbors(a, moves):
                if nb == b:                          # b é mesmo vizinho de a
                    total += step                    # soma o custo desse passo
                    break
            else:
                # o "else" do for corre quando o ciclo acaba SEM break:
                # b não é vizinho de a, logo o caminho tem um passo impossível
                raise ValueError(f"passo inválido {a} -> {b} no modelo de {moves} direções")
        return total