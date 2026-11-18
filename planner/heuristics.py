"""Heurísticas h(n): estimativa do custo que falta de n até ao objetivo.
Todas assumem que cada célula custa pelo menos 1 (por isso nunca sobrestimam)."""
from __future__ import annotations

import math

Pos = tuple[int, int]
SQRT2 = math.sqrt(2)


# manhattan
# O que faz: soma das diferenças em linhas e em colunas.
# 4 direções: admissível e consistente (cada passo reduz no máximo 1 e custa >= 1).
# 8 direções: NÃO admissível (uma diagonal reduz 2, mas custa só cerca de 1,41).
def manhattan(a: Pos, b: Pos) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# euclidean
# O que faz: distância em linha reta.
# Admissível em 4 e em 8 direções (nenhum caminho é mais curto do que a linha reta).
# Em 4 direções é fraca: subestima muito.
def euclidean(a: Pos, b: Pos) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# chebyshev
# O que faz: a maior das duas diferenças (em linhas ou em colunas).
# Admissível em 4 e em 8 direções (cada passo reduz no máximo 1 e custa >= 1).
# É fraca porque subestima.
def chebyshev(a: Pos, b: Pos) -> float:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


# octile 
# O que faz: custo exato em 8 direções num mapa vazio com custo 1.
#            Faz as diagonais possíveis (min) e o resto em linha reta (max - min).
#            A conta dá: max + (raiz de 2 - 1) * min.
# Admissível e consistente em 8 direções. É a mais informada entre as admissíveis.
def octile(a: Pos, b: Pos) -> float:
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) + (SQRT2 - 1) * min(dr, dc)


# HEURISTICS: nome -> função (usado pelos testes)
HEURISTICS = {
    "manhattan": manhattan,
    "euclidiana": euclidean,
    "chebyshev": chebyshev,
    "octile": octile,
}

# ADMISSIBLE_IN: em que modelos (4 ou 8) cada heurística é admissível.
# Os testes só verificam a admissibilidade onde ela deve valer.
ADMISSIBLE_IN = {
    "manhattan": {4},
    "euclidiana": {4, 8},
    "chebyshev": {4, 8},
    "octile": {4, 8},
}