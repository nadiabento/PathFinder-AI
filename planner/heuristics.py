"""Heurísticas h(n) = estimativa do custo restante de n até ao objetivo (secção 4.2 e 4.3).

Todas assumem que o custo mínimo de entrada numa célula é 1 (garantido pelo formato dos
cenários: '.', 'S', 'G' custam 1 e os terrenos custam 2..9). Isto é o que as torna admissíveis
em grelhas com terrenos variados: nunca sobrestimam, porque cada passo custa pelo menos 1
(ou sqrt(2) na diagonal).

Resumo da justificação (ver também tests/test_heuristics.py, que verifica numericamente):

+-------------+---------------------------+---------------------------------------------------+
| Heurística  | 4 direções                | 8 direções                                        |
+-------------+---------------------------+---------------------------------------------------+
| Manhattan   | admissível e consistente  | NÃO admissível (diagonal custa sqrt2 mas reduz 2) |
| Euclidiana  | admissível e consistente  | admissível e consistente (mas pouco informada)    |
| Chebyshev   | admissível e consistente  | admissível e consistente (mas pouco informada)    |
| Octile      | admissível (fraca)        | admissível e consistente; a mais justa (extra)    |
+-------------+---------------------------+---------------------------------------------------+
"""
from __future__ import annotations

import math

Pos = tuple[int, int]
SQRT2 = math.sqrt(2)


def manhattan(a: Pos, b: Pos) -> float:
    """|dr| + |dc|.

    4 direções: cada passo custa >= 1 e reduz a distância de Manhattan em exatamente 1 (no máximo),
    logo h(n) <= custo real. Consistente: h(n) <= c(n,n') + h(n') pois |h(n)-h(n')| <= 1 <= c(n,n').
    8 direções: um passo diagonal custa >= sqrt2 ~ 1.41 mas reduz Manhattan em 2 -> sobrestima
    (ex.: dois passos na diagonal custam 2*sqrt2 ~ 2.83 < 4 = Manhattan(...)). NÃO é admissível.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: Pos, b: Pos) -> float:
    """Distância em linha reta.

    Qualquer caminho na grelha é uma linha poligonal entre a e b, logo o seu comprimento é
    >= distância em linha reta; como cada passo custa >= o seu comprimento (1 ou sqrt2, vezes
    um custo de terreno >= 1), h <= custo real, em 4 e em 8 direções. É consistente pela
    desigualdade triangular. Em 4 direções é mais fraca (informa menos) do que Manhattan.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev(a: Pos, b: Pos) -> float:
    """max(|dr|, |dc|).

    8 direções: cada passo reduz Chebyshev em no máximo 1 e custa >= 1 -> admissível e
    consistente. Em 4 direções também é admissível (Chebyshev <= Manhattan <= custo real),
    mas ainda mais fraca. É a heurística "natural" de 8 direções se os passos custassem 1.
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def octile(a: Pos, b: Pos) -> float:
    """max + (sqrt2 - 1) * min: custo exato em 8 direções num mapa vazio com custos 1.

    Extra (não pedida no enunciado): a heurística admissível mais informada para 8 direções.
    Admissível e consistente em 8 direções. Em 4 direções continua admissível (<= Manhattan),
    mas é mais fraca que Manhattan.
    """
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) + (SQRT2 - 1) * min(dr, dc)


HEURISTICS = {
    "manhattan": manhattan,
    "euclidiana": euclidean,
    "chebyshev": chebyshev,
    "octile": octile,
}

# Para os testes e para o relatório: em que modelos é admissível (assumindo custo mínimo 1).
ADMISSIBLE_IN = {
    "manhattan": {4},
    "euclidiana": {4, 8},
    "chebyshev": {4, 8},
    "octile": {4, 8},
}
