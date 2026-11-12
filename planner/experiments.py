"""Corre as experiências e guarda as métricas numa tabela."""
import math
import time

import pandas as pd


def correr(cenarios, configs, direcoes_lista=(4, 8), repeticoes=5):
    """Testa todas as configurações em todos os cenários.

    cenarios: {nome: Grid}
    configs:  lista de (algoritmo, heuristica, w, função)
              função(grid, direções) -> SearchResult do E1
              A primeira tem de ser o Dijkstra (serve de referência).
    Devolve uma tabela (DataFrame) com uma linha por teste.
    """
    linhas = []
    for nome, grid in cenarios.items():
        for d in direcoes_lista:                 # 4 e 8 direções
            ref = None                           # custo do Dijkstra
            for alg, heur, w, fn in configs:
                # repete várias vezes e faz a média do tempo
                tempos = []
                for _ in range(repeticoes):
                    t0 = time.perf_counter()
                    res = fn(grid, d)            # SearchResult
                    tempos.append(time.perf_counter() - t0)
                custo = res.cost
                if ref is None:
                    ref = custo                  # guarda o custo ótimo
                # é ótimo se o custo for igual ao do Dijkstra
                # (isclose evita erros de arredondamento com √2)
                ambos_inf = math.isinf(custo) and math.isinf(ref)
                otimo = ambos_inf or math.isclose(custo, ref, rel_tol=1e-9)
                # quanto % pior que o ótimo (vazio se não houver caminho)
                gap = (100 * (custo - ref) / ref
                       if math.isfinite(custo) and math.isfinite(ref) and ref > 0
                       else float("nan"))
                linhas.append({
                    "cenario": nome, "direcoes": d, "algoritmo": alg,
                    "heuristica": heur, "w": w,
                    "custo": round(custo, 4) if math.isfinite(custo) else float("inf"),
                    "expandidos": res.expanded,
                    "gerados": res.generated,            # memória (nós na fronteira)
                    "fronteira_max": res.max_frontier,   # memória (pico)
                    "tempo_ms": 1000 * sum(tempos) / len(tempos),   # média em ms
                    "passos": len(res.path) - 1 if res.found else None,
                    "otimo": otimo, "gap_pct": gap,
                })
    return pd.DataFrame(linhas)