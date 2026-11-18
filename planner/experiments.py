"""Corre as experiências e guarda as métricas numa tabela."""
import math
import time

import pandas as pd


# correr  (é chamada nas células 9 e 23 do notebook)
# O que faz: testa todas as configurações em todos os cenários e devolve uma tabela.
# Recebe:
#   cenarios:       {nome: Grid}
#   configs:        lista de (algoritmo, heurística, w, função). A função recebe (grid, direções)
#                   e devolve um SearchResult. A PRIMEIRA tem de ser o Dijkstra,
#                   porque o seu custo é a referência para saber se os outros são ótimos.
#   direcoes_lista: 4 e/ou 8 direções.
#   repeticoes:     quantas vezes repete cada teste (o tempo é a média).
# Passos:
#   1) para cada cenário e cada nº de direções, corre cada configuração várias vezes;
#   2) mede o tempo de cada execução e faz a média;
#   3) compara o custo com o do Dijkstra (isclose evita erros de arredondamento com raiz de 2);
#   4) calcula o desvio ao ótimo em % (vazio se não houver caminho);
#   5) guarda uma linha na tabela com custo, nós expandidos, gerados, pico da fronteira,
#      tempo, passos, "é ótimo?" e desvio.
# Devolve: um DataFrame (o df do notebook).
def correr(cenarios, configs, direcoes_lista=(4, 8), repeticoes=5):
    linhas = []
    for nome, grid in cenarios.items():
        for d in direcoes_lista:
            ref = None
            for alg, heur, w, fn in configs:
                tempos = []
                for _ in range(repeticoes):
                    t0 = time.perf_counter()
                    res = fn(grid, d)
                    tempos.append(time.perf_counter() - t0)
                custo = res.cost
                if ref is None:
                    ref = custo
                ambos_inf = math.isinf(custo) and math.isinf(ref)
                otimo = ambos_inf or math.isclose(custo, ref, rel_tol=1e-9)
                gap = (100 * (custo - ref) / ref
                       if math.isfinite(custo) and math.isfinite(ref) and ref > 0
                       else float("nan"))
                linhas.append({
                    "cenario": nome, "direcoes": d, "algoritmo": alg,
                    "heuristica": heur, "w": w,
                    "custo": round(custo, 4) if math.isfinite(custo) else float("inf"),
                    "expandidos": res.expanded,
                    "gerados": res.generated,
                    "fronteira_max": res.max_frontier,
                    "tempo_ms": 1000 * sum(tempos) / len(tempos),
                    "passos": len(res.path) - 1 if res.found else None,
                    "otimo": otimo, "gap_pct": gap,
                })
    return pd.DataFrame(linhas)