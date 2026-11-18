"""Geradores de cenários com semente fixa.
Grelha: 0 = obstáculo, 1..9 = custo da célula. Posição = (linha, coluna)."""
import random
from collections import deque
from pathlib import Path

import numpy as np


# _resolvivel
# O que faz: diz se existe caminho de s até g (pesquisa em largura, 4 direções).
# Para quê:  garantir que os cenários gerados têm solução.
# Recebe:    a matriz, a posição inicial e a final.  Devolve: True ou False.
def _resolvivel(grid, s, g):
    R, C = grid.shape
    vistos, fila = {s}, deque([s])
    while fila:
        r, c = fila.popleft()
        if (r, c) == g:
            return True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < R and 0 <= n[1] < C and grid[n] != 0 and n not in vistos:
                vistos.add(n)
                fila.append(n)
    return False


# labirinto
# O que faz: cria um labirinto grande (51x51 por omissão) com a mesma semente = mesmo labirinto.
# Como funciona (DFS com retrocesso):
#   1) começa com tudo parede (0) e escava a célula (1,1);
#   2) a partir da célula atual, escolhe ao acaso uma vizinha a 2 de distância ainda por escavar,
#      abre a parede do meio e escava-a;
#   3) se não houver vizinhas livres (beco sem saída), volta atrás;
#   4) no fim abre 5% de paredes ao acaso ("lacunas") para haver caminhos alternativos.
# Recebe:  nº de linhas e colunas (forçadas a ímpares), a semente e a % de lacunas.
# Devolve: (matriz, início, objetivo). O objetivo fica no canto oposto.
def labirinto(linhas=51, colunas=51, seed=42, lacunas=0.05):
    rng = random.Random(seed)
    linhas |= 1
    colunas |= 1
    grid = np.zeros((linhas, colunas), dtype=int)
    inicio = (1, 1)
    grid[inicio] = 1
    pilha = [inicio]
    while pilha:
        r, c = pilha[-1]
        viz = [(r + dr, c + dc, dr, dc)
               for dr, dc in ((2, 0), (-2, 0), (0, 2), (0, -2))
               if 0 < r + dr < linhas - 1 and 0 < c + dc < colunas - 1
               and grid[r + dr, c + dc] == 0]
        if not viz:
            pilha.pop()
            continue
        nr, nc, dr, dc = rng.choice(viz)
        grid[r + dr // 2, c + dc // 2] = 1
        grid[nr, nc] = 1
        pilha.append((nr, nc))
    for r in range(1, linhas - 1):
        for c in range(1, colunas - 1):
            if (r + c) % 2 == 1 and grid[r, c] == 0 and rng.random() < lacunas:
                grid[r, c] = 1
    return grid, inicio, (linhas - 2, colunas - 2)


# aberto
# O que faz: cria um espaço grande (100x100) com 12% de obstáculos ao acaso, custo 1.
# Para quê:  testar a escalabilidade (poucos obstáculos e caminho longo).
# S e G ficam nos cantos opostos e nunca são obstáculos.
# Se o sorteio bloquear o caminho, sorteia outra vez até haver solução.
# Devolve: (matriz, início, objetivo).
def aberto(linhas=100, colunas=100, densidade=0.12, seed=42):
    rng = random.Random(seed)
    s, g = (0, 0), (linhas - 1, colunas - 1)
    while True:
        grid = np.ones((linhas, colunas), dtype=int)
        for r in range(linhas):
            for c in range(colunas):
                if rng.random() < densidade:
                    grid[r, c] = 0
        grid[s] = grid[g] = 1
        if _resolvivel(grid, s, g):
            return grid, s, g


# custos_variados
# O que faz: cria um mapa 30x40 em que o caminho com menos passos NÃO é o mais barato.
# Como funciona:
#   1) manchas de terreno com custo 2 a 5 e 4% de obstáculos;
#   2) uma barreira de custo 9 (4 colunas) a meio do mapa;
#   3) uma passagem barata (custo 1) um pouco abaixo da linha de S e G.
# Atravessar a barreira dá menos passos mas custa muito; contornar dá mais passos e custa menos.
# Devolve: (matriz, início, objetivo).
def custos_variados(linhas=30, colunas=40, seed=42):
    rng = random.Random(seed)
    meio = linhas // 2
    s, g = (meio, 0), (meio, colunas - 1)
    cb = colunas // 2 - 2
    while True:
        grid = np.ones((linhas, colunas), dtype=int)
        for _ in range(12):
            r0, c0 = rng.randrange(linhas - 6), rng.randrange(colunas - 8)
            grid[r0:r0 + rng.randint(3, 6), c0:c0 + rng.randint(4, 8)] = rng.choice([2, 3, 4, 5])
        for r in range(linhas):
            for c in range(colunas):
                if rng.random() < 0.04:
                    grid[r, c] = 0
        grid[:, cb:cb + 4] = 9
        grid[meio + 4:meio + 6, cb:cb + 4] = 1
        grid[s] = grid[g] = 1
        if _resolvivel(grid, s, g):
            return grid, s, g


# guardar
# O que faz: escreve um cenário num ficheiro de texto no formato do enunciado
#            (S, G, "." para custo 1, dígitos 2-9, "#" para obstáculo).
# Recebe:    o caminho do ficheiro, a matriz, a posição de S e a de G.
# Cria a pasta se ainda não existir.
def guardar(caminho, grid, s, g):
    linhas = []
    for r in range(grid.shape[0]):
        linha = ""
        for c in range(grid.shape[1]):
            v = grid[r, c]
            if (r, c) == s:
                linha += "S"
            elif (r, c) == g:
                linha += "G"
            elif v == 0:
                linha += "#"
            elif v == 1:
                linha += "."
            else:
                linha += str(v)
        linhas.append(linha)
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    Path(caminho).write_text("\n".join(linhas) + "\n", encoding="utf-8")


# gerar_todos
# O que faz: gera e guarda os 3 cenários do grupo (labirinto, aberto e custos variados).
# A semente fixa (2026) garante que dá sempre os mesmos ficheiros.
# É chamada na célula 4 do notebook.
def gerar_todos(pasta="scenarios/custom", seed=2026):
    guardar(f"{pasta}/labirinto_51x51.txt", *labirinto(51, 51, seed))
    guardar(f"{pasta}/aberto_100x100.txt", *aberto(100, 100, 0.12, seed))
    guardar(f"{pasta}/custos_variados.txt", *custos_variados(30, 40, seed))


if __name__ == "__main__":
    gerar_todos()