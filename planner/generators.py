"""Geradores de cenários com semente fixa.
Grelha: 0 = obstáculo, 1..9 = custo da célula. Posição = (linha, coluna)."""
import random
from collections import deque
from pathlib import Path

import numpy as np


def _resolvivel(grid, s, g):
    """Diz se existe caminho de s até g (pesquisa simples em largura)."""
    R, C = grid.shape
    vistos, fila = {s}, deque([s])
    while fila:
        r, c = fila.popleft()
        if (r, c) == g:
            return True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            # só avança se estiver dentro da grelha, não for parede e for nova
            if 0 <= n[0] < R and 0 <= n[1] < C and grid[n] != 0 and n not in vistos:
                vistos.add(n)
                fila.append(n)
    return False


def labirinto(linhas=51, colunas=51, seed=42, lacunas=0.05):
    """Labirinto grande. Começa tudo parede e vai escavando corredores."""
    rng = random.Random(seed)
    linhas |= 1                      # dimensões ímpares
    colunas |= 1
    grid = np.zeros((linhas, colunas), dtype=int)   # tudo parede
    inicio = (1, 1)
    grid[inicio] = 1
    pilha = [inicio]
    while pilha:
        r, c = pilha[-1]
        # células a 2 de distância ainda por escavar
        viz = [(r + dr, c + dc, dr, dc)
               for dr, dc in ((2, 0), (-2, 0), (0, 2), (0, -2))
               if 0 < r + dr < linhas - 1 and 0 < c + dc < colunas - 1
               and grid[r + dr, c + dc] == 0]
        if not viz:                  # sem saída: volta atrás
            pilha.pop()
            continue
        nr, nc, dr, dc = rng.choice(viz)
        grid[r + dr // 2, c + dc // 2] = 1   # abre a parede do meio
        grid[nr, nc] = 1                     # escava a nova célula
        pilha.append((nr, nc))
    # abre algumas paredes para haver caminhos alternativos
    for r in range(1, linhas - 1):
        for c in range(1, colunas - 1):
            if (r + c) % 2 == 1 and grid[r, c] == 0 and rng.random() < lacunas:
                grid[r, c] = 1
    return grid, inicio, (linhas - 2, colunas - 2)   # G no canto oposto


def aberto(linhas=100, colunas=100, densidade=0.12, seed=42):
    """Espaço grande com obstáculos ao acaso (para testar escalabilidade)."""
    rng = random.Random(seed)
    s, g = (0, 0), (linhas - 1, colunas - 1)         # cantos opostos
    while True:
        grid = np.ones((linhas, colunas), dtype=int)
        for r in range(linhas):
            for c in range(colunas):
                if rng.random() < densidade:         # sorteia obstáculos
                    grid[r, c] = 0
        grid[s] = grid[g] = 1                        # S e G livres
        if _resolvivel(grid, s, g):                  # repete se não houver caminho
            return grid, s, g


def custos_variados(linhas=30, colunas=40, seed=42):
    """Mapa com terrenos diferentes. Tem uma barreira cara (custo 9) com uma
    passagem barata: o caminho com menos passos NÃO é o mais barato."""
    rng = random.Random(seed)
    meio = linhas // 2
    s, g = (meio, 0), (meio, colunas - 1)    # S à esquerda, G à direita
    cb = colunas // 2 - 2                    # coluna onde começa a barreira
    while True:
        grid = np.ones((linhas, colunas), dtype=int)
        for _ in range(12):                  # manchas de terreno com custo 2 a 5
            r0, c0 = rng.randrange(linhas - 6), rng.randrange(colunas - 8)
            grid[r0:r0 + rng.randint(3, 6), c0:c0 + rng.randint(4, 8)] = rng.choice([2, 3, 4, 5])
        for r in range(linhas):              # poucos obstáculos (4%)
            for c in range(colunas):
                if rng.random() < 0.04:
                    grid[r, c] = 0
        grid[:, cb:cb + 4] = 9                      # barreira cara
        grid[meio + 4:meio + 6, cb:cb + 4] = 1      # passagem barata
        grid[s] = grid[g] = 1
        if _resolvivel(grid, s, g):
            return grid, s, g


def guardar(caminho, grid, s, g):
    """Guarda o cenário em texto (S, G, '.', 2-9, '#')."""
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
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)   # cria a pasta
    Path(caminho).write_text("\n".join(linhas) + "\n", encoding="utf-8")


def gerar_todos(pasta="scenarios/custom", seed=2026):
    """Gera os 3 cenários do grupo. Mesma semente = mesmos ficheiros."""
    guardar(f"{pasta}/labirinto_51x51.txt", *labirinto(51, 51, seed))
    guardar(f"{pasta}/aberto_100x100.txt", *aberto(100, 100, 0.12, seed))
    guardar(f"{pasta}/custos_variados.txt", *custos_variados(30, 40, seed))


if __name__ == "__main__":
    gerar_todos()