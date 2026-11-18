"""Desenha os mapas e os gráficos."""
import matplotlib.pyplot as plt
import numpy as np


# _matriz
# O que faz: converte a Grid (com None nos obstáculos) numa matriz numpy
#            (0 = obstáculo, 1..9 = custo), que é o que o matplotlib desenha.
def _matriz(grid):
    return np.array([[0 if v is None else v for v in linha] for linha in grid.cost])


# mostrar  (células 5, 16, 18 e 21)
# O que faz: desenha o mapa com várias camadas, de baixo para cima:
#   1) o mapa de custos (castanho mais escuro = terreno mais caro; obstáculos a preto);
#   2) os nós explorados (azul semitransparente);
#   3) o caminho encontrado (linha vermelha);
#   4) o início (círculo verde) e o objetivo (estrela vermelha).
# Recebe:  a Grid, o caminho, os nós explorados, o início, o fim, um título e (opcional)
#          um "ax", para poder pôr vários mapas lado a lado.
# Se caminho ou explorados forem None, essa camada não é desenhada (célula 5 só mostra o mapa).
def mostrar(grid, caminho, explorados, inicio, fim, titulo="", ax=None):
    mapa = _matriz(grid)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.YlOrBr.copy()
    cmap.set_bad("black")
    ax.imshow(np.ma.masked_where(mapa == 0, mapa), cmap=cmap, vmin=1, vmax=9)
    if explorados:
        m = np.zeros(mapa.shape)
        for r, c in explorados:
            m[r, c] = 1
        ax.imshow(np.ma.masked_where(m == 0, m), cmap="Blues", alpha=0.5, vmin=0, vmax=1.5)
    if caminho:
        ax.plot([p[1] for p in caminho], [p[0] for p in caminho], color="red", lw=2)
    ax.plot(inicio[1], inicio[0], "go", ms=8)
    ax.plot(fim[1], fim[0], "r*", ms=12)
    ax.set_title(titulo, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


# grafico_expandidos  (célula 14)
# O que faz: gráfico de barras com os nós expandidos por cenário e por configuração.
# Usa escala logarítmica porque os cenários têm tamanhos muito diferentes.
# Só usa as versões exatas (w = 1); o efeito do peso vê-se no grafico_peso.
# Recebe: a tabela df e o nº de direções.  Devolve: o ax do gráfico.
def grafico_expandidos(df, direcoes=4, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    d = df[(df.direcoes == direcoes) & (df.w == 1)].copy()
    d["config"] = d.apply(lambda x: x.algoritmo if x.heuristica == "-"
                          else f"{x.algoritmo} ({x.heuristica})", axis=1)
    d.pivot(index="cenario", columns="config", values="expandidos").plot.bar(ax=ax, logy=True)
    ax.set_ylabel("nós expandidos (log)")
    ax.set_title(f"Nós expandidos, {direcoes} direções")
    ax.tick_params(axis="x", rotation=30)
    return ax


# grafico_peso  (célula 24)
# O que faz: mostra o compromisso do Weighted A* num cenário, com dois eixos verticais:
#            azul = nós expandidos (quanto menos, mais rápido);
#            vermelho = desvio ao ótimo em % (quanto mais, pior o caminho).
# Recebe: a tabela df_w, o nome do cenário, a heurística e o nº de direções.
# Devolve: a figura (para poder guardá-la com savefig).
def grafico_peso(df, cenario, heuristica, direcoes=4):
    d = df[(df.cenario == cenario) & (df.direcoes == direcoes)
           & (df.algoritmo == "A*") & (df.heuristica == heuristica)].sort_values("w")
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(d.w, d.expandidos, "o-", color="tab:blue")
    ax1.set_xlabel("peso w")
    ax1.set_ylabel("nós expandidos", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(d.w, d.gap_pct, "s--", color="tab:red")
    ax2.set_ylabel("desvio ao ótimo (%)", color="tab:red")
    ax1.set_title(f"Weighted A*: {cenario} ({direcoes} dir.)")
    fig.tight_layout()
    return fig
    """Weighted A*: nós expandidos (azul) vs desvio ao ótimo (vermelho)."""
    d = df[(df.cenario == cenario) & (df.direcoes == direcoes)
           & (df.algoritmo == "A*") & (df.heuristica == heuristica)].sort_values("w")
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(d.w, d.expandidos, "o-", color="tab:blue")
    ax1.set_xlabel("peso w")
    ax1.set_ylabel("nós expandidos", color="tab:blue")
    ax2 = ax1.twinx()                            # segundo eixo vertical
    ax2.plot(d.w, d.gap_pct, "s--", color="tab:red")
    ax2.set_ylabel("desvio ao ótimo (%)", color="tab:red")
    ax1.set_title(f"Weighted A*: {cenario} ({direcoes} dir.)")
    fig.tight_layout()
    return fig