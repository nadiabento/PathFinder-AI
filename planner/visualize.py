"""Desenha os mapas e os gráficos."""
import matplotlib.pyplot as plt
import numpy as np

def _matriz(grid):
    """Converte o Grid numa matriz numpy: 0 = obstáculo, 1..9 = custo."""
    return np.array([[0 if v is None else v for v in linha] for linha in grid.cost])

def mostrar(grid, caminho, explorados, inicio, fim, titulo="", ax=None):
    """Mapa com: custos (castanho), explorados (azul), caminho (vermelho)."""
    mapa = _matriz(grid)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.YlOrBr.copy()
    cmap.set_bad("black")                        # obstáculos a preto
    ax.imshow(np.ma.masked_where(mapa == 0, mapa), cmap=cmap, vmin=1, vmax=9)
    if explorados:
        m = np.zeros(mapa.shape)                 # marca os nós explorados
        for r, c in explorados:
            m[r, c] = 1
        ax.imshow(np.ma.masked_where(m == 0, m), cmap="Blues", alpha=0.5, vmin=0, vmax=1.5)
    if caminho:
        ax.plot([p[1] for p in caminho], [p[0] for p in caminho], color="red", lw=2)
    ax.plot(inicio[1], inicio[0], "go", ms=8)    # S
    ax.plot(fim[1], fim[0], "r*", ms=12)         # G
    ax.set_title(titulo, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def grafico_expandidos(df, direcoes=4, ax=None):
    """Barras com os nós expandidos por cenário (escala log)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    d = df[(df.direcoes == direcoes) & (df.w == 1)].copy()   # só versões exatas
    # nome da barra: "Dijkstra" ou "A* (Manhattan)"
    d["config"] = d.apply(lambda x: x.algoritmo if x.heuristica == "-"
                          else f"{x.algoritmo} ({x.heuristica})", axis=1)
    d.pivot(index="cenario", columns="config", values="expandidos").plot.bar(ax=ax, logy=True)
    ax.set_ylabel("nós expandidos (log)")
    ax.set_title(f"Nós expandidos, {direcoes} direções")
    ax.tick_params(axis="x", rotation=30)
    return ax


def grafico_peso(df, cenario, heuristica, direcoes=4):
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