"""Desenha os mapas e os gráficos."""
import matplotlib.pyplot as plt
import numpy as np

def _matriz(grid):
    """Converte o Grid numa matriz numpy: 0 = obstáculo, 1..9 = custo."""
    return np.array([[0 if v is None else v for v in linha] for linha in grid.cost])

COR_S = "#2e9e5b"      # verde: início (S)
COR_G = "#7b2cbf"      # roxo: objetivo (G)
COR_OBSTACULO = "#2b2f3a"


def _marca(ax, pos, letra, cor, tam):
    """Desenha a letra S ou G dentro de uma caixa colorida, na célula pos."""
    ax.text(pos[1], pos[0], letra, ha="center", va="center", fontsize=tam,
            fontweight="bold", color="white", zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc=cor, ec="white", lw=1.2))


def mostrar(grid, caminho, explorados, inicio, fim, titulo="", ax=None):
    """Mapa com: custos (castanho), obstáculos (cinzento escuro), explorados (azul),
    caminho (vermelho) e as letras S (início) e G (objetivo)."""
    mapa = _matriz(grid)
    R, C = mapa.shape
    grande = max(R, C)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.YlOrBr.copy()
    cmap.set_bad(COR_OBSTACULO)                  # obstáculos
    ax.imshow(np.ma.masked_where(mapa == 0, mapa), cmap=cmap, vmin=0, vmax=9)
    if explorados:
        m = np.zeros(mapa.shape)                 # marca os nós explorados
        for r, c in explorados:
            m[r, c] = 1
        ax.imshow(np.ma.masked_where(m == 0, m), cmap="Blues", alpha=0.5, vmin=0, vmax=1.5)
    if grande <= 15:                             # mapas pequenos: grelha e custos visíveis
        ax.set_xticks(np.arange(-0.5, C, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, R, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        for r in range(R):
            for c in range(C):
                if mapa[r, c] > 1:
                    ax.text(c, r, int(mapa[r, c]), ha="center", va="center",
                            fontsize=11, fontweight="bold",
                            color="white" if mapa[r, c] >= 6 else "#3b2300")
    ax.tick_params(which="minor", length=0)
    if caminho:
        ax.plot([p[1] for p in caminho], [p[0] for p in caminho], color="red", lw=2, zorder=4)
    tam = 15 if grande <= 15 else 11 if grande <= 60 else 9
    _marca(ax, inicio, "S", COR_S, tam)
    _marca(ax, fim, "G", COR_G, tam)
    ax.set_title(titulo, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if grande <= 15:                             # voltar a pôr a grelha só nas células
        ax.set_xticks(np.arange(-0.5, C, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, R, 1), minor=True)
    return ax


def mostrar_cenarios(cenarios, colunas=3):
    """Painel com todos os cenários, com legenda comum: S, G, obstáculo e escala de custos."""
    from matplotlib.patches import Patch
    n = len(cenarios)
    linhas = -(-n // colunas)
    fig, axs = plt.subplots(linhas, colunas, figsize=(4.6 * colunas, 4.9 * linhas))
    axs = np.array(axs).ravel()
    for ax, (nome, g) in zip(axs, cenarios.items()):
        mostrar(g, None, None, g.start, g.goal, f"{nome}  ({g.rows}×{g.cols})", ax)
    for ax in axs[n:]:
        ax.axis("off")
    cmap = plt.cm.YlOrBr
    legenda = [
        Patch(fc=COR_S, ec="white", label="S = início"),
        Patch(fc=COR_G, ec="white", label="G = objetivo"),
        Patch(fc=COR_OBSTACULO, label="Obstáculo (não se atravessa)"),
        Patch(fc=cmap(1 / 9), label="Custo 1 (livre)"),
        Patch(fc=cmap(5 / 9), label="Custo médio (3 a 5)"),
        Patch(fc=cmap(9 / 9), label="Custo 9 (muito caro)"),
    ]
    fig.legend(handles=legenda, loc="lower center", ncol=6, frameon=False, fontsize=10)
    fig.suptitle("Os 6 cenários de teste", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


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