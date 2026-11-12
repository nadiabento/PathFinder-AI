# PathFinder-AI - Planeamento de Trajetos

Trabalho 2 de Inteligencia Artificial (UC 90498), ESTGA, Universidade de Aveiro.

Planeador de trajetos em grelha 2D com Dijkstra, A* e Weighted A*, em 4 e 8
direcoes, com varias heuristicas e 6 cenarios de teste.

Autores: Nome 1 (n. 00000) e Nome 2 (n. 00000)

## REQUISITOS

- Python 3.9 ou superior (testado com 3.14)
- Bibliotecas: numpy, pandas, matplotlib e jupyter

Instalacao (na raiz do projeto):

    python -m venv .venv
    .venv\Scripts\activate          (Windows)
    source .venv/bin/activate       (Linux / macOS)
    pip install -r requirements.txt

## COMO REPRODUZIR OS RESULTADOS

1. Abrir notebooks/experiments.ipynb (no VS Code ou no Jupyter).
2. Executar tudo: Restart & Run All.

O notebook usa caminhos relativos, por isso deve correr a partir da pasta
notebooks (e o que acontece por omissao). Ao correr:

- regenera os 3 cenarios do grupo com a semente 2026;
- guarda as tabelas em results/tables e os graficos em results/figures.

## TESTES

Na raiz do projeto:

    python -m tests.test_anexo_a
    python -m tests.test_heuristics

Cada teste imprime "ok" quando passa.

## CENARIOS

Formato: um ficheiro de texto, uma linha por linha da grelha.

    S = inicio   G = objetivo   . = custo 1   2 a 9 = custo   # = obstaculo

- scenarios/anexo_a : os 3 cenarios do enunciado (A, B e C)
- scenarios/custom : os 3 cenarios do grupo (labirinto 51x51, espaco aberto
  100x100 e custos variados), gerados com a semente 2026

Para regenerar os cenarios do grupo (na raiz do projeto):

    python -m planner.generators

## ESTRUTURA

PathFinder-AI/
+-- README.md
+-- requirements.txt
+-- planner/
| +-- grid.py modelo da grelha e leitura de cenarios
| +-- search.py Dijkstra, A* e Weighted A*
| +-- heuristics.py Manhattan, Euclidiana, Chebyshev e Octile
| +-- generators.py geradores de cenarios (semente fixa)
| +-- experiments.py execucao das experiencias e metricas
| +-- visualize.py mapas e graficos
+-- scenarios/ cenarios de teste (anexo_a e custom)
+-- tests/ testes do Anexo A e das heuristicas
+-- notebooks/ experiments.ipynb (experiencias e resultados)
+-- results/ tables (CSV) e figures (PNG)
+-- report/ mini-relatorio e declaracao de uso de IA

## EXTENSAO

Weighted A\*: estudo do compromisso entre qualidade do caminho e rapidez ao
variar o peso da heuristica (seccao 5 do notebook).

## DECLARACAO DE USO DE IA GENERATIVA

Ver o mini-relatorio.
