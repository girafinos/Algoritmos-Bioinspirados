# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# Algoritmo Genético para o Problema do Caixeiro Viajante (TSP)
# -----------------------------------------------------------------------------------

import random
import matplotlib.pyplot as plt
import yaml

# ---------------- Parâmetros ----------------

POP_SIZE = 20
GENERATIONS = 300
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.9
ELITISM = POP_SIZE // 10
TOURNAMENT_SIZE = POP_SIZE // 5

# Escolha: "OX" ou "PMX"
CROSSOVER_OPERATOR = "OX"

# random.seed(42)

# ---------------- Dados da instância ----------------
# Exemplo pequeno para testar.
# Depois você troca pela matriz da LAU15 ou SGB128.

def load_config(config_path="train_config.yaml"):
    """
    Loads the training configuration from the YAML file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

grid_search = config["grid_search"]

# POP_SIZE = config.get("POP_SIZE", 20)
# GENERATIONS = config.get("GENERATIONS", 300)
# MUTATION_RATE = config.get("MUTATION_RATE", 0.3)
# CROSSOVER_OPERATOR = config.get("CROSSOVER_OPERATOR", "OX")

def ler_matriz_distancias(caminho_arquivo):
    matriz = []

    with open(caminho_arquivo, 'r') as f:
        for linha in f:
            linha = linha.strip()

            if not linha:
                continue

            if linha.startswith('#'):
                continue

            valores = list(map(int, linha.split()))

            matriz.append(valores)

    return matriz

MATRIZ_DISTANCIAS = ler_matriz_distancias("Algoritmos-Bioinspirados/TP1/matriz_distancias.txt")
N_CIDADES = len(MATRIZ_DISTANCIAS)

# ---------------- Funções do problema ----------------

def calcular_distancia_total(rota):
    distancia = 0

    for i in range(len(rota) - 1):
        cidade_atual = rota[i]
        proxima_cidade = rota[i + 1]
        distancia += MATRIZ_DISTANCIAS[cidade_atual][proxima_cidade]

    # retorna para a cidade inicial
    distancia += MATRIZ_DISTANCIAS[rota[-1]][rota[0]]

    return distancia


def rota_valida(rota):
    return len(rota) == N_CIDADES and len(set(rota)) == N_CIDADES


def fitness(individuo):
    return -calcular_distancia_total(individuo)

# ---------------- Funções do AG ----------------

def criar_individuo():
    individuo = list(range(N_CIDADES))
    random.shuffle(individuo)
    return individuo


def criar_populacao():
    return [criar_individuo() for _ in range(POP_SIZE)]


def torneio(populacao):
    candidatos = random.sample(populacao, TOURNAMENT_SIZE)
    return max(candidatos, key=fitness)

# ---------------- Crossover OX ----------------

def gerar_filho_ox(pai1, pai2):
    tamanho = len(pai1)
    filho = [None] * tamanho

    inicio, fim = sorted(random.sample(range(tamanho), 2))

    filho[inicio:fim + 1] = pai1[inicio:fim + 1]

    pos = (fim + 1) % tamanho

    for gene in pai2:
        if gene not in filho:
            filho[pos] = gene
            pos = (pos + 1) % tamanho

    return filho


def crossover_ox(pai1, pai2):
    filho1 = gerar_filho_ox(pai1, pai2)
    filho2 = gerar_filho_ox(pai2, pai1)

    return filho1, filho2

# ---------------- Crossover PMX ----------------

def gerar_filho_pmx(pai1, pai2):
    tamanho = len(pai1)
    filho = [None] * tamanho

    inicio, fim = sorted(random.sample(range(tamanho), 2))

    filho[inicio:fim + 1] = pai1[inicio:fim + 1]

    for i in range(inicio, fim + 1):
        gene_pai2 = pai2[i]

        if gene_pai2 not in filho:
            pos = i

            while True:
                gene_pai1 = pai1[pos]
                pos = pai2.index(gene_pai1)

                if filho[pos] is None:
                    filho[pos] = gene_pai2
                    break

    for i in range(tamanho):
        if filho[i] is None:
            filho[i] = pai2[i]

    return filho


def crossover_pmx(pai1, pai2):
    filho1 = gerar_filho_pmx(pai1, pai2)
    filho2 = gerar_filho_pmx(pai2, pai1)

    return filho1, filho2


def crossover(pai1, pai2):
    if random.random() > CROSSOVER_RATE:
        return pai1[:], pai2[:]

    if CROSSOVER_OPERATOR == "OX":
        return crossover_ox(pai1, pai2)

    elif CROSSOVER_OPERATOR == "PMX":
        return crossover_pmx(pai1, pai2)

    else:
        raise ValueError("Operador de crossover inválido. Use 'OX' ou 'PMX'.")

# ---------------- Mutação ----------------

def mutacao(individuo):
    filho = individuo[:]

    if random.random() < MUTATION_RATE:
        i, j = sorted(random.sample(range(N_CIDADES), 2))

        filho[i:j + 1] = reversed(filho[i:j + 1])

    return filho

# ---------------- Nova geração ----------------

def nova_geracao(populacao):
    populacao_ordenada = sorted(populacao, key=fitness, reverse=True)

    nova_pop = [ind[:] for ind in populacao_ordenada[:ELITISM]]

    while len(nova_pop) < POP_SIZE:
        pai1 = torneio(populacao)
        pai2 = torneio(populacao)

        filho1, filho2 = crossover(pai1, pai2)

        filho1 = mutacao(filho1)
        filho2 = mutacao(filho2)

        if not rota_valida(filho1):
            raise ValueError(f"Rota inválida gerada: {filho1}")

        if not rota_valida(filho2):
            raise ValueError(f"Rota inválida gerada: {filho2}")

        nova_pop.append(filho1)

        if len(nova_pop) < POP_SIZE:
            nova_pop.append(filho2)

    return nova_pop

# ---------------- Execução ----------------

populacao = criar_populacao()

melhores_distancias = []

for geracao in range(GENERATIONS):
    melhor = max(populacao, key=fitness)

    melhor_distancia = calcular_distancia_total(melhor)
    melhores_distancias.append(melhor_distancia)

    if geracao % 10 == 0:
        print(
            f"Geração {geracao}: "
            f"rota = {melhor}, "
            f"distância = {melhor_distancia}, "
            f"fitness = {fitness(melhor)}, "
            f"válida = {rota_valida(melhor)}"
        )

    populacao = nova_geracao(populacao)

# ---------------- Resultado final ----------------

melhor_final = max(populacao, key=fitness)
distancia_final = calcular_distancia_total(melhor_final)

print("\n--- Melhor solução encontrada ---")
print("Rota:", melhor_final)
print("Distância total:", distancia_final)
print("Rota válida:", rota_valida(melhor_final))

# ---------------- Gráfico de convergência ----------------

plt.figure(figsize=(10, 5))
plt.plot(melhores_distancias, linewidth=2, label="Melhor distância")

plt.xlabel("Geração")
plt.ylabel("Distância total")
plt.title(f"Convergência do AG para TSP - Operador {CROSSOVER_OPERATOR}")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("grafico_convergencia_tsp.png", dpi=300)
plt.close()

print("\nGráfico salvo com sucesso:")
print(" - grafico_convergencia_tsp.png")