# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# Algoritmo Genético para o Problema do Caixeiro Viajante (TSP)
# -----------------------------------------------------------------------------------

import random
import matplotlib.pyplot as plt
import yaml
import csv
import statistics
import os
from itertools import product

# ---------------- Parâmetros ----------------

def load_config(config_path="config.yaml"):
    """
    Loads the training configuration from the YAML file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

grid_search = config["grid_search"]
basic_config = config["basic"]

# ---------------- Dados da instância ----------------

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

MATRIZ_DISTANCIAS = ler_matriz_distancias("matriz_distancias.txt")
N_CIDADES = len(MATRIZ_DISTANCIAS)

os.makedirs("resultados", exist_ok=True)
os.makedirs("resultados/graficos", exist_ok=True)

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

def rodar_ag():
    populacao = criar_populacao()

    historico_melhor = []

    for geracao in range(GENERATIONS):
        melhor = max(populacao, key=fitness)
        melhor_distancia = calcular_distancia_total(melhor)
        historico_melhor.append(melhor_distancia)

        populacao = nova_geracao(populacao)

    melhor_final = max(populacao, key=fitness)
    distancia_final = calcular_distancia_total(melhor_final)

    historico_melhor.append(distancia_final)

    return distancia_final, melhor_final, historico_melhor

# ---------------- Grid Search ----------------

resultados_execucoes = []
resultados_resumo = []
historicos_convergencia = {}

if grid_search["enabled"]:
    pop_sizes = grid_search.get("POP_SIZE")
    generations_list = grid_search.get("GENERATIONS")
    mutation_rates = grid_search.get("MUTATION_RATE")
    crossover_rates = grid_search.get("CROSSOVER_RATE")
    crossover_operators = grid_search.get("CROSSOVER_OPERATOR")

    combinacoes = list(product(
        pop_sizes,
        generations_list,
        mutation_rates,
        crossover_rates,
        crossover_operators
    ))

    print(f"Total de combinações: {len(combinacoes)}")

    id_config = 0

    for pop_size, generations, mutation, crossover_rate, crossover_operator in combinacoes:
        id_config += 1

        CROSSOVER_OPERATOR = crossover_operator
        POP_SIZE = pop_size
        GENERATIONS = generations
        MUTATION_RATE = mutation
        CROSSOVER_RATE = crossover_rate
        ELITISM = max(1, POP_SIZE // 10)
        TOURNAMENT_SIZE = max(2, POP_SIZE // 5)

        print(
            f"\nConfiguração {id_config}: "
            f"operador={CROSSOVER_OPERATOR}, "
            f"pop={POP_SIZE}, "
            f"gerações={GENERATIONS}, "
            f"crossover={CROSSOVER_RATE}, "
            f"mutação={MUTATION_RATE}"
        )

        distancias_config = []
        historicos_config = []

        for execucao in range(basic_config["iterations"]):
            seed = 1000 * id_config + execucao
            random.seed(seed)

            distancia_final, melhor_rota, historico = rodar_ag()

            distancias_config.append(distancia_final)
            historicos_config.append(historico)

            resultados_execucoes.append({
                "id_config": id_config,
                "execucao": execucao + 1,
                "seed": seed,
                "operador": CROSSOVER_OPERATOR,
                "populacao": POP_SIZE,
                "geracoes": GENERATIONS,
                "taxa_crossover": CROSSOVER_RATE,
                "taxa_mutacao": MUTATION_RATE,
                "distancia_final": distancia_final,
                "melhor_rota": melhor_rota
            })

            print(
                f"  Execução {execucao + 1}: "
                f"seed={seed}, distância={distancia_final}"
            )
            
        melhor = min(distancias_config)
        pior = max(distancias_config)
        media = statistics.mean(distancias_config)
        mediana = statistics.median(distancias_config)

        if len(distancias_config) > 1:
            desvio_padrao = statistics.stdev(distancias_config)
        else:
            desvio_padrao = 0

        resultados_resumo.append({
            "id_config": id_config,
            "operador": CROSSOVER_OPERATOR,
            "populacao": POP_SIZE,
            "geracoes": GENERATIONS,
            "taxa_crossover": CROSSOVER_RATE,
            "taxa_mutacao": MUTATION_RATE,
            "melhor_resultado": melhor,
            "pior_resultado": pior,
            "media": media,
            "mediana": mediana,
            "desvio_padrao": desvio_padrao
        })

        historicos_convergencia[id_config] = historicos_config

# ---------------- CSV dos resultados ----------------
    
with open("resultados/resultados_execucoes.csv", "w", newline="") as arquivo:
    campos = [
        "id_config",
        "execucao",
        "seed",
        "operador",
        "populacao",
        "geracoes",
        "taxa_crossover",
        "taxa_mutacao",
        "distancia_final",
        "melhor_rota"
    ]

    writer = csv.DictWriter(arquivo, fieldnames=campos)
    writer.writeheader()
    writer.writerows(resultados_execucoes)

with open("resultados/resumo_resultados.csv", "w", newline="") as arquivo:
    campos = [
        "id_config",
        "operador",
        "populacao",
        "geracoes",
        "taxa_crossover",
        "taxa_mutacao",
        "melhor_resultado",
        "pior_resultado",
        "media",
        "mediana",
        "desvio_padrao"
    ]

    writer = csv.DictWriter(arquivo, fieldnames=campos)
    writer.writeheader()
    writer.writerows(resultados_resumo)

# ---------------- BOXPLOT ----------------

dados_boxplot = []

for resumo in resultados_resumo:
    id_config = resumo["id_config"]

    distancias = [
        r["distancia_final"]
        for r in resultados_execucoes
        if r["id_config"] == id_config
    ]

    dados_boxplot.append(distancias)

labels = [str(r["id_config"]) for r in resultados_resumo]

plt.figure(figsize=(12, 6))
plt.boxplot(dados_boxplot, labels=labels)

plt.xlabel("Configuração")
plt.ylabel("Distância final")
plt.title("Boxplot das distâncias finais por configuração")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("resultados/graficos/boxplot_configuracoes.png", dpi=300)
plt.close()

# ---------------- Gráfico de barras (média e desvio padrão) ----------------

ids = [r["id_config"] for r in resultados_resumo]
medias = [r["media"] for r in resultados_resumo]
desvios = [r["desvio_padrao"] for r in resultados_resumo]

plt.figure(figsize=(12, 6))
plt.bar(ids, medias, yerr=desvios, capsize=5)

plt.xlabel("Configuração")
plt.ylabel("Distância média")
plt.title("Média e desvio padrão por configuração")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("resultados/graficos/barras_media_desvio.png", dpi=300)
plt.close()

# ---------------- Gráfico de convergência da melhor configuração ----------------

melhor_config = min(resultados_resumo, key=lambda x: x["media"])
id_melhor_config = melhor_config["id_config"]

historicos = historicos_convergencia[id_melhor_config]
tamanho_historico = len(historicos[0])

media_convergencia = []

for i in range(tamanho_historico):
    valores_geracao = [historico[i] for historico in historicos]
    media_convergencia.append(statistics.mean(valores_geracao))

plt.figure(figsize=(10, 5))
plt.plot(media_convergencia, linewidth=2, label=f"Configuração {id_melhor_config}")

plt.xlabel("Geração")
plt.ylabel("Melhor distância média")
plt.title("Convergência média da melhor configuração")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("resultados/graficos/convergencia_melhor_config.png", dpi=300)
plt.close()