# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# Algoritmo Imunológico para o Problema do Caixeiro Viajante (TSP)
# --------------------------------------------------------------------------------

import random
import matplotlib.pyplot as plt
import numpy as np
import copy
import math


#------------ Parâmetros-------------------
POP_SIZE = 50
ITERACOES = 50

BETA_CLONAGEM = 0.9
RHO_MUTACAO = 0.3

D_MEMORIA = int(round(POP_SIZE * 0.5))
NUM_SELECIONADOS = int(round(POP_SIZE * 0.8))# Quantos são selecionados para clonagem

MELHOR_CAMINHO = None
MENOR_DISTANCIA = float('inf')
MENORES_DISTANCIAS = []


#------------- Lendo a Matriz de Distâncias -------------

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

MATRIZ_DISTANCIAS = ler_matriz_distancias("/home/gabriel/Área de trabalho/Faculdade/7periodo/BIoinspirados/Algoritmos-Bioinspirados-main/lesson_7/lau15_dist.txt")
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

#----------------- Funções do Clonalg ----------------

def criar_individuo():
    individuo = list(range(N_CIDADES))
    random.shuffle(individuo)
    return individuo


def criar_populacao():
    return [criar_individuo() for _ in range(POP_SIZE)]

def normalizar(populacao):
    vetor_distancias = np.array([calcular_distancia_total(i) for i in populacao])

    min_val = np.min(vetor_distancias)
    max_val = np.max(vetor_distancias)

    denominador = max_val - min_val

    if denominador == 0:
        return np.ones_like(vetor_distancias)  # todos iguais = alta afinidade

    normalizado = (vetor_distancias - min_val) / denominador

    afinidade = 1 - normalizado
    return afinidade

def selecao(populacao):
    pop_selecionada = sorted(populacao, key=calcular_distancia_total)
    return pop_selecionada[:NUM_SELECIONADOS]

def clonagem(populacao):
    clones = []

    for indice, individuo in enumerate(populacao):
        quantidade = math.ceil(BETA_CLONAGEM * (POP_SIZE / (indice + 1)))

        for _ in range(quantidade):
            clones.append(copy.deepcopy(individuo))

    random.shuffle(clones)
    return clones[:POP_SIZE]  

def mutacao(individuo, taxa_mutacao):
    filho = individuo[:]
    if random.random() < taxa_mutacao:
        i, j = sorted(random.sample(range(N_CIDADES), 2))
        filho[i:j + 1] = reversed(filho[i:j + 1])
    return filho

def mutar_pop(populacao):
    afinidades = normalizar(populacao)
    pop_mutada = []
 
    for i, individuo in enumerate(populacao):
        taxa_mutacao = math.exp(-RHO_MUTACAO * afinidades[i])
        individuo_mutado = mutacao(individuo, taxa_mutacao)
        pop_mutada.append(individuo_mutado)
 
    return pop_mutada


def memoria(populacao_mutada, populacao_anterior, melhor_global):
    combinada = populacao_mutada + populacao_anterior
    combinada = sorted(combinada, key=calcular_distancia_total)
    combinada = combinada[:D_MEMORIA]  # mantém os D melhores
 
    # Completa com indivíduos aleatórios (apenas a fração necessária)
    while len(combinada) < POP_SIZE:
        combinada.append(criar_individuo())
 
    # ELITISMO: garante que o melhor global nunca é perdido
    combinada[0] = copy.deepcopy(melhor_global)
 
    return combinada[:POP_SIZE]

# ---------------- Execução -------------------

pop = criar_populacao()
pop_anterior = copy.deepcopy(pop)

print(f"\nConfiguração: POP={POP_SIZE}, ITER={ITERACOES}, BETA={BETA_CLONAGEM}, RHO={RHO_MUTACAO}")
print(f"Selecionados={NUM_SELECIONADOS}, Memória={D_MEMORIA}\n")
 
melhor_global = min(pop, key=calcular_distancia_total)
MENOR_DISTANCIA = calcular_distancia_total(melhor_global)
MELHOR_CAMINHO = melhor_global[:]


for iteracao in range(ITERACOES):
    selecionados = selecao(pop)
 
    clones = clonagem(selecionados)
 
    pop_mutada = mutar_pop(clones)

    candidato = min(pop_mutada, key=calcular_distancia_total)
    dist_candidato = calcular_distancia_total(candidato)
    if dist_candidato < MENOR_DISTANCIA:
        MENOR_DISTANCIA = dist_candidato
        MELHOR_CAMINHO = candidato[:]
        melhor_global = candidato[:]

    pop_anterior = copy.deepcopy(pop)
    pop = memoria(pop_mutada, pop_anterior, melhor_global)
 
    MENORES_DISTANCIAS.append(MENOR_DISTANCIA)
 
    if (iteracao + 1) % 10 == 0:
        print(f"Iteração {iteracao + 1:3d} | Melhor distância: {MENOR_DISTANCIA}")


print("\nResultados finais:")
print("Melhor caminho encontrado:", MELHOR_CAMINHO)
print("Melhor distância encontrada:", MENOR_DISTANCIA)
print("Menores distâncias ao longo das iterações:", MENORES_DISTANCIAS)


#------------------ Gráfico da evolução da melhor distância ----------------

plt.figure(figsize=(14, 6))
plt.plot(range(1, ITERACOES + 1), MENORES_DISTANCIAS, linestyle='-')
plt.axhline(y=291, linestyle='--', label='Ótimo (291)')

plt.title('Evolução da melhor distância')

plt.xlabel('Iteração')
plt.ylabel('Melhor distância')
plt.grid(True)

# Texto com informações
texto_info = (
    f"Solução ótima esperada: 291\n"
    f"Menor distancia: {MENOR_DISTANCIA}\n"
    f"População: {POP_SIZE}\n"
    f"Iterações: {ITERACOES}\n"
    f"BETA (clonagem): {BETA_CLONAGEM}\n"
    f"RHO (Mutaçao): {RHO_MUTACAO}\n"
    f"D (Memória): {D_MEMORIA}\n"
    f"Número de selecionados para clonagem: {NUM_SELECIONADOS}"
)

plt.text(
    0.8, 0.95, texto_info,
    transform=plt.gca().transAxes,  # usa coordenadas relativas (0 a 1)
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='square', facecolor='white', alpha=0.8)
)

plt.savefig("lesson_7/grafico_distancia_lau15.png", dpi=300)
plt.show()
