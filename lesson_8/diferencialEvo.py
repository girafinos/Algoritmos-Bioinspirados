# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# Algoritmo Evolução Diferencial
# -----------------------------------------------------------------------------------

import random
import math
import matplotlib.pyplot as plt
import csv

#----------------- Parâmetros -----------------
POP_SIZE = 50
GENERATIONS = 100

#F = 0.3
#CR = 0.3
DIMENSA0 = 4

X_MIN, X_MAX = -1000000, 1000000

MENOR_SOLUCAO = math.inf
MENOR_INDIVIDUO = []
MELHORES_INDIVUOS = []

random.seed(51)

#---------------- Funções do problema ---------------

#Função sphere
def sphere(individuo):
    soma = 0
    
    for x in individuo:
        soma+= x**2

    return soma

#Função Rosenbrock
def rosenbrock(individuo):
    soma = 0
    
    for i in range(len(individuo) - 1):
        xi = individuo[i]
        x_next = individuo[i + 1]
        
        soma += 100 * (x_next - xi**2)**2 + (1 - xi)**2
    
    return soma

#Função Rastrigin
def rastrigin(individuo):
    n = len(individuo)
    soma = 0
    
    for x in individuo:
        soma += x**2 - 10 * math.cos(2 * math.pi * x)
    
    return 10 * n + soma

def fitness(individuo):
    #return sphere(individuo)
    #return rosenbrock(individuo)
    return rastrigin(individuo)
    
#---------------- Funções do DE -------------------

def criar_individuo():
    ind = []
    for _ in range(DIMENSA0):
        ind.append(random.uniform(X_MIN, X_MAX))
    return ind

def criar_populacao():
    pop = []
    for _ in range(POP_SIZE):
        pop.append(criar_individuo())
    return pop

def eh_menor(ind1, ind2):
    if (fitness(ind1) < fitness(ind2)):
        return ind1
    else:
        return ind2

def crossover(ind_x, ind_mutante, CR):
    vetor_teste = []
    for i in range(len(ind_mutante)):
        sorteio = random.uniform(0,1)
        if sorteio >= CR:
            vetor_teste.append(ind_mutante[i])
        else:
            vetor_teste.append(ind_x[i])
    return vetor_teste

def mutacao_dif(ind_x, ind1, ind2, ind3, F, CR):
    ind_mutante= []
    for i in range(len(ind1)):
        ind_mutante.append(ind1[i]+ F * ind2[i]-ind3[i])
    return crossover(ind_x, ind_mutante, CR)

def mutar_pop(populacao, F, CR):
    nova_pop = []
    for i in range(len(populacao)):
        indice1 = random.randint(0, POP_SIZE-1) 
        indice2 = random.randint(0, POP_SIZE-1) 
        indice3 = random.randint(0, POP_SIZE-1) 
        individuo = eh_menor(populacao[i], mutacao_dif(populacao[i],populacao[indice1],
                                                       populacao[indice2], populacao[indice3], F, CR))
        nova_pop.append(individuo)

    return nova_pop
    
#----------------- Função para rodar o algoritmo ------------------
def rodar_de(F, CR):
    MENOR_SOLUCAO = math.inf
    MELHORES_INDIVIDUOS = []

    populacao = criar_populacao()

    for i in range(GENERATIONS):
        populacao = mutar_pop(populacao, F, CR)
        populacao_aux = sorted(populacao, key=fitness)

        if fitness(populacao_aux[0]) < MENOR_SOLUCAO:
            MENOR_SOLUCAO = fitness(populacao_aux[0])

        MELHORES_INDIVIDUOS.append(fitness(populacao_aux[0]))

    return MENOR_SOLUCAO, MELHORES_INDIVIDUOS

#----------------- Main com teste fatorial ------------------

valores = [0.3, 0.5, 0.9]

resultados = []

for F in valores:
    for CR in valores:
        print(f"\nTestando F={F}, CR={CR}")

        melhor, evolucao = rodar_de(F, CR)

        resultados.append({
            "F": F,
            "CR": CR,
            "melhor": melhor,
            "evolucao": evolucao
        })

        print(f"Melhor solução: {melhor}")


#----------------------------- Salvando Resultados -------------------------

step = 10  # intervalo de gerações no csv

with open("lesson_8/resultados_comparacao.csv", mode="w", newline="") as arquivo:
    writer = csv.writer(arquivo)
    
    header = ["geracao"]
    for r in resultados:
        header.append(f"F={r['F']}_CR={r['CR']}")
    writer.writerow(header)
    
    indices = list(range(0, GENERATIONS, step))
    
    if (GENERATIONS - 1) not in indices:
        indices.append(GENERATIONS - 1)
    
    indices = sorted(indices)
    
    for i in indices:
        linha = [i + 1]  
        
        for r in resultados:
            valor = r["evolucao"][i]
            linha.append(f"{valor:.4f}")  
        
        writer.writerow(linha)

#----------------------------- Gráfico de evolução -------------------------

plt.figure(figsize=(14, 6))

for r in resultados:
    label = f"F={r['F']}, CR={r['CR']}"
    plt.plot(
        range(1, GENERATIONS + 1),
        r["evolucao"],
        label=label
    )

plt.axhline(y=0, linestyle='--', label='Ótimo (0)')

plt.title('Comparação da evolução (Teste Fatorial)')
plt.xlabel('Iteração')
plt.ylabel('Melhor distância')
plt.grid(True)

plt.legend(fontsize=8)  # importante pq são 16 curvas
plt.tight_layout()

plt.savefig("lesson_8/grafico_comparacao.png", dpi=300)
plt.show()

#----------------------------- Gráfico de evolução (Ultimas 50% gerações) -------------------------

plt.figure(figsize=(14, 6))

inicio = int(GENERATIONS * 0.5)

valores_finais = []

for r in resultados:
    trecho = r["evolucao"][inicio:]
    valores_finais.extend(trecho)

    label = f"F={r['F']}, CR={r['CR']}"
    
    plt.plot(
        range(inicio + 1, GENERATIONS + 1),
        trecho,
        label=label
    )

ymin = min(valores_finais)
ymax = max(valores_finais)*0.1

margem = (ymax - ymin) * 0.1

plt.ylim(ymin - margem, ymax + margem)

plt.axhline(y=0, linestyle='--', label='Ótimo (0)')

plt.title('Evolução - Últimos 50% das Gerações (Zoom)')
plt.xlabel('Iteração')
plt.ylabel('Melhor valor')
plt.grid(True)

plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig("lesson_8/grafico_ultimos_50.png", dpi=300)
plt.show()

#----------------------------- Gráfico de evolução (Ultimas 10% gerações) -------------------------
plt.figure(figsize=(14, 6))

inicio = int(GENERATIONS * 0.9)

valores_finais = []

for r in resultados:
    trecho = r["evolucao"][inicio:]
    valores_finais.extend(trecho)

    label = f"F={r['F']}, CR={r['CR']}"
    
    plt.plot(
        range(inicio + 1, GENERATIONS + 1),
        trecho,
        label=label
    )

ymin = min(valores_finais)
ymax = max(valores_finais)*0.01

margem = (ymax - ymin) * 0.05 

plt.ylim(ymin - margem, ymax + margem)

plt.axhline(y=0, linestyle='--', label='Ótimo (0)')

plt.title('Evolução - Últimos 10% das Gerações (Zoom)')
plt.xlabel('Iteração')
plt.ylabel('Melhor valor')
plt.grid(True)

plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig("lesson_8/grafico_ultimos_10.png", dpi=300)
plt.show()