# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# Algoritmo Genético com representação real (sem limite de bits)
# -----------------------------------------------------------------------------------

import random
# import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Individuo:
    Velocidade : list
    Pbest : list
    Pos : list

# ---------------- Parâmetros ----------------
POP_SIZE = 20
GENERATIONS = 100
GEN_ATUAL = 0

X1_MIN, X1_MAX = -10.0, 10.0
X2_MIN, X2_MAX = -10.0, 10.0

G_BEST = [0.0, 0.0]
W = 0.9
C1 = 2.05
C2 = 2.05

# ---------------- Funções do problema ----------------
def funcao_objetivo(x1, x2):
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

def fitness(x1, x2):
    f = funcao_objetivo(x1, x2)
    return f 

# ---------------- Funções do PSO ----------------     
def criar_individuo():
    x1 = random.uniform(X1_MIN, X1_MAX)
    x2 = random.uniform(X2_MIN, X2_MAX)
    velocidade = [random.uniform(X1_MIN, X1_MAX), random.uniform(X2_MIN, X2_MAX)]
    
    return Individuo(Pos=[x1, x2], Velocidade=velocidade, Pbest=[x1, x2])

def criar_populacao():
    return [criar_individuo() for _ in range(POP_SIZE)]

def calculo_velocidade(individuo):
    for i in range(len(individuo.Pos)):
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0,1)
        W = 0.9 - (GEN_ATUAL / GENERATIONS) * 0.5  
        individuo.Velocidade[i] = (W*individuo.Velocidade[i] + C1*r1*(individuo.Pbest[i] - individuo.Pos[i]) + C2*r2*(G_BEST[i] - individuo.Pos[i]))

def atualiza_posicao(individuo):
    calculo_velocidade(individuo)
    for i in range(len(individuo.Pos)):
        individuo.Pos[i] = individuo.Pos[i] + individuo.Velocidade[i]
    
    individuo.Pos[0], individuo.Pos[1] = limite(individuo.Pos[0], individuo.Pos[1])
    
    pbest_fitness = fitness(individuo.Pbest[0], individuo.Pbest[1])
    fitness_atual = fitness(individuo.Pos[0], individuo.Pos[1])
    if fitness_atual < pbest_fitness:
        individuo.Pbest = individuo.Pos.copy()
    
    gbest_fitness = fitness(G_BEST[0], G_BEST[1])
    if fitness_atual < gbest_fitness:
        G_BEST[0] = individuo.Pos[0]
        G_BEST[1] = individuo.Pos[1]

def limite(x1,x2):
    if x1 < X1_MIN:
        x1 = X1_MIN
    elif x1 > X1_MAX:
        x1 = X1_MAX
    if x2 < X2_MIN:
        x2 = X2_MIN
    elif x2 > X2_MAX:
        x2 = X2_MAX
    return x1, x2

def atualiza_pop(populacao):
    for individuo in populacao:
        atualiza_posicao(individuo)

# ---------------- Execução ----------------
populacao = criar_populacao()

# Inicializar G_BEST com o melhor da população inicial
melhor_inicial = min(populacao, key=lambda ind: fitness(ind.Pos[0], ind.Pos[1]))
G_BEST = melhor_inicial.Pos.copy()

for iteracao in range(GENERATIONS):
    atualiza_pop(populacao)
    GEN_ATUAL += 1

# Debug: print da população
print("População criada:")
for ind in populacao:
    print(ind)

print("Posição do primeiro indivíduo:", populacao[0].Pos)
print("Velocidade x2 do primeiro indivíduo:", populacao[0].Velocidade[1])

melhores_fitness = []
melhores_f_obj = []
melhores_violacoes = []

print("População inicial:")
for i, individuo in enumerate(populacao):
    x1, x2 = individuo.Pos
    print(
        f"Indivíduo {i}: "
        f"x1 = {x1:.6f}, "
        f"x2 = {x2:.6f}, "
        f"f(x) = {funcao_objetivo(x1, x2):.6f}, "
        f"fitness = {fitness(x1, x2):.6f}"
    )

# ---------------- Resultado final ----------------

melhor_final = min(populacao, key=lambda ind: fitness(ind.Pos[0], ind.Pos[1]))
x1_final, x2_final = melhor_final.Pos
f_final = funcao_objetivo(x1_final, x2_final)
fitness_final = fitness(x1_final, x2_final)

print("\n--- Melhor solução encontrada ---")
print(f"x1 = {x1_final:.10f}")
print(f"x2 = {x2_final:.10f}")
print(f"f(x) = {f_final:.10f}")
print(f"Fitness penalizado = {fitness_final:.10f}")

print("\n--- Ótimo teórico esperado ---")
print("x1 = 1")
print("x2 = 3")
print("f(x) = 0")