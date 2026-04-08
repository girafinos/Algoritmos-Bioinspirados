# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# Algoritmo Genético com representação real (sem limite de bits)
# -----------------------------------------------------------------------------------

import random
import matplotlib.pyplot as plt

# ---------------- Parâmetros ----------------
POP_SIZE = 50
GENERATIONS = 10
TOURNAMENT_SIZE = 2
MUTATION_RATE = 0.2
MUTATION_STD = 0.05
ELITISM = 2

X1_MIN, X1_MAX = -10.0, 10.0
X2_MIN, X2_MAX = -10.0, 10.0

# Para reproduzir resultados
#random.seed(42)

# ---------------- Funções do problema ----------------

def funcao_objetivo(x1, x2):
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

def fitness(individuo):
    x1, x2 = individuo
    f = funcao_objetivo(x1, x2)
    return f 

# ---------------- Funções do AG ----------------

def limitar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))

def criar_individuo():
    x1 = random.uniform(X1_MIN, X1_MAX)
    x2 = random.uniform(X2_MIN, X2_MAX)
    return [x1, x2]

def criar_populacao():
    return [criar_individuo() for _ in range(POP_SIZE)]

def torneio(populacao):
    candidatos = random.sample(populacao, TOURNAMENT_SIZE)
    return min(candidatos, key=fitness)  # minimização

def crossover_aritmetico(pai1, pai2):
    alpha = random.random()

    filho1_x1 = alpha * pai1[0] + (1 - alpha) * pai2[0]
    filho1_x2 = alpha * pai1[1] + (1 - alpha) * pai2[1]

    filho2_x1 = alpha * pai2[0] + (1 - alpha) * pai1[0]
    filho2_x2 = alpha * pai2[1] + (1 - alpha) * pai1[1]

    filho1 = [filho1_x1, filho1_x2]
    filho2 = [filho2_x1, filho2_x2]

    return filho1, filho2

def mutacao(individuo):
    x1, x2 = individuo

    if random.random() < MUTATION_RATE:
        x1 += random.gauss(0, MUTATION_STD)

    if random.random() < MUTATION_RATE:
        x2 += random.gauss(0, MUTATION_STD)

    x1 = limitar(x1, X1_MIN, X1_MAX)
    x2 = limitar(x2, X2_MIN, X2_MAX)

    return [x1, x2]

def nova_geracao(populacao):
    populacao_ordenada = sorted(populacao, key=fitness)

    nova_pop = [ind[:] for ind in populacao_ordenada[:ELITISM]]

    while len(nova_pop) < POP_SIZE:
        pai1 = torneio(populacao)
        pai2 = torneio(populacao)

        filho1, filho2 = crossover_aritmetico(pai1, pai2)

        filho1 = mutacao(filho1)
        filho2 = mutacao(filho2)

        nova_pop.append(filho1)
        if len(nova_pop) < POP_SIZE:
            nova_pop.append(filho2)

    return nova_pop

# ---------------- Execução ----------------

populacao = criar_populacao()

melhores_fitness = []
melhores_f_obj = []
melhores_violacoes = []

print("População inicial:")
for i, individuo in enumerate(populacao):
    x1, x2 = individuo
    print(
        f"Indivíduo {i}: "
        f"x1 = {x1:.6f}, "
        f"x2 = {x2:.6f}, "
        f"f(x) = {funcao_objetivo(x1, x2):.6f}, "
        f"fitness = {fitness(individuo):.6f}"
    )

for geracao in range(GENERATIONS):
    melhor = min(populacao, key=fitness)
    x1_best, x2_best = melhor

    best_f = funcao_objetivo(x1_best, x2_best)
    best_fit = fitness(melhor)

    melhores_fitness.append(best_fit)
    melhores_f_obj.append(best_f)

    if geracao % 100 == 0:
        print(
            f"Geração {geracao}: "
            f"x1 = {x1_best:.6f}, "
            f"x2 = {x2_best:.6f}, "
            f"f(x) = {best_f:.6f}, "
            f"fitness = {best_fit:.6f}"
        )

    populacao = nova_geracao(populacao)

# ---------------- Resultado final ----------------

melhor_final = min(populacao, key=fitness)
x1_final, x2_final = melhor_final
f_final = funcao_objetivo(x1_final, x2_final)
fitness_final = fitness(melhor_final)

print("\n--- Melhor solução encontrada ---")
print(f"x1 = {x1_final:.10f}")
print(f"x2 = {x2_final:.10f}")
print(f"f(x) = {f_final:.10f}")
print(f"Fitness penalizado = {fitness_final:.10f}")

print("\n--- Ótimo teórico esperado ---")
print("x1 = 1")
print("x2 = 3")
print("f(x) = 0")

# ---------------- Gráfico 1: função objetivo ----------------

plt.figure(figsize=(10, 5))
plt.plot(melhores_f_obj, linewidth=2, label="f(x) (melhor)")
plt.axhline(y=0, linestyle='--', label='Ótimo (0)')

plt.xlabel("Geração")
plt.ylabel("f(x)")
plt.title("Convergência da função objetivo")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("grafico_funcao_objetivo.png", dpi=300)
plt.close()


# ---------------- Gráfico 2: fitness ----------------

plt.figure(figsize=(10, 5))
plt.plot(melhores_fitness, linewidth=2, label="Fitness penalizado")

plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.title("Convergência do fitness")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("grafico_fitness.png", dpi=300)
plt.close()
