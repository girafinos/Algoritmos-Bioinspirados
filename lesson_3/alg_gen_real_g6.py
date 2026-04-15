# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# Algoritmo Genético com representação real para o problema G06
# -----------------------------------------------------------------------------------

import random
import matplotlib.pyplot as plt

# ---------------- Parâmetros ----------------
POP_SIZE = 100
GENERATIONS = 500
ROULETTE_SIZE = 20
MUTATION_RATE = 0.2
MUTATION_STD = 1.5
ELITISM = 10
PENALTY_LAMBDA = 2000.0

X1_MIN, X1_MAX = 13.0, 100.0
X2_MIN, X2_MAX = 0.0, 100.0

# Para reproduzir resultados
# random.seed(42)

# ---------------- Funções do problema ----------------

def funcao_objetivo(x1, x2):
    return (x1 - 10)**3 + (x2 - 20)**3

def g1(x1, x2):
    return -(x1 - 5)**2 - (x2 - 5)**2 + 100

def g2(x1, x2):
    return (x1 - 6)**2 + (x2 - 5)**2 - 82.81

def violacao_total(x1, x2):
    v1 = max(0, g1(x1, x2))
    v2 = max(0, g2(x1, x2))
    return v1 + v2

def fitness(individuo):
    x1, x2 = individuo
    f = funcao_objetivo(x1, x2)
    penalidade = PENALTY_LAMBDA * violacao_total(x1, x2)
    return f + penalidade

# ---------------- Funções do AG ----------------

def limitar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))

def criar_individuo():
    x1 = random.uniform(X1_MIN, X1_MAX)
    x2 = random.uniform(X2_MIN, X2_MAX)
    return [x1, x2]

def criar_populacao():
    return [criar_individuo() for _ in range(POP_SIZE)]

def roleta(populacao):
    candidatos = populacao

    fit_val = []
    for i in range(len(candidatos)):
        individuo = candidatos[i]
        fit = fitness(individuo)

        # Evita divisão por zero ou valor negativo/zero
        if fit <= 0:
            fit = 1e-6

        fit_val.append(fit)

    for i in range(len(fit_val)):
        fit_val[i] = 1 / fit_val[i]

    somaFitness = sum(fit_val)
    vetorRoleta = []

    for i in range(len(fit_val)):
        vetorRoleta.append(fit_val[i] / somaFitness)

    valor_sorteado = random.uniform(0, 1)
    aux = 0

    for i in range(len(vetorRoleta)):
        aux = aux + vetorRoleta[i]
        if valor_sorteado <= aux:
            return candidatos[i]

    return candidatos[-1]

def crossover_blx_alpha(pai1, pai2):
    alpha = 0.5

    filho1 = []
    filho2 = []

    for i in range(len(pai1)):
        d = abs(pai1[i] - pai2[i])
        limite_inf = min(pai1[i], pai2[i]) - alpha * d
        limite_sup = max(pai1[i], pai2[i]) + alpha * d

        filho1.append(random.uniform(limite_inf, limite_sup))
        filho2.append(random.uniform(limite_inf, limite_sup))

    # Limitar ao domínio
    filho1[0] = limitar(filho1[0], X1_MIN, X1_MAX)
    filho1[1] = limitar(filho1[1], X2_MIN, X2_MAX)

    filho2[0] = limitar(filho2[0], X1_MIN, X1_MAX)
    filho2[1] = limitar(filho2[1], X2_MIN, X2_MAX)

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
        pai1 = roleta(populacao)
        pai2 = roleta(populacao)

        filho1, filho2 = crossover_blx_alpha(pai1, pai2)

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

for geracao in range(GENERATIONS):
    melhor = min(populacao, key=fitness)
    x1_best, x2_best = melhor

    best_f = funcao_objetivo(x1_best, x2_best)
    best_violation = violacao_total(x1_best, x2_best)
    best_fit = fitness(melhor)

    melhores_fitness.append(best_fit)
    melhores_f_obj.append(best_f)
    melhores_violacoes.append(best_violation)

    if geracao % 10 == 0:
        print(
            f"Geração {geracao}: "
            f"x1 = {x1_best:.6f}, "
            f"x2 = {x2_best:.6f}, "
            f"f(x) = {best_f:.6f}, "
            f"violação = {best_violation:.6f}, "
            f"fitness = {best_fit:.6f}"
        )

    populacao = nova_geracao(populacao)

# ---------------- Resultado final ----------------

melhor_final = min(populacao, key=fitness)
x1_final, x2_final = melhor_final
f_final = funcao_objetivo(x1_final, x2_final)
violacao_final = violacao_total(x1_final, x2_final)
fitness_final = fitness(melhor_final)

print("\n--- Melhor solução encontrada ---")
print(f"x1 = {x1_final:.10f}")
print(f"x2 = {x2_final:.10f}")
print(f"f(x) = {f_final:.10f}")
print(f"g1(x) = {g1(x1_final, x2_final):.10f}")
print(f"g2(x) = {g2(x1_final, x2_final):.10f}")
print(f"Violação total = {violacao_final:.10f}")
print(f"Fitness penalizado = {fitness_final:.10f}")

print("\n--- Referência teórica esperada para o G06 ---")
print("x1 ≈ 14.0950000000")
print("x2 ≈ 0.8429600000")
print("f(x) ≈ -6961.8138760000")

# ---------------- Gráfico 1: função objetivo ----------------

plt.figure(figsize=(10, 5))
plt.plot(melhores_f_obj, linewidth=2, label="f(x) (melhor)")
plt.axhline(y=-6961.813876, linestyle='--', label='Referência (~ -6961.813876)')

plt.xlabel("Geração")
plt.ylabel("f(x)")
plt.title("Convergência da função objetivo")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("grafico_funcao_objetivo_g06.png", dpi=300)
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

plt.savefig("grafico_fitness_g06.png", dpi=300)
plt.close()

# ---------------- Gráfico 3: restrições ----------------

plt.figure(figsize=(10, 5))
plt.plot(melhores_violacoes, linewidth=2, label="Violação total")

plt.xlabel("Geração")
plt.ylabel("Violação")
plt.title("Convergência das restrições")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("grafico_restricao_g06.png", dpi=300)
plt.close()

print("\nGráficos salvos com sucesso:")
print(" - grafico_funcao_objetivo_g06.png")
print(" - grafico_fitness_g06.png")
print(" - grafico_restricao_g06.png")