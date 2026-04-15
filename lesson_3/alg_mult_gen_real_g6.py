# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# Algoritmo Genético com representação real para o problema G06
# -----------------------------------------------------------------------------------

import random
import matplotlib.pyplot as plt

# ---------------- Parâmetros ----------------
POP_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.1
MUTATION_STD = 1.5
ELITISM = 5
PENALTY_LAMBDA = 2000.0

X1_MIN, X1_MAX = 13.0, 100.0
X2_MIN, X2_MAX = 0.0, 100.0

NUM_RUNS = 10  # Número de execuções independentes

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
    candidatos = populacao[:]

    fit_val = []
    for individuo in candidatos:
        fit = fitness(individuo)

        # Evita divisão por zero ou fitness muito pequeno/negativo
        if fit <= 0:
            fit = 1e-6

        fit_val.append(1.0 / fit)

    soma_fitness = sum(fit_val)

    if soma_fitness == 0:
        return random.choice(candidatos)

    probabilidades = [valor / soma_fitness for valor in fit_val]

    valor_sorteado = random.uniform(0, 1)
    acumulado = 0.0

    for i in range(len(probabilidades)):
        acumulado += probabilidades[i]
        if valor_sorteado <= acumulado:
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

        gene_filho1 = random.uniform(limite_inf, limite_sup)
        gene_filho2 = random.uniform(limite_inf, limite_sup)

        filho1.append(gene_filho1)
        filho2.append(gene_filho2)

    # Garantir que os filhos respeitem os limites do domínio
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

all_melhores_f_obj = []
all_melhores_fitness = []
all_melhores_violacoes = []

melhores_finais_execucoes = []

for run in range(NUM_RUNS):
    print(f"\n--- Execução {run + 1} ---")

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

        if geracao % 50 == 0:
            print(
                f"Geração {geracao}: "
                f"x1 = {x1_best:.6f}, "
                f"x2 = {x2_best:.6f}, "
                f"f(x) = {best_f:.6f}, "
                f"violação = {best_violation:.6f}, "
                f"fitness = {best_fit:.6f}"
            )

        populacao = nova_geracao(populacao)

    melhor_final = min(populacao, key=fitness)
    x1_final, x2_final = melhor_final
    f_final = funcao_objetivo(x1_final, x2_final)
    violacao_final = violacao_total(x1_final, x2_final)
    fitness_final = fitness(melhor_final)

    all_melhores_f_obj.append(melhores_f_obj)
    all_melhores_fitness.append(melhores_fitness)
    all_melhores_violacoes.append(melhores_violacoes)
    melhores_finais_execucoes.append((f_final, x1_final, x2_final, violacao_final, fitness_final))

    print(f"Melhor da execução {run + 1}:")
    print(f"x1 = {x1_final:.6f}")
    print(f"x2 = {x2_final:.6f}")
    print(f"f(x) = {f_final:.6f}")
    print(f"g1 = {g1(x1_final, x2_final):.6f}")
    print(f"g2 = {g2(x1_final, x2_final):.6f}")
    print(f"violação = {violacao_final:.6f}")
    print(f"fitness = {fitness_final:.6f}")

# ---------------- Resultado final ----------------

melhor_global = min(melhores_finais_execucoes, key=lambda x: x[4])
f_best, x1_best, x2_best, viol_best, fit_best = melhor_global

print("\n--- Melhor resultado global entre todas as execuções ---")
print(f"x1 = {x1_best:.6f}")
print(f"x2 = {x2_best:.6f}")
print(f"f(x) = {f_best:.6f}")
print(f"g1 = {g1(x1_best, x2_best):.6f}")
print(f"g2 = {g2(x1_best, x2_best):.6f}")
print(f"violação = {viol_best:.6f}")
print(f"fitness = {fit_best:.6f}")

# ---------------- Gráficos ----------------

plt.figure(figsize=(12, 6))
for run in range(NUM_RUNS):
    plt.plot(all_melhores_f_obj[run], linewidth=1, label=f"Execução {run + 1}")
plt.xlabel("Geração")
plt.ylabel("f(x)")
plt.title(f"Convergência da função objetivo - {NUM_RUNS} execuções")
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("grafico_funcao_objetivo_g06.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
for run in range(NUM_RUNS):
    plt.plot(all_melhores_fitness[run], linewidth=1, label=f"Execução {run + 1}")
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.title(f"Convergência do fitness - {NUM_RUNS} execuções")
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("grafico_fitness_g06.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
for run in range(NUM_RUNS):
    plt.plot(all_melhores_violacoes[run], linewidth=1, label=f"Execução {run + 1}")
plt.xlabel("Geração")
plt.ylabel("Violação total")
plt.title(f"Convergência da violação das restrições - {NUM_RUNS} execuções")
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("grafico_restricoes_g06.png", dpi=300)
plt.close()

print("\nGráficos salvos com sucesso:")
print(" - grafico_funcao_objetivo_g06.png")
print(" - grafico_fitness_g06.png")
print(" - grafico_restricoes_g06.png")