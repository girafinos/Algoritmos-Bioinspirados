# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# Algoritmo Genético para o problema da Mochila Binária (instância p03)
# -----------------------------------------------------------------------------------

import random
import matplotlib.pyplot as plt

# ---------------- Parâmetros ----------------
POP_SIZE = 10
GENERATIONS = 50
MUTATION_RATE = 0.2
ELITISM = 1
TOURNAMENT_SIZE = 3

# Para reproduzir resultados, se quiser:
# random.seed(42)

# ---------------- Dados da instância p03 ----------------
# p03_w.txt
PESOS = [56, 59, 80, 64, 75, 17]

# p03_p.txt
VALORES = [50, 50, 64, 46, 50, 5]

# p03_c.txt
CAPACIDADE = 190

# p03_s.txt (solução ótima conhecida)
SOLUCAO_OTIMA = [1, 1, 0, 0, 1, 0]

N_ITEMS = len(PESOS)

# ---------------- Funções do problema ----------------

def peso_total(individuo):
    return sum(individuo[i] * PESOS[i] for i in range(N_ITEMS))

def valor_total(individuo):
    return sum(individuo[i] * VALORES[i] for i in range(N_ITEMS))

def excesso_peso(individuo):
    return max(0, peso_total(individuo) - CAPACIDADE)

def solucao_viavel(individuo):
    return peso_total(individuo) <= CAPACIDADE

def fitness(individuo):
    peso = peso_total(individuo)
    valor = valor_total(individuo)

    if peso <= CAPACIDADE:
        return valor
    else:
        # penalização linear simples
        excesso = peso - CAPACIDADE
        return valor - 10 * excesso

# ---------------- Funções do AG ----------------

def criar_individuo():
    return [random.randint(0, 1) for _ in range(N_ITEMS)]

def criar_populacao():
    return [criar_individuo() for _ in range(POP_SIZE)]

def torneio(populacao):
    candidatos = random.sample(populacao, TOURNAMENT_SIZE)
    return max(candidatos, key=fitness)

def crossover_1_ponto(pai1, pai2):
    corte = random.randint(1, N_ITEMS - 1)

    filho1 = pai1[:corte] + pai2[corte:]
    filho2 = pai2[:corte] + pai1[corte:]

    return filho1, filho2

def mutacao(individuo):
    filho = individuo[:]

    for i in range(N_ITEMS):
        if random.random() < MUTATION_RATE:
            filho[i] = 1 - filho[i]

    return filho

def nova_geracao(populacao):
    populacao_ordenada = sorted(populacao, key=fitness, reverse=True)

    nova_pop = [ind[:] for ind in populacao_ordenada[:ELITISM]]

    while len(nova_pop) < POP_SIZE:
        pai1 = torneio(populacao)
        pai2 = torneio(populacao)

        filho1, filho2 = crossover_1_ponto(pai1, pai2)

        filho1 = mutacao(filho1)
        filho2 = mutacao(filho2)

        nova_pop.append(filho1)
        if len(nova_pop) < POP_SIZE:
            nova_pop.append(filho2)

    return nova_pop

# ---------------- Referência ótima ----------------

valor_otimo = valor_total(SOLUCAO_OTIMA)
peso_otimo = peso_total(SOLUCAO_OTIMA)

# ---------------- Execução ----------------

populacao = criar_populacao()

melhores_fitness = []
melhores_valores = []
melhores_pesos = []
melhores_excessos = []

for geracao in range(GENERATIONS):
    melhor = max(populacao, key=fitness)

    best_fit = fitness(melhor)
    best_valor = valor_total(melhor)
    best_peso = peso_total(melhor)
    best_excesso = excesso_peso(melhor)

    melhores_fitness.append(best_fit)
    melhores_valores.append(best_valor)
    melhores_pesos.append(best_peso)
    melhores_excessos.append(best_excesso)

    if geracao % 10 == 0:
        print(
            f"Geração {geracao}: "
            f"indivíduo = {melhor}, "
            f"valor = {best_valor}, "
            f"peso = {best_peso}, "
            f"excesso = {best_excesso}, "
            f"fitness = {best_fit}, "
            f"viável = {solucao_viavel(melhor)}"
        )

    populacao = nova_geracao(populacao)

# ---------------- Resultado final ----------------

melhor_final = max(populacao, key=fitness)

valor_final = valor_total(melhor_final)
peso_final = peso_total(melhor_final)
excesso_final = excesso_peso(melhor_final)
fitness_final = fitness(melhor_final)

print("\n--- Melhor solução encontrada ---")
print("Indivíduo:", melhor_final)
print("Valor total:", valor_final)
print("Peso total:", peso_final)
print("Excesso de peso:", excesso_final)
print("Fitness:", fitness_final)
print("Solução viável:", solucao_viavel(melhor_final))

print("\n--- Solução ótima conhecida da instância p03 ---")
print("Indivíduo ótimo:", SOLUCAO_OTIMA)
print("Valor ótimo:", valor_otimo)
print("Peso ótimo:", peso_otimo)

if valor_final == valor_otimo and solucao_viavel(melhor_final):
    print("\nO AG encontrou o valor ótimo da instância.")
else:
    print("\nO AG não encontrou exatamente o ótimo conhecido nesta execução.")

# ---------------- Gráfico 1: fitness ----------------

plt.figure(figsize=(10, 5))
plt.plot(melhores_fitness, linewidth=2, label="Melhor fitness")
plt.axhline(y=valor_otimo, linestyle='--', label=f'Ótimo conhecido = {valor_otimo}')

plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.title("Convergência do fitness - Mochila Binária (p03)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("grafico_fitness_mochila_p03.png", dpi=300)
plt.close()

# ---------------- Gráfico 2: valor ----------------

plt.figure(figsize=(10, 5))
plt.plot(melhores_valores, linewidth=2, label="Melhor valor")
plt.axhline(y=valor_otimo, linestyle='--', label=f'Ótimo conhecido = {valor_otimo}')

plt.xlabel("Geração")
plt.ylabel("Valor total")
plt.title("Convergência do valor total - Mochila Binária (p03)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("grafico_valor_mochila_p03.png", dpi=300)
plt.close()

# ---------------- Gráfico 3: peso ----------------

plt.figure(figsize=(10, 5))
plt.plot(melhores_pesos, linewidth=2, label="Peso da melhor solução")
plt.axhline(y=CAPACIDADE, linestyle='--', label=f'Capacidade = {CAPACIDADE}')

plt.xlabel("Geração")
plt.ylabel("Peso total")
plt.title("Evolução do peso da melhor solução - Mochila Binária (p03)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("grafico_peso_mochila_p03.png", dpi=300)
plt.close()

print("\nGráficos salvos com sucesso:")
print(" - grafico_fitness_mochila_p03.png")
print(" - grafico_valor_mochila_p03.png")
print(" - grafico_peso_mochila_p03.png")