import random
import matplotlib.pyplot as plt

# ---------------- Parâmetros ----------------
VAR_MUT_CHANCE = 1      # chance de mutação por bit (%)
VAR_RANGE = 50          # tamanho da população
VAR_ITERATIONS = 20  # número de gerações

LIMITE_INF = -10
LIMITE_SUP = 10

BITS_POR_VARIAVEL = 16  # 16 bits para x1 e 16 bits para x2

z = []
x1 = [random.uniform(LIMITE_INF, LIMITE_SUP) for _ in range(VAR_RANGE)]
x2 = [random.uniform(LIMITE_INF, LIMITE_SUP) for _ in range(VAR_RANGE)]

# ---------------- Função objetivo ----------------
def funcao_obj_g11(x1, x2):
    fx = (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2
    return fx

# ---------------- Conversão binária (ponto fixo) ----------------
def real_to_bin(n, minimo=LIMITE_INF, maximo=LIMITE_SUP, bits=BITS_POR_VARIAVEL):
    # Garante que o valor esteja dentro do intervalo
    n = max(minimo, min(maximo, n))

    max_int = (2**bits) - 1
    inteiro = round((n - minimo) * max_int / (maximo - minimo))
    return format(inteiro, f'0{bits}b')

def bin_to_real(bin_str, minimo=LIMITE_INF, maximo=LIMITE_SUP, bits=BITS_POR_VARIAVEL):
    max_int = (2**bits) - 1
    inteiro = int(bin_str, 2)
    real = minimo + (inteiro * (maximo - minimo) / max_int)
    return real

# ---------------- Cruzamento ----------------
def cruzamento(pai1, pai2):
    cut_point = random.randint(1, BITS_POR_VARIAVEL - 1)

    pai1bin = [real_to_bin(pai1[0]), real_to_bin(pai1[1])]
    pai2bin = [real_to_bin(pai2[0]), real_to_bin(pai2[1])]

    # x1
    filho1_x1 = pai1bin[0][:cut_point] + pai2bin[0][cut_point:]
    filho2_x1 = pai2bin[0][:cut_point] + pai1bin[0][cut_point:]

    # x2
    filho1_x2 = pai1bin[1][:cut_point] + pai2bin[1][cut_point:]
    filho2_x2 = pai2bin[1][:cut_point] + pai1bin[1][cut_point:]

    filho1 = [bin_to_real(filho1_x1), bin_to_real(filho1_x2)]
    filho2 = [bin_to_real(filho2_x1), bin_to_real(filho2_x2)]

    return filho1, filho2

# ---------------- Seleção por torneio ----------------
def selecionar_pai(z, x1, x2):
    i1 = random.randint(0, len(z) - 1)
    i2 = random.randint(0, len(z) - 1)

    if z[i1] < z[i2]:
        return [x1[i1], x2[i1]]
    else:
        return [x1[i2], x2[i2]]

# ---------------- Mutação ----------------
def mutacao(populacao):
    nova_populacao = []

    for i in range(VAR_RANGE):
        individuo = populacao[i]

        individuo_bin = [
            real_to_bin(individuo[0]),
            real_to_bin(individuo[1])
        ]

        individuo_bin = [list(individuo_bin[0]), list(individuo_bin[1])]

        for j in range(BITS_POR_VARIAVEL):
            if random.randint(1, 100) <= VAR_MUT_CHANCE:
                individuo_bin[0][j] = '1' if individuo_bin[0][j] == '0' else '0'

            if random.randint(1, 100) <= VAR_MUT_CHANCE:
                individuo_bin[1][j] = '1' if individuo_bin[1][j] == '0' else '0'

        bin_x1 = ''.join(individuo_bin[0])
        bin_x2 = ''.join(individuo_bin[1])

        individuo_mutado = [bin_to_real(bin_x1), bin_to_real(bin_x2)]
        nova_populacao.append(individuo_mutado)

    return nova_populacao

# ---------------- Iteração ----------------
def iteracao(z, x1, x2):
    nova_populacao = []

    for _ in range(VAR_RANGE // 2):
        pai1 = selecionar_pai(z, x1, x2)
        pai2 = selecionar_pai(z, x1, x2)

        filho1, filho2 = cruzamento(pai1, pai2)
        nova_populacao.append(filho1)
        nova_populacao.append(filho2)

    nova_populacao = mutacao(nova_populacao)
    return nova_populacao

# ---------------- Melhor indivíduo ----------------
def melhor_individuo(x1, x2, z):
    index = z.index(min(z))
    return x1[index], x2[index], z[index]

# ---------------- Execução principal ----------------
for i in range(VAR_RANGE):
    result = funcao_obj_g11(x1[i], x2[i])
    z.append(result)

melhores_por_geracao = []

x1_best, x2_best, z_best = melhor_individuo(x1, x2, z)
melhores_por_geracao.append(z_best)

for i in range(VAR_ITERATIONS):
    nova_populacao = iteracao(z, x1, x2)

    x1 = [ind[0] for ind in nova_populacao]
    x2 = [ind[1] for ind in nova_populacao]

    z = [funcao_obj_g11(x1[j], x2[j]) for j in range(VAR_RANGE)]

    x1_best, x2_best, z_best = melhor_individuo(x1, x2, z)
    melhores_por_geracao.append(z_best)

    
    print(f"Geração {i + 1}: melhor z = {z_best:.10f} (x1 = {x1_best:.6f}, x2 = {x2_best:.6f})")

plt.figure(figsize=(12, 6))
plt.plot(melhores_por_geracao, linewidth=2)
plt.xlabel('Geração')
plt.ylabel('Melhor Fitness')
plt.title('Evolução do Melhor Fitness ao Longo das Gerações')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('grafico_evolucao_v0.png', dpi=300)

print("\nGráfico salvo como 'grafico_evolucao_v0.png'")
print(f"Melhor valor final: {melhores_por_geracao[-1]:.10f}")
print(f"Melhor indivíduo final: x1 = {x1_best:.6f}, x2 = {x2_best:.6f}")

plt.show()