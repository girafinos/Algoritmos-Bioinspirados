# -----------------------------------------------------------------------------------
# Evolução Diferencial (DE) para VRPTW
# Adaptado do DE simples original para resolver o problema de roteamento de veículos
# com janelas de tempo (VRPTW)
# -----------------------------------------------------------------------------------

import os
import math
import random
import time

# -------------------- Parâmetros globais --------------------

POP_SIZE   = 50
GERACOES   = 150
F          = 0.3    # fator de mutação
CR         = 0.3   # taxa de crossover
SEMENTE    = 1

# Pesos da função de fitness
ALPHA = 1000   # peso para número de veículos
BETA  = 1       # peso para distância total
GAMMA = 10     # peso para penalidades (atrasos e excesso de carga)

# Limites do espaço contínuo (random keys)
X_MIN, X_MAX = 0.0, 1.0

CAMINHO_INSTANCIA = "TP2/instâncias_teste/rc2_4_9.txt"
PASTA_RESULTADOS  = "TP2/DE/resultados"

# Pare após este número de segundos (None para desativar)
MAX_SECONDS = 480

# -------------------- Melhor solução global --------------------

MELHOR_SOLUCAO   = None
MELHOR_NUM_ROTAS = math.inf
MELHOR_DISTANCIA = math.inf
MELHOR_GERACAO   = -1
MELHOR_TEMPO     = 0.0

# -------------------- Leitura de instância --------------------

def ler_instancia(caminho):
    with open(caminho, 'r') as f:
        linhas = [l.strip() for l in f if l.strip()]

    nome = linhas[0]
    capacidade = None

    for i, linha in enumerate(linhas):
        if linha.upper().startswith('NUMBER'):
            partes = linhas[i + 1].split()
            capacidade = int(partes[1])

    clientes = []
    lendo = False

    for linha in linhas:
        if linha.upper().startswith('CUST'):
            lendo = True
            continue
        if lendo:
            partes = linha.split()
            if len(partes) >= 7:
                clientes.append({
                    'id':           int(partes[0]),
                    'x':            float(partes[1]),
                    'y':            float(partes[2]),
                    'demand':       float(partes[3]),
                    'ready_time':   float(partes[4]),
                    'due_date':     float(partes[5]),
                    'service_time': float(partes[6]),
                })

    return nome, capacidade, clientes

# -------------------- Distâncias --------------------

def construir_matriz_distancias(clientes):
    n = len(clientes)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = clientes[i]['x'] - clientes[j]['x']
            dy = clientes[i]['y'] - clientes[j]['y']
            dist[i][j] = math.hypot(dx, dy)
    return dist

# -------------------- Decodificação -----------------------

def keys_para_permutacao(keys):
    # Ordena os índices 1..n pelo valor do key correspondente
    return sorted(range(1, len(keys) + 1), key=lambda i: keys[i - 1])

#decodificador melhor porem mais lento
#decodificador anterior ela muito guloso e gerava muitas rotas desnecessariamente
def decodificar(permutacao, clientes, capacidade, dist):
    rotas = []

    for cid in permutacao:
        melhor_rota = None
        melhor_pos  = None
        melhor_custo = float('inf')

        cliente = clientes[cid]

        # tenta inserir em todas as rotas existentes
        for r_idx, rota in enumerate(rotas):
            for pos in range(1, len(rota)):  # posições internas
                nova_rota = rota[:pos] + [cid] + rota[pos:]

                if rota_viavel(nova_rota, clientes, capacidade, dist):
                    custo = custo_rota(nova_rota, dist)

                    if custo < melhor_custo:
                        melhor_custo = custo
                        melhor_rota  = r_idx
                        melhor_pos   = pos

        # se achou posição viável → insere
        if melhor_rota is not None:
            rotas[melhor_rota].insert(melhor_pos, cid)
        else:
            # cria nova rota
            rotas.append([0, cid, 0])

    return rotas

    
# -------------------- Avaliação --------------------

def custo_rota(rota, dist):
    return sum(dist[rota[i]][rota[i + 1]] for i in range(len(rota) - 1))


def avaliar_rotas(rotas, dist):
    td = sum(custo_rota(r, dist) for r in rotas)
    return len(rotas), td


def compute_penalty(rotas, clientes, capacidade, dist):
    total_delay  = 0.0
    total_excess = 0.0

    for rota in rotas:
        carga = 0.0
        tempo = 0.0
        for i in range(len(rota) - 1):
            a, b  = rota[i], rota[i + 1]
            cust  = clientes[b]
            arrival = tempo + dist[a][b]
            inicio  = max(arrival, cust['ready_time'])
            if inicio > cust['due_date']:
                total_delay += inicio - cust['due_date']
            carga += cust['demand']
            tempo  = inicio + cust['service_time']
        if carga > capacidade:
            total_excess += carga - capacidade

    return total_delay, total_excess


def fitness(individuo):
    """
    Função de fitness: recebe o vetor contínuo (random keys),
    decodifica em rotas e avalia.
    Mantém a mesma assinatura do DE original: fitness(individuo) → float.
    """
    perm  = keys_para_permutacao(individuo)
    rotas = decodificar(perm, CLIENTES, CAPACIDADE, DIST)
    nv, td = avaliar_rotas(rotas, DIST)
    delay, excess = compute_penalty(rotas, CLIENTES, CAPACIDADE, DIST)
    penalidade = delay + excess
    return ALPHA * nv + BETA * td + GAMMA * penalidade

#funcao auxiliar criada para a decodificacao
def rota_viavel(rota, clientes, capacidade, dist):
    carga = 0.0
    tempo = 0.0

    for i in range(len(rota) - 1):
        a, b = rota[i], rota[i + 1]
        cliente = clientes[b]

        chegada = tempo + dist[a][b]
        inicio  = max(chegada, cliente['ready_time'])

        if inicio > cliente['due_date']:
            return False

        carga += cliente['demand']
        tempo  = inicio + cliente['service_time']

    return carga <= capacidade

# -------------------- Funções do DE (idênticas ao original) --------------------

def criar_individuo():
    """Cria um vetor de random keys com (n_clientes - 1) dimensões."""
    return [random.uniform(X_MIN, X_MAX) for _ in range(DIMENSAO)]


def criar_populacao():
    pop = []
    for _ in range(POP_SIZE):
        pop.append(criar_individuo())
    return pop


def eh_menor(ind1, ind2):
    if fitness(ind1) < fitness(ind2):
        return ind1
    else:
        return ind2


def crossover(ind_x, ind_mutante, CR):
    vetor_teste = []
    for i in range(len(ind_mutante)):
        sorteio = random.uniform(0, 1)
        if sorteio >= CR:
            vetor_teste.append(ind_mutante[i])
        else:
            vetor_teste.append(ind_x[i])
    return vetor_teste


def mutacao_dif(ind_x, ind1, ind2, ind3, F, CR):
    ind_mutante = []
    for i in range(len(ind1)):
        # Clamp para manter dentro de [X_MIN, X_MAX]
        valor = ind1[i] + F * (ind2[i] - ind3[i])
        valor = min(max(valor, X_MIN), X_MAX)
        ind_mutante.append(valor)
    return crossover(ind_x, ind_mutante, CR)


#funcao alterada para não repetir veiculos na mesma rota e nao gerar ruidos aleatorios
#adicionado elitismo na mutacao
def mutar_pop(populacao, F, CR, elite_size=5):
    tamanho = len(populacao)

    # ordena população por fitness
    pop_ordenada = sorted(populacao, key=lambda ind: fitness(ind))

    # guarda elite
    elite = pop_ordenada[:elite_size]

    nova_pop = []

    for i in range(tamanho):
        indices = list(range(tamanho))
        indices.remove(i)

        r1, r2, r3 = random.sample(indices, 3)

        ind_x = populacao[i]
        ind1 = populacao[r1]
        ind2 = populacao[r2]
        ind3 = populacao[r3]

        trial = mutacao_dif(ind_x, ind1, ind2, ind3, F, CR)

        if fitness(trial) < fitness(ind_x):
            nova_pop.append(trial)
        else:
            nova_pop.append(ind_x)

    nova_pop_ordenada = sorted(nova_pop, key=lambda ind: fitness(ind), reverse=True)

    for i in range(elite_size):
        nova_pop_ordenada[i] = elite[i]

    return nova_pop_ordenada

# -------------------- Output --------------------

def formatar_rotas(rotas):
    linhas = []
    for i, r in enumerate(rotas, 1):
        caminho = ' '.join(str(c) for c in r[1:-1])
        linhas.append(f'Rota {i}: {caminho}')
    return '\n'.join(linhas)


def salvar_resultado(nome, rotas, nv, td, geracao, tempo, tempo_total):
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    caminho = os.path.join(PASTA_RESULTADOS, f"{nome}_de.txt")
    with open(caminho, 'w') as f:
        f.write(
            f'Nome da instância: {nome}\n'
            f'Melhor encontrado na geração: {geracao}\n'
            f'Tempo até melhor resultado: {tempo:.2f}s\n'
            f'Tempo total de execução: {tempo_total:.2f}s\n'
            f'Número de veículos: {nv}\n'
            f'Distância total: {td:.4f}\n'
            'Rotas:\n' + formatar_rotas(rotas)
        )
    return caminho

# -------------------- Execução --------------------

random.seed(SEMENTE)
inicio = time.time()

# Carrega instância em variáveis globais para que fitness() acesse diretamente,
# mantendo a assinatura original fitness(individuo) sem parâmetros extras.
NOME, CAPACIDADE, CLIENTES = ler_instancia(CAMINHO_INSTANCIA)
DIST     = construir_matriz_distancias(CLIENTES)
DIMENSAO = len(CLIENTES) - 1   # um key por cliente (excluindo depósito)

# Cria e avalia população inicial
populacao = criar_populacao()
populacao = sorted(populacao, key=fitness)

melhor_global    = populacao[0]
MELHOR_NUM_ROTAS = len(decodificar(keys_para_permutacao(melhor_global), CLIENTES, CAPACIDADE, DIST))
MELHOR_DISTANCIA = avaliar_rotas(decodificar(keys_para_permutacao(melhor_global), CLIENTES, CAPACIDADE, DIST), DIST)[1]
MELHOR_SOLUCAO   = decodificar(keys_para_permutacao(melhor_global), CLIENTES, CAPACIDADE, DIST)

stop_early = False

for geracao in range(1, GERACOES + 1):

    if MAX_SECONDS is not None and (time.time() - inicio) >= MAX_SECONDS:
        print(f"Limite de tempo atingido: {MAX_SECONDS}s — interrompendo na geração {geracao}")
        break

    populacao = mutar_pop(populacao, F, CR)

    populacao_aux = sorted(populacao, key=fitness)
    melhor_geracao = populacao_aux[0]

    # Decodifica melhor da geração para exibir métricas
    perm_ger           = keys_para_permutacao(melhor_geracao)
    rotas_ger          = decodificar(perm_ger, CLIENTES, CAPACIDADE, DIST)
    melhor_nv_geracao, melhor_td_geracao = avaliar_rotas(rotas_ger, DIST)

    # Atualiza melhor global se houver melhora
    if fitness(melhor_geracao) < fitness(melhor_global):
        melhor_global    = melhor_geracao
        MELHOR_SOLUCAO   = rotas_ger
        MELHOR_NUM_ROTAS = melhor_nv_geracao
        MELHOR_DISTANCIA = melhor_td_geracao
        MELHOR_GERACAO   = geracao
        MELHOR_TEMPO     = time.time() - inicio

    tempo_atual = time.time() - inicio

    print(
        f'Geração {geracao} | Veículos geração: {melhor_nv_geracao} | '
        f'Distância geração: {melhor_td_geracao:.4f} | '
        f'Melhor global: {MELHOR_NUM_ROTAS}/{MELHOR_DISTANCIA:.4f} | '
        f'Tempo: {tempo_atual:.2f}s'
    )

tempo_total = time.time() - inicio

caminho = salvar_resultado(
    NOME,
    MELHOR_SOLUCAO,
    MELHOR_NUM_ROTAS,
    MELHOR_DISTANCIA,
    MELHOR_GERACAO,
    MELHOR_TEMPO,
    tempo_total
)

print("\nMelhor solução global:")
print(f'Número de veículos: {MELHOR_NUM_ROTAS}')
print(f'Distância total: {MELHOR_DISTANCIA:.4f}')
print(f'Geração: {MELHOR_GERACAO} | Tempo: {MELHOR_TEMPO:.2f}s')
print(f'Resultado salvo em: {caminho}')
print(f'Tempo total: {tempo_total:.2f}s')