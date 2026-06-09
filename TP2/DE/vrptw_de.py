# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# Evolução Diferencial (DE) para VRPTW
# -----------------------------------------------------------------------------------

import os
import math
import random
import time
from typing import List, Dict, Tuple, Optional

# -------------------- Parâmetros globais --------------------

POP_SIZE = 50
GERACOES = 150
F = 0.3
CR = 0.3
SEMENTE = 1

ALPHA = 10000 #peso para número de veículos
BETA = 1 #peso para distância total
GAMMA = 100 #peso para penalidades (atrasos e excesso de carga)

X_MIN, X_MAX = 0.0, 1.0

CAMINHO_INSTANCIA = "TP2/Instancias_teste/c1_2_1.txt"
PASTA_RESULTADOS = "TP2/resultados/resultados_diferencial"

# Pare após este número de segundos (None para desativar)
MAX_SECONDS = 600

# -------------------- Melhor solução global --------------------

MELHOR_SOLUCAO: Optional[List[List[int]]] = None
MELHOR_NUM_ROTAS = float('inf')
MELHOR_DISTANCIA = float('inf')
MELHOR_GERACAO = -1
MELHOR_TEMPO = 0.0

Cliente = Dict[str, float]

# -------------------- Leitura de instância --------------------

def ler_instancia(caminho: str) -> Tuple[str, int, List[Cliente]]:
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
                    'id': int(partes[0]),
                    'x': float(partes[1]),
                    'y': float(partes[2]),
                    'demand': float(partes[3]),
                    'ready_time': float(partes[4]),
                    'due_date': float(partes[5]),
                    'service_time': float(partes[6]),
                })

    return nome, capacidade, clientes

# -------------------- Distâncias --------------------

def construir_matriz_distancias(clientes):
    n = len(clientes)
    dist = [[0.0]*n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            dx = clientes[i]['x'] - clientes[j]['x']
            dy = clientes[i]['y'] - clientes[j]['y']
            dist[i][j] = math.hypot(dx, dy)

    return dist

# -------------------- Decodificação --------------------

def keys_para_permutacao(keys):
    return sorted(range(1, len(keys)+1), key=lambda i: keys[i-1])
'''
def decodificar(permutacao, clientes, capacidade, dist):
    deposito_due = clientes[0]['due_date']
    rotas = []

    rota = [0]
    carga = 0.0
    tempo = 0.0

    for cid in permutacao:
        cliente = clientes[cid]
        prev = rota[-1]

        chegada = tempo + dist[prev][cid]
        inicio = max(chegada, cliente['ready_time'])

        if (inicio <= cliente['due_date'] and
            carga + cliente['demand'] <= capacidade and
            inicio + cliente['service_time'] + dist[cid][0] <= deposito_due):

            rota.append(cid)
            carga += cliente['demand']
            tempo = inicio + cliente['service_time']

        else:
            rotas.append(rota + [0])
            rota = [0, cid]
            carga = cliente['demand']
            tempo = max(dist[0][cid], cliente['ready_time']) + cliente['service_time']

    rotas.append(rota + [0])
    return rotas
'''

#novo decode

def try_insertion(route, cid, customers, capacity, dist):
    # Tenta inserir o cliente em todas as posições possíveis e simula apenas
    # a rota resultante (checando janelas/loads) — evita chamadas redundantes.
    best_route = None
    best_cost = math.inf

    for i in range(1, len(route)):
        new_route = route[:i] + [cid] + route[i:]

        # Simula a rota incrementalmente e calcula custo ao mesmo tempo
        load = 0.0
        t = 0.0
        feasible = True
        cost = 0.0

        for k in range(len(new_route) - 1):
            a = new_route[k]
            b = new_route[k + 1]
            arrival = t + dist[a][b]
            start = max(arrival, customers[b]['ready_time'])
            if start > customers[b]['due_date']:
                feasible = False
                break
            if b != 0:
                load += customers[b]['demand']
                if load > capacity:
                    feasible = False
                    break
            cost += dist[a][b]
            t = start + customers[b]['service_time']

        if feasible and cost < best_cost:
            best_cost = cost
            best_route = new_route

    return best_route, best_cost


def decode(permutation, customers, capacity, dist):
    routes = []

    for cid in permutation:
        best_route_idx = -1
        best_route = None
        best_cost = math.inf

        # tenta inserir em rotas existentes
        for r_idx, route in enumerate(routes):
            new_route, cost = try_insertion(route, cid, customers, capacity, dist)
            if new_route is not None and cost < best_cost:
                best_cost = cost
                best_route = new_route
                best_route_idx = r_idx

        # se conseguiu inserir em alguma rota
        if best_route is not None:
            routes[best_route_idx] = best_route
        else:
            # cria nova rota
            routes.append([0, cid, 0])

    # Busca local rápida: 2-opt por rota (melhora sequência interna)
    return local_search_routes(routes, customers, capacity, dist)


def two_opt_route(route, customers, capacity, dist):
    # Aplica um único passe de 2-opt (aceita melhorias factíveis)
    n = len(route)
    if n <= 4:
        return route

    improved = True
    current = route[:]
    current_cost = custo_rota(current, dist)

    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                cand = current[:i] + current[i:j + 1][::-1] + current[j + 1:]
                if rota_viavel(cand, customers, capacity, dist):
                    cand_cost = custo_rota(cand, dist)
                    if cand_cost + 1e-9 < current_cost:
                        current = cand
                        current_cost = cand_cost
                        improved = True
                        break
            if improved:
                break
    return current


def local_search_routes(routes, customers, capacity, dist):
    new_routes = []
    for r in routes:
        r_opt = two_opt_route(r, customers, capacity, dist)
        new_routes.append(r_opt)
    return new_routes

# -------------------- Avaliação --------------------

def custo_rota(rota, dist):
    return sum(dist[rota[i]][rota[i+1]] for i in range(len(rota)-1))

def avaliar_rotas(rotas, dist):
    td = sum(custo_rota(r, dist) for r in rotas)
    return len(rotas), td

def compute_penalty(routes, customers, capacity, dist):
    total_delay = 0.0
    total_excess = 0.0

    for route in routes:
        load = 0.0
        t = 0.0

        for i in range(len(route) - 1):
            a = route[i]
            b = route[i + 1]
            cust = customers[b]

            arrival = t + dist[a][b]
            start = max(arrival, cust['ready_time'])

            # atraso
            if start > cust['due_date']:
                total_delay += start - cust['due_date']

            load += cust['demand']

            t = start + cust['service_time']

        # excesso de capacidade
        if load > capacity:
            total_excess += load - capacity

    return total_delay, total_excess 

def fitness(num_routes, total_distance, routes, customers, capacity, dist):
   

    delay, excess = compute_penalty(routes, customers, capacity, dist)

    penalty = delay + excess

    return ALPHA * num_routes + BETA * total_distance + GAMMA * penalty

# -------------------- Viabilidade --------------------

def rota_viavel(rota, clientes, capacidade, dist):
    carga = 0.0
    tempo = 0.0
    deposito_due = clientes[0]['due_date']

    for i in range(len(rota)-1):
        a, b = rota[i], rota[i+1]
        cliente = clientes[b]

        tempo = max(tempo + dist[a][b], cliente['ready_time'])

        if tempo > cliente['due_date']:
            return False

        carga += cliente['demand']
        if carga > capacidade:
            return False

        tempo += cliente['service_time']

    return tempo <= deposito_due

def solucao_viavel(rotas, clientes, capacidade, dist):
    return all(rota_viavel(r, clientes, capacidade, dist) for r in rotas)

# -------------------- DE --------------------

def criar_individuo(n):
    return [random.uniform(X_MIN, X_MAX) for _ in range(n-1)]

def mutacao(a, b, c):
    return [min(max(a[i] + F*(b[i]-c[i]), X_MIN), X_MAX) for i in range(len(a))]

def crossover(target, mutant):
    trial = []
    j_rand = random.randrange(len(target))

    for j in range(len(target)):
        if random.random() < CR or j == j_rand:
            trial.append(mutant[j])
        else:
            trial.append(target[j])

    return trial

def avaliar_individuo(keys, clientes, capacidade, dist):
    perm = keys_para_permutacao(keys)
    rotas = decode(perm, clientes, capacidade, dist)

    nv, td = avaliar_rotas(rotas, dist)

    fit = fitness(nv, td, rotas, clientes, capacidade, dist)

    return {
        'keys': keys,
        'rotas': rotas,
        'nv': nv,
        'td': td,
        'fit': fit
    }

# -------------------- Output --------------------

def formatar_rotas(rotas):
    linhas = []
    for i, r in enumerate(rotas, 1):
        caminho = ' '.join(str(c) for c in r[1:-1])
        linhas.append(f'Rota {i}: {caminho}')
    return '\n'.join(linhas)

def salvar_resultado(nome, rotas, nv, td, geracao, tempo):
    pasta = PASTA_RESULTADOS
    os.makedirs(pasta, exist_ok=True)

    caminho = os.path.join(pasta, f"{nome}_de.txt")

    with open(caminho, 'w') as f:
        f.write(
            f'Nome da instância: {nome}\n'
            f'Melhor geração: {geracao}\n'
            f'Tempo até melhor: {tempo:.2f}s\n'
            f'Veículos: {nv}\n'
            f'Distância: {td:.4f}\n'
            'Rotas:\n' + formatar_rotas(rotas)
        )

    return caminho

# -------------------- Execução --------------------

random.seed(SEMENTE)

nome, capacidade, clientes = ler_instancia(CAMINHO_INSTANCIA)
dist = construir_matriz_distancias(clientes)
n = len(clientes)

populacao = [avaliar_individuo(criar_individuo(n), clientes, capacidade, dist) for _ in range(POP_SIZE)]

melhor_global = min(populacao, key=lambda x: x['fit'])
MELHOR_NUM_ROTAS = melhor_global['nv']
MELHOR_DISTANCIA = melhor_global['td']

inicio = time.time()
stop_early = False

for g in range(1, GERACOES+1):
    nova_pop = []

    # Verifica limite de tempo antes de começar a geração
    if MAX_SECONDS is not None and (time.time() - inicio) >= MAX_SECONDS:
        print(f"Limite de tempo atingido: {MAX_SECONDS}s — interrompendo em geração {g}")
        break

    for i, target in enumerate(populacao):
        idxs = list(range(POP_SIZE))
        idxs.remove(i)
        a, b, c = random.sample(idxs, 3)

        mutant = mutacao(populacao[a]['keys'], populacao[b]['keys'], populacao[c]['keys'])
        trial_keys = crossover(target['keys'], mutant)
        trial = avaliar_individuo(trial_keys, clientes, capacidade, dist)

        nova_pop.append(trial if trial['fit'] <= target['fit'] else target)

        # Verifica limite de tempo também durante a geração
        if MAX_SECONDS is not None and (time.time() - inicio) >= MAX_SECONDS:
            stop_early = True
            break

    populacao = nova_pop

    if stop_early:
        print(f"Interrompido por tempo após geração {g}")
        break

    melhor_geracao = min(populacao, key=lambda x: x['fit'])

    if melhor_geracao['fit'] < melhor_global['fit']:
        melhor_global = melhor_geracao
        MELHOR_NUM_ROTAS = melhor_global['nv']
        MELHOR_DISTANCIA = melhor_global['td']
        MELHOR_SOLUCAO = melhor_global['rotas']
        MELHOR_GERACAO = g
        MELHOR_TEMPO = time.time() - inicio

    tempo_atual = time.time() - inicio

    print(
        f'Geração {g} | Veículos geração: {melhor_geracao["nv"]} | '
        f'Distância geração: {melhor_geracao["td"]:.4f} | '
        f'Melhor global: {MELHOR_NUM_ROTAS}/{MELHOR_DISTANCIA:.4f} | '
        f'Tempo: {tempo_atual:.2f}s'
    )

tempo_total = time.time() - inicio

caminho = salvar_resultado(
    nome,
    MELHOR_SOLUCAO,
    MELHOR_NUM_ROTAS,
    MELHOR_DISTANCIA,
    MELHOR_GERACAO,
    MELHOR_TEMPO
)

print("\nMelhor solução global:")
print(f'Veículos: {MELHOR_NUM_ROTAS}')
print(f'Distância: {MELHOR_DISTANCIA:.4f}')
print(f'Geração: {MELHOR_GERACAO} | Tempo: {MELHOR_TEMPO:.2f}s')
print(f'Resultado salvo em: {caminho}')
print(f'Tempo total: {tempo_total:.2f}s')