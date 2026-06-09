# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# Algoritmo Evolução Diferencial para VRPTW
# Adaptação: representação contínua (vetor de prioridades) → decodificação em rotas
# -----------------------------------------------------------------------------------

import random
import math
import time
import os
from typing import List, Tuple, Dict, TypedDict

# ========================= PARÂMETROS GLOBAIS =========================

POP_SIZE    = 50
GENERATIONS = 150

F           = 0.5       # Fator de escala da mutação diferencial
CR          = 0.8       # Taxa de crossover
TEMPO_LIMITE = 120.0    # Critério de parada por tempo (segundos)

# Pesos da função fitness (minimizar veículos primeiro, depois distância)
PESO_VEICULOS  = 10_000.0
PESO_DISTANCIA = 1.0

X_MIN, X_MAX = 0.0, 1.0   # Domínio do vetor de prioridades

CAMINHO_INSTANCIA = "/home/gabriel/Desktop/Faculdade/7periodo/BioInspirados/Algoritmos-Bioinspirados/TP2/Instancias_teste/rc2_4_9.txt"

SAIDA = 'TP2/resultados/resultados_diferencial/c1_2_1_de.txt'

random.seed(42)

# ========================= TIPOS =========================

class Cliente(TypedDict):
    id:           int
    x:            float
    y:            float
    demand:       float
    ready_time:   float
    due_date:     float
    service_time: float

# ========================= LEITURA DA INSTÂNCIA =========================

def ler_instancia(caminho: str) -> Tuple[str, int, List[Cliente]]:
    with open(caminho, 'r') as arquivo:
        linhas = [linha.strip() for linha in arquivo if linha.strip() and not linha.startswith('#')]

    nome = linhas[0] if linhas else ''
    capacidade = 0
    clientes: List[Cliente] = []
    lendo_clientes = False

    for indice, linha in enumerate(linhas):
        if linha.upper().startswith('NUMBER'):
            partes = linhas[indice + 1].split()
            if len(partes) >= 2:
                capacidade = int(partes[1])
        if linha.upper().startswith('CUST'):
            lendo_clientes = True
            continue
        if lendo_clientes:
            partes = linha.split()
            if len(partes) >= 7:
                try:
                    clientes.append({
                        'id':           int(partes[0]),
                        'x':            float(partes[1]),
                        'y':            float(partes[2]),
                        'demand':       float(partes[3]),
                        'ready_time':   float(partes[4]),
                        'due_date':     float(partes[5]),
                        'service_time': float(partes[6]),
                    })
                except ValueError:
                    continue

    return nome, capacidade, clientes

# ========================= DISTÂNCIAS =========================

def construir_matriz_distancias(clientes: List[Cliente]) -> List[List[float]]:
    n = len(clientes)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = clientes[i]['x'] - clientes[j]['x']
            dy = clientes[i]['y'] - clientes[j]['y']
            dist[i][j] = math.hypot(dx, dy)
    return dist

# ========================= AVALIAÇÃO =========================

def custo_rota(rota: List[int], dist: List[List[float]]) -> float:
    if len(rota) < 2:
        return 0.0
    total = 0.0
    for i in range(len(rota) - 1):
        total += dist[rota[i]][rota[i + 1]]
    return total


def avaliar_rotas(rotas: List[List[int]], dist: List[List[float]]) -> Tuple[int, float]:
    distancia_total = sum(custo_rota(rota, dist) for rota in rotas)
    return len(rotas), distancia_total


def fitness(nv: int, td: float) -> float:
    return nv * PESO_VEICULOS + td * PESO_DISTANCIA


def comparar_solucoes(sol_a: Tuple[int, float], sol_b: Tuple[int, float]) -> bool:
    """Retorna True se sol_a é melhor que sol_b (menos veículos; a igual, menor distância)."""
    if sol_a[0] != sol_b[0]:
        return sol_a[0] < sol_b[0]
    return sol_a[1] < sol_b[1]

# ========================= VIABILIDADE =========================

def rota_eh_viavel(rota: List[int], clientes: List[Cliente],
                   capacidade: int, dist: List[List[float]]) -> bool:
    tempo = 0.0
    carga = 0.0
    deposito_due = clientes[0]['due_date']

    for i in range(len(rota) - 1):
        origem  = rota[i]
        destino = rota[i + 1]
        chegada = max(tempo + dist[origem][destino], clientes[destino]['ready_time'])
        if chegada > clientes[destino]['due_date']:
            return False
        if destino != 0:
            carga += clientes[destino]['demand']
            if carga > capacidade:
                return False
        tempo = chegada + clientes[destino]['service_time']

    return tempo <= deposito_due

# ========================= DECODIFICAÇÃO =========================
# O indivíduo do DE é um vetor contínuo de comprimento (n_clientes - 1),
# onde cada posição corresponde a um cliente (excluindo o depósito, índice 0).
# A ordem de atendimento é dada pelo ranking crescente dos valores.
# A heurística de inserção sequencial monta as rotas respeitando TW e capacidade.

def decodificar(individuo: List[float], clientes: List[Cliente],
                capacidade: int, dist: List[List[float]]) -> List[List[int]]:
    """
    Converte vetor contínuo em lista de rotas viáveis.
    Clientes são visitados na ordem definida pelo ranking do vetor.
    Cada rota começa e termina no depósito (índice 0).
    """
    n_clientes = len(clientes) - 1          # exclui depósito
    ordem = sorted(range(n_clientes), key=lambda i: individuo[i])
    ordem_clientes = [i + 1 for i in ordem] # índices reais (depósito = 0)

    rotas: List[List[int]] = []
    rota_atual = [0]
    tempo_atual = 0.0
    carga_atual = 0.0

    for cliente_id in ordem_clientes:
        c = clientes[cliente_id]

        chegada_tentativa = max(
            tempo_atual + dist[rota_atual[-1]][cliente_id],
            c['ready_time']
        )

        cabe_carga = (carga_atual + c['demand']) <= capacidade
        cabe_tempo = chegada_tentativa <= c['due_date']

        # Verifica se o veículo consegue voltar ao depósito depois deste cliente
        saida_deposito = chegada_tentativa + c['service_time']
        volta_deposito = saida_deposito + dist[cliente_id][0]
        cabe_retorno   = volta_deposito <= clientes[0]['due_date']

        if cabe_carga and cabe_tempo and cabe_retorno:
            rota_atual.append(cliente_id)
            tempo_atual  = chegada_tentativa + c['service_time']
            carga_atual += c['demand']
        else:
            # Fecha rota atual e abre nova
            rota_atual.append(0)
            rotas.append(rota_atual)
            rota_atual  = [0, cliente_id]
            tempo_atual = max(dist[0][cliente_id], c['ready_time']) + c['service_time']
            carga_atual = c['demand']

    rota_atual.append(0)
    rotas.append(rota_atual)

    return rotas


def fitness_individuo(individuo: List[float], clientes: List[Cliente],
                      capacidade: int, dist: List[List[float]]) -> float:
    rotas = decodificar(individuo, clientes, capacidade, dist)
    nv, td = avaliar_rotas(rotas, dist)
    return fitness(nv, td)

# ========================= OPERADORES DO DE =========================

def criar_individuo(dimensao: int) -> List[float]:
    return [random.uniform(X_MIN, X_MAX) for _ in range(dimensao)]


def criar_populacao(tamanho: int, dimensao: int) -> List[List[float]]:
    return [criar_individuo(dimensao) for _ in range(tamanho)]


def mutacao_dif(ind1: List[float], ind2: List[float],
                ind3: List[float], f: float) -> List[float]:
    return [ind1[i] + f * (ind2[i] - ind3[i]) for i in range(len(ind1))]


def crossover(ind_x: List[float], ind_mutante: List[float], cr: float) -> List[float]:
    # Garante pelo menos um gene do mutante (índice garantido aleatório)
    idx_garantido = random.randint(0, len(ind_x) - 1)
    return [
        ind_mutante[i] if (random.random() < cr or i == idx_garantido) else ind_x[i]
        for i in range(len(ind_x))
    ]


def clipar(individuo: List[float]) -> List[float]:
    """Mantém os valores dentro do domínio [X_MIN, X_MAX]."""
    return [max(X_MIN, min(X_MAX, v)) for v in individuo]


def mutar_populacao(populacao: List[List[float]], f: float, cr: float,
                    clientes: List[Cliente], capacidade: int,
                    dist: List[List[float]]) -> List[List[float]]:
    n = len(populacao)
    nova_pop = []

    for i in range(n):
        # Seleciona 3 índices distintos e diferentes de i
        candidatos = [j for j in range(n) if j != i]
        r1, r2, r3 = random.sample(candidatos, 3)

        mutante  = mutacao_dif(populacao[r1], populacao[r2], populacao[r3], f)
        mutante  = clipar(mutante)
        trial    = crossover(populacao[i], mutante, cr)

        # Seleção: mantém o melhor entre o atual e o trial
        fit_atual = fitness_individuo(populacao[i], clientes, capacidade, dist)
        fit_trial = fitness_individuo(trial, clientes, capacidade, dist)

        nova_pop.append(trial if fit_trial <= fit_atual else populacao[i])

    return nova_pop

# ========================= SAÍDA =========================

def formatar_saida(nome: str, rotas: List[List[int]], dist: List[List[float]]) -> str:
    nv, td = avaliar_rotas(rotas, dist)
    linhas = [
        f"Nome da instância: {nome}",
        f"Número de veículos: {nv}",
        f"Distância total: {td:.4f}",
        "Rotas:",
    ]
    for idx, rota in enumerate(rotas, 1):
        # Remove depósito (0) do início e fim para exibição
        clientes_rota = [str(c) for c in rota if c != 0]
        linhas.append(f"Rota {idx}: {' '.join(clientes_rota)}")
    return '\n'.join(linhas)


def salvar_resultado(caminho_saida: str, conteudo: str) -> None:
    os.makedirs(os.path.dirname(caminho_saida) or '.', exist_ok=True)
    with open(caminho_saida, 'w') as f:
        f.write(conteudo)
    print(f"\nResultado salvo em: {caminho_saida}")

# ========================= EXECUÇÃO PRINCIPAL =========================

caminho_instancia = CAMINHO_INSTANCIA
caminho_saida = SAIDA

nome, capacidade, clientes = ler_instancia(caminho_instancia)
dist = construir_matriz_distancias(clientes)
dimensao = len(clientes) - 1  # um gene por cliente (sem depósito)

print(f"\n{'='*60}")
print(f"Instância : {nome}  |  Clientes: {dimensao}  |  Capacidade: {capacidade}")
print(f"POP={POP_SIZE}  GER={GENERATIONS}  F={F}  CR={CR}  T_MAX={TEMPO_LIMITE}s")
print('='*60)

# --- Inicialização ---
populacao = criar_populacao(POP_SIZE, dimensao)

MELHOR_INDIVIDUO  = None
MELHOR_NUM_ROTAS  = math.inf
MELHOR_DISTANCIA  = math.inf
MELHOR_FITNESS    = math.inf

tempo_inicio = time.time()

# --- Loop de gerações ---
for g in range(1, GENERATIONS + 1):

    # Critério de parada por tempo
    tempo_atual = time.time() - tempo_inicio
    if tempo_atual >= TEMPO_LIMITE:
        print(f"\n[PARADA] Tempo limite de {TEMPO_LIMITE}s atingido na geração {g}.")
        break

    populacao = mutar_populacao(populacao, F, CR, clientes, capacidade, dist)

    # Melhor da geração
    fits_geracao = [
        (fitness_individuo(ind, clientes, capacidade, dist), ind)
        for ind in populacao
    ]
    fits_geracao.sort(key=lambda x: x[0])
    fit_ger, melhor_ind_ger = fits_geracao[0]

    rotas_ger   = decodificar(melhor_ind_ger, clientes, capacidade, dist)
    nv_ger, td_ger = avaliar_rotas(rotas_ger, dist)

    # Atualiza melhor global
    if fit_ger < MELHOR_FITNESS:
        MELHOR_FITNESS   = fit_ger
        MELHOR_INDIVIDUO = melhor_ind_ger[:]
        MELHOR_NUM_ROTAS = nv_ger
        MELHOR_DISTANCIA = td_ger

    print(
        f"Geração {g:>4} | "
        f"Veículos geração: {nv_ger:>3} | "
        f"Distância geração: {td_ger:>10.4f} | "
        f"Melhor global: {MELHOR_NUM_ROTAS}/{MELHOR_DISTANCIA:.4f} | "
        f"Tempo: {tempo_atual:.2f}s"
    )

# --- Resultado final ---
rotas_finais = decodificar(MELHOR_INDIVIDUO, clientes, capacidade, dist)
conteudo = formatar_saida(nome, rotas_finais, dist)

print(f"\n{'='*60}")
print(conteudo)

if caminho_saida:
    salvar_resultado(caminho_saida, conteudo)

