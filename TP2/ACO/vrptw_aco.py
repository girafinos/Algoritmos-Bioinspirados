# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# Algoritmo de Colônia de Formigas para o VRPTW (Vehicle Routing Problem with Time Windows)
# -----------------------------------------------------------------------------------

import os
import math
import random
import time
from typing import Dict, List, Optional, Tuple

# -------------------- Parâmetros globais --------------------
#ALPHA = peso do feromônio (0 a 5)
#BETA = peso da heuristica gulosa (0 a 5)
#RHO = taxa de evaporação do feromônio (0.1 a 0.9)
#Q = quantidade de feromônio depositada
POP_SIZE = 50
ITERACOES = 150

ALPHA = 2.0
BETA = 3.0
RHO = 0.6
FEROMONIO_INICIAL = 0.001
Q = 1.0
SEMENTE = 1

TAU_MIN = 1e-4
TAU_MAX = 100.0

PATIENCE = 70
TEMPO_MAXIMO = 480.0  

PESO_VEICULOS = 50.0
PESO_DISTANCIA = 1.0

MELHOR_SOLUCAO: Optional[List[List[int]]] = None
MELHOR_NUM_ROTAS = float('inf')
MELHOR_DISTANCIA = float('inf')
MELHOR_GERACAO = -1
MELHOR_TEMPO = 0.0
ULTIMA_MELHORA = 0

Cliente = Dict[str, float]

CAMINHO_INSTANCIA = "/home/gabriel/Desktop/Faculdade/7periodo/BioInspirados/Algoritmos-Bioinspirados/TP2/Instancias_teste/rc2_4_9.txt"

# -------------------- Leitura de instância --------------------

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
                        'id': int(partes[0]),
                        'x': float(partes[1]),
                        'y': float(partes[2]),
                        'demand': float(partes[3]),
                        'ready_time': float(partes[4]),
                        'due_date': float(partes[5]),
                        'service_time': float(partes[6]),
                    })
                except ValueError:
                    continue

    return nome, capacidade, clientes

# -------------------- Distância --------------------

def construir_matriz_distancias(clientes: List[Cliente]) -> List[List[float]]:
    n = len(clientes)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = clientes[i]['x'] - clientes[j]['x']
            dy = clientes[i]['y'] - clientes[j]['y']
            dist[i][j] = math.hypot(dx, dy)
    return dist

# -------------------- Avaliação --------------------

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
    if sol_a[0] != sol_b[0]:
        return sol_a[0] < sol_b[0]
    return sol_a[1] < sol_b[1]

# -------------------- Viabilidade de rota --------------------

def rota_eh_viavel(rota: List[int], clientes: List[Cliente], capacidade: int, dist: List[List[float]]) -> bool:
    tempo = 0.0
    carga = 0.0
    deposito_due = clientes[0]['due_date']

    for i in range(len(rota) - 1):
        origem = rota[i]
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


def pode_inserir_cidade(
    atual: int,
    candidato: int,
    carga: float,
    tempo: float,
    clientes: List[Cliente],
    capacidade: int,
    dist: List[List[float]],
) -> bool:
    nova_carga = carga + clientes[candidato]['demand']
    if nova_carga > capacidade:
        return False

    chegada = max(tempo + dist[atual][candidato], clientes[candidato]['ready_time'])
    if chegada > clientes[candidato]['due_date']:
        return False

    retorno_deposito = chegada + clientes[candidato]['service_time'] + dist[candidato][0]
    if retorno_deposito > clientes[0]['due_date']:
        return False

    return True


def pode_iniciar_rota(candidato: int, clientes: List[Cliente], capacidade: int, dist: List[List[float]]) -> bool:
    if clientes[candidato]['demand'] > capacidade:
        return False

    chegada = max(dist[0][candidato], clientes[candidato]['ready_time'])
    if chegada > clientes[candidato]['due_date']:
        return False

    retorno_deposito = chegada + clientes[candidato]['service_time'] + dist[candidato][0]
    return retorno_deposito <= clientes[0]['due_date']

# -------------------- Heurística ACO --------------------
'''
def heuristica(cidade_atual, candidato, clientes, dist, tempo_atual):
    distancia = dist[cidade_atual][candidato]
    cliente = clientes[candidato]

    chegada = tempo_atual + distancia
    atraso = max(0.0, chegada - cliente['due_date'])

    janela = max(1.0, clientes[candidato]['due_date'] - clientes[candidato]['ready_time'])
    return 1.0 / (distancia + 1.0) * (1.0 / (janela + atraso + 1))
'''

def heuristica(cidade_atual, candidato, clientes, dist, tempo_atual):
    d = dist[cidade_atual][candidato]
    cliente = clientes[candidato]

    chegada = tempo_atual + d

    espera = max(0, cliente['ready_time'] - chegada)
    atraso = max(0, chegada - cliente['due_date'])

    urgencia = 1 / (cliente['due_date'] - tempo_atual + 1)

    return (
        1/(d + 1) *
        1/(1 + espera) *
        1/(1 + atraso) *
        urgencia
    )


def escolher_proxima_cidade(
    cidade_atual: int,
    candidatos: List[int],
    feromonios: List[List[float]],
    clientes: List[Cliente],
    dist: List[List[float]],
    tempo_atual
):
    pesos = []
    for cid in candidatos:
        tau = feromonios[cidade_atual][cid] ** ALPHA
        eta = heuristica(cidade_atual, cid, clientes, dist, tempo_atual) ** BETA
        pesos.append(tau * eta)

    total = sum(pesos)
    if total <= 0:
        return random.choice(candidatos)

    limite = random.uniform(0, total)
    acumulado = 0.0
    for cid, peso in zip(candidatos, pesos):
        acumulado += peso
        if acumulado >= limite:
            return cid
    return candidatos[-1]

# -------------------- Construção de soluções --------------------

def construir_rotas_ant(
    feromonios: List[List[float]],
    clientes: List[Cliente],
    capacidade: int,
    dist: List[List[float]],
) -> Optional[List[List[int]]]:
    nao_visitados = set(range(1, len(clientes)))
    rotas: List[List[int]] = []

    while nao_visitados:
        rota = [0]
        carga = 0.0
        tempo = 0.0

        while True:
            cidade_atual = rota[-1]
            candidatos = [cid for cid in nao_visitados if pode_inserir_cidade(cidade_atual, cid, carga, tempo, clientes, capacidade, dist)]

            if not candidatos:
                break

            proximo = escolher_proxima_cidade(cidade_atual, candidatos, feromonios, clientes, dist, tempo)
            chegada = max(tempo + dist[cidade_atual][proximo], clientes[proximo]['ready_time'])
            carga += clientes[proximo]['demand']
            tempo = chegada + clientes[proximo]['service_time']
            rota.append(proximo)
            nao_visitados.remove(proximo)

        if len(rota) == 1:
            return None

        rota.append(0)
        rotas.append(rota)

        if nao_visitados and not any(pode_iniciar_rota(cid, clientes, capacidade, dist) for cid in nao_visitados):
            return None

    return rotas

# -------------------- Feromônio --------------------

def criar_matriz_feromonios(n: int) -> List[List[float]]:
    return [[FEROMONIO_INICIAL for _ in range(n)] for _ in range(n)]


def atualizar_feromonio(
    feromonios: List[List[float]],
    rotas: List[List[int]],
    dist: List[List[float]],
) -> None:
    n = len(feromonios)
    for i in range(n):
        for j in range(n):
            feromonios[i][j] *= (1.0 - RHO)
            feromonios[i][j] = max(TAU_MIN, min(TAU_MAX, feromonios[i][j]))
            #feromonios[i][j] = max(feromonios[i][j], 1e-9)

    _, distancia_total = avaliar_rotas(rotas, dist)
    if distancia_total <= 0:
        return

    deposito = Q / distancia_total
    for rota in rotas:
        for i in range(len(rota) - 1):
            u = rota[i]
            v = rota[i + 1]
            feromonios[u][v] += deposito
            feromonios[v][u] += deposito

# -------------------- Apresentação --------------------

def formatar_rotas_txt(rotas: List[List[int]]) -> str:
    linhas = []
    for idx, rota in enumerate(rotas, start=1):
        caminho = ' '.join(str(c) for c in rota[1:-1])
        linhas.append(f'Rota {idx}: {caminho}')
    return '\n'.join(linhas)


def salvar_resultado_txt(
    nome_instancia: str,
    rotas: List[List[int]],
    nv: int,
    td: float,
    geracao_melhor: int,
    tempo_melhor: float,
) -> str:
    
    pasta_saida = "TP2/resultados/resultados_aco"
    os.makedirs(pasta_saida, exist_ok=True) 
    caminho_saida = os.path.join(pasta_saida, f'{nome_instancia}_resultado.txt')

    linhas = [
        f'Nome da instância: {nome_instancia}',
        f'Melhor encontrado na geração: {geracao_melhor}',
        f'Tempo até melhor resultado: {tempo_melhor:.2f}s',
        f'Número de veículos: {nv}',
        f'Distância total: {td:.4f}',
        'Rotas:',
        formatar_rotas_txt(rotas)
    ]

    with open(caminho_saida, 'w', encoding='utf-8') as arquivo:
        arquivo.write('\n'.join(linhas) + '\n')

    return caminho_saida

# -------------------- Execução do ACO --------------------


random.seed(SEMENTE)
nome_instancia, capacidade, clientes = ler_instancia(CAMINHO_INSTANCIA)
dist = construir_matriz_distancias(clientes)
n = len(clientes)
feromonios = criar_matriz_feromonios(n)

inicio_total = time.time()

for geracao in range(1, ITERACOES + 1):
    tempo_atual = time.time() - inicio_total
    if tempo_atual >= TEMPO_MAXIMO:
        print(f'\nTempo máximo de {TEMPO_MAXIMO:.0f}s atingido antes da geração {geracao}. Parando execução.')
        break

    if geracao - ULTIMA_MELHORA > PATIENCE:
        print("Resetando feromônio!")
        feromonios = criar_matriz_feromonios(n)
        ULTIMA_MELHORA = geracao

    melhor_rotas_geracao: Optional[List[List[int]]] = None
    melhor_pontuacao_geracao = float('inf')
    melhor_nv_geracao = float('inf')
    melhor_td_geracao = float('inf')

    for _ in range(POP_SIZE):
        if time.time() - inicio_total >= TEMPO_MAXIMO:
            break

        rotas = construir_rotas_ant(feromonios, clientes, capacidade, dist)
        if rotas is None:
            continue

        nv, td = avaliar_rotas(rotas, dist)
        pontuacao = fitness(nv, td)
        if pontuacao < melhor_pontuacao_geracao:
            melhor_pontuacao_geracao = pontuacao
            melhor_rotas_geracao = rotas
            melhor_nv_geracao = nv
            melhor_td_geracao = td

    tempo_atual = time.time() - inicio_total
    if tempo_atual >= TEMPO_MAXIMO:
        print(f'\nTempo máximo de {TEMPO_MAXIMO:.0f}s atingido durante a geração {geracao}. Parando execução.')
        break

    if melhor_rotas_geracao is None:
        print(f'Geração {geracao} | não viável | melhor global {MELHOR_NUM_ROTAS}/{MELHOR_DISTANCIA:.4f} | tempo {tempo_atual:.2f}s')
        continue

    if comparar_solucoes((melhor_nv_geracao, melhor_td_geracao), (MELHOR_NUM_ROTAS, MELHOR_DISTANCIA)):
        MELHOR_NUM_ROTAS = melhor_nv_geracao
        MELHOR_DISTANCIA = melhor_td_geracao
        MELHOR_SOLUCAO = melhor_rotas_geracao
        MELHOR_GERACAO = geracao
        MELHOR_TEMPO = tempo_atual
        ULTIMA_MELHORA = geracao


    atualizar_feromonio(feromonios, melhor_rotas_geracao, dist)

    if MELHOR_SOLUCAO:
        atualizar_feromonio(feromonios, MELHOR_SOLUCAO, dist)

    print(
        f'Geração {geracao} | Veículos geração: {melhor_nv_geracao} | Distância geração: {melhor_td_geracao:.4f} | '
        f'Melhor global: {MELHOR_NUM_ROTAS}/{MELHOR_DISTANCIA:.4f} | Tempo: {tempo_atual:.2f}s'
    )

tempo_total = time.time() - inicio_total

caminho_saida = salvar_resultado_txt(
    nome_instancia,
    MELHOR_SOLUCAO,
    MELHOR_NUM_ROTAS,
    MELHOR_DISTANCIA,
    MELHOR_GERACAO,
    MELHOR_TEMPO,
)

print('\nMelhor solução global:')
print(f'Número de veículos: {MELHOR_NUM_ROTAS}')
print(f'Distância total: {MELHOR_DISTANCIA:.4f}')
print(f'Melhor encontrado na geração: {MELHOR_GERACAO} | tempo: {MELHOR_TEMPO:.2f}s')
print(f'Resultado salvo em: {caminho_saida}')
print(f'Tempo total de execução: {tempo_total:.2f}s')
