# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# Algoritmo Imunologico Clonal (CLONALG) para o VRPTW
# Vehicle Routing Problem with Time Windows
# -----------------------------------------------------------------------------------

import copy
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ------------ Parametros -------------------
INSTANCE_FILE = sys.argv[1] if len(sys.argv) > 1 else "rc208.txt"
AUTHORS = "Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini"

POP_SIZE = 120
ITERACOES = 20000

BETA_CLONAGEM = 1.25
RHO_MUTACAO = 1.35

D_MEMORIA = int(round(POP_SIZE * 0.35))
NUM_SELECIONADOS = int(round(POP_SIZE * 0.65))
TIME_LIMIT = 460.0
PENALIDADE_INVIAVEL = 1_000_000_000_000.0
PESO_VEICULOS = 1_000_000.0

RANDOM_SEED = 42
VERBOSE = True

MELHOR_SOLUCAO = None
MENOR_FITNESS = float("inf")
MENOR_DISTANCIA = float("inf")
MENOR_NUM_VEICULOS = float("inf")
MELHORES_FITNESSES = []
MENORES_DISTANCIAS = []
MENORES_VEICULOS = []
TEMPO_MELHOR = 0.0


# ------------- Estruturas da instancia -------------

@dataclass
class Cliente:
    idx: int
    x: float
    y: float
    demanda: float
    ready_time: float
    due_date: float
    service_time: float


@dataclass
class Instancia:
    nome: str
    max_veiculos: int
    capacidade: float
    clientes: List[Cliente] = field(default_factory=list)

    @property
    def deposito(self):
        return self.clientes[0]

    @property
    def n_clientes(self):
        return len(self.clientes) - 1


# ------------- Lendo a instancia Solomon VRPTW -------------

def resolver_caminho_instancia(nome_arquivo):
    candidatos = [
        nome_arquivo,
        os.path.join(os.path.dirname(__file__), nome_arquivo),
        os.path.join(os.path.dirname(__file__), "..", "Instancias_teste", nome_arquivo),
    ]

    for caminho in candidatos:
        if os.path.exists(caminho):
            return caminho

    return nome_arquivo


def ler_instancia(caminho_arquivo):
    caminho_arquivo = resolver_caminho_instancia(caminho_arquivo)
    nome = os.path.splitext(os.path.basename(caminho_arquivo))[0]

    with open(caminho_arquivo, "r", encoding="utf-8") as arquivo:
        linhas = arquivo.readlines()

    max_veiculos = 0
    capacidade = 0.0
    clientes = []
    modo = None

    for linha in linhas:
        linha = linha.strip()
        if not linha:
            continue

        upper = linha.upper()

        if "VEHICLE" in upper or ("NUMBER" in upper and "CAPACITY" in upper):
            modo = "VEHICLE"
            continue

        if "CUSTOMER" in upper or ("CUST" in upper and "NO" in upper):
            modo = "CUSTOMER"
            continue

        tokens = linha.split()

        try:
            valores = [float(token) for token in tokens]
        except ValueError:
            continue

        if modo == "VEHICLE" and len(valores) >= 2:
            max_veiculos = int(valores[0])
            capacidade = valores[1]
            modo = None
            continue

        if modo == "CUSTOMER" and len(valores) >= 7:
            clientes.append(
                Cliente(
                    idx=int(valores[0]),
                    x=valores[1],
                    y=valores[2],
                    demanda=valores[3],
                    ready_time=valores[4],
                    due_date=valores[5],
                    service_time=valores[6],
                )
            )
            continue

        if len(valores) == 2 and max_veiculos == 0 and capacidade == 0.0:
            max_veiculos = int(valores[0])
            capacidade = valores[1]
            continue

        if len(valores) >= 7:
            clientes.append(
                Cliente(
                    idx=int(valores[0]),
                    x=valores[1],
                    y=valores[2],
                    demanda=valores[3],
                    ready_time=valores[4],
                    due_date=valores[5],
                    service_time=valores[6],
                )
            )

    if not clientes:
        raise ValueError(f"Nenhum cliente encontrado em '{caminho_arquivo}'.")

    if capacidade == 0.0:
        raise ValueError("Capacidade do veiculo nao encontrada no arquivo.")

    return Instancia(
        nome=nome,
        max_veiculos=max_veiculos if max_veiculos > 0 else len(clientes),
        capacidade=capacidade,
        clientes=clientes,
    )


INSTANCIA = ler_instancia(INSTANCE_FILE)
N_CLIENTES = INSTANCIA.n_clientes

# Configuracao geral para a coletanea: mais exploracao que a versao original,
# mas com memoria mais seletiva para nao destruir as melhores rotas.
POP_SIZE = min(180, max(90, int(0.45 * N_CLIENTES)))
D_MEMORIA = int(round(POP_SIZE * 0.35))
NUM_SELECIONADOS = int(round(POP_SIZE * 0.65))


# ---------------- Funcoes do problema ----------------

def distancia_euclidiana(cliente_a, cliente_b):
    return math.sqrt((cliente_a.x - cliente_b.x) ** 2 + (cliente_a.y - cliente_b.y) ** 2)


def calcular_distancia_rota(rota):
    if not rota:
        return 0.0

    clientes = INSTANCIA.clientes
    deposito = INSTANCIA.deposito

    distancia = distancia_euclidiana(deposito, clientes[rota[0]])

    for i in range(len(rota) - 1):
        distancia += distancia_euclidiana(clientes[rota[i]], clientes[rota[i + 1]])

    distancia += distancia_euclidiana(clientes[rota[-1]], deposito)

    return distancia


def rota_viavel(rota):
    clientes = INSTANCIA.clientes
    deposito = INSTANCIA.deposito

    carga = sum(clientes[cliente_id].demanda for cliente_id in rota)
    if carga > INSTANCIA.capacidade:
        return False

    tempo_atual = 0.0
    cliente_atual = deposito

    for cliente_id in rota:
        cliente = clientes[cliente_id]
        tempo_atual += distancia_euclidiana(cliente_atual, cliente)

        if tempo_atual < cliente.ready_time:
            tempo_atual = cliente.ready_time

        if tempo_atual > cliente.due_date:
            return False

        tempo_atual += cliente.service_time
        cliente_atual = cliente

    tempo_atual += distancia_euclidiana(cliente_atual, deposito)

    return tempo_atual <= deposito.due_date


def decodificar_solucao(individuo):
    clientes = INSTANCIA.clientes
    deposito = INSTANCIA.deposito

    rotas = []
    rota_atual = []
    carga_atual = 0.0
    tempo_atual = 0.0
    cliente_atual = deposito

    for cliente_id in individuo:
        cliente = clientes[cliente_id]

        tempo_chegada = tempo_atual + distancia_euclidiana(cliente_atual, cliente)
        tempo_inicio = max(tempo_chegada, cliente.ready_time)

        pode_adicionar = (
            carga_atual + cliente.demanda <= INSTANCIA.capacidade
            and tempo_inicio <= cliente.due_date
        )

        if pode_adicionar:
            tempo_apos = tempo_inicio + cliente.service_time
            retorno_deposito = distancia_euclidiana(cliente, deposito)

            if tempo_apos + retorno_deposito > deposito.due_date:
                pode_adicionar = False

        if pode_adicionar:
            rota_atual.append(cliente_id)
            carga_atual += cliente.demanda
            tempo_atual = tempo_inicio + cliente.service_time
            cliente_atual = cliente
        else:
            if rota_atual:
                rotas.append(rota_atual)

            rota_atual = [cliente_id]
            carga_atual = cliente.demanda
            tempo_chegada = distancia_euclidiana(deposito, cliente)
            tempo_atual = max(tempo_chegada, cliente.ready_time) + cliente.service_time
            cliente_atual = cliente

    if rota_atual:
        rotas.append(rota_atual)

    return rotas


def calcular_distancia_total(individuo):
    return sum(calcular_distancia_rota(rota) for rota in decodificar_solucao(individuo))


def calcular_num_veiculos(individuo):
    return len(decodificar_solucao(individuo))


def solucao_viavel(individuo):
    return all(rota_viavel(rota) for rota in decodificar_solucao(individuo))


def calcular_fitness(individuo):
    rotas = decodificar_solucao(individuo)
    distancia = sum(calcular_distancia_rota(rota) for rota in rotas)
    penalidade = 0.0

    if not all(rota_viavel(rota) for rota in rotas):
        penalidade = PENALIDADE_INVIAVEL

    return penalidade + len(rotas) * PESO_VEICULOS + distancia


# ----------------- Funcoes do Clonalg ----------------

def criar_individuo():
    individuo = list(range(1, N_CLIENTES + 1))
    random.shuffle(individuo)
    return individuo


def criar_individuo_heuristico():
    clientes = INSTANCIA.clientes
    individuo = list(range(1, N_CLIENTES + 1))

    def centro_janela(cliente_id):
        cliente = clientes[cliente_id]
        return (cliente.ready_time + cliente.due_date) / 2

    individuo.sort(key=centro_janela)

    for _ in range(len(individuo) // 5):
        i, j = random.sample(range(len(individuo)), 2)
        individuo[i], individuo[j] = individuo[j], individuo[i]

    return individuo


def criar_populacao():
    populacao = []
    n_heuristicos = int(POP_SIZE * 0.3)

    for _ in range(n_heuristicos):
        populacao.append(criar_individuo_heuristico())

    for _ in range(POP_SIZE - n_heuristicos):
        populacao.append(criar_individuo())

    return populacao


def normalizar(populacao):
    vetor_fitness = [calcular_fitness(individuo) for individuo in populacao]

    min_val = min(vetor_fitness)
    max_val = max(vetor_fitness)
    denominador = max_val - min_val

    if denominador == 0:
        return [1.0 for _ in vetor_fitness]

    normalizado = [(valor - min_val) / denominador for valor in vetor_fitness]
    afinidade = [1.0 - valor for valor in normalizado]

    return afinidade


def selecao(populacao):
    pop_selecionada = sorted(populacao, key=calcular_fitness)
    return pop_selecionada[:NUM_SELECIONADOS]


def clonagem(populacao):
    clones = []

    for indice, individuo in enumerate(populacao):
        quantidade = math.ceil(BETA_CLONAGEM * (POP_SIZE / (indice + 1)))

        for _ in range(quantidade):
            clones.append(copy.deepcopy(individuo))

    random.shuffle(clones)
    return clones[:POP_SIZE]


def mutacao_swap(individuo):
    filho = individuo[:]
    i, j = random.sample(range(N_CLIENTES), 2)
    filho[i], filho[j] = filho[j], filho[i]
    return filho


def mutacao_inversao(individuo):
    filho = individuo[:]
    i, j = sorted(random.sample(range(N_CLIENTES), 2))
    filho[i:j + 1] = reversed(filho[i:j + 1])
    return filho


def mutacao_oropt(individuo):
    filho = individuo[:]
    i = random.randrange(N_CLIENTES)
    gene = filho.pop(i)
    j = random.randrange(N_CLIENTES)
    filho.insert(j, gene)
    return filho


def mutacao(individuo, taxa_mutacao):
    filho = individuo[:]

    if random.random() < taxa_mutacao:
        escolha = random.random()

        if escolha < 0.4:
            filho = mutacao_swap(filho)
        elif escolha < 0.7:
            filho = mutacao_inversao(filho)
        else:
            filho = mutacao_oropt(filho)

    return filho


def mutar_pop(populacao):
    afinidades = normalizar(populacao)
    pop_mutada = []

    for i, individuo in enumerate(populacao):
        taxa_mutacao = math.exp(-RHO_MUTACAO * afinidades[i])
        individuo_mutado = mutacao(individuo, taxa_mutacao)
        pop_mutada.append(individuo_mutado)

    return pop_mutada


def memoria(populacao_mutada, populacao_anterior, melhor_global):
    combinada = populacao_mutada + populacao_anterior
    combinada = sorted(combinada, key=calcular_fitness)
    combinada = combinada[:D_MEMORIA]

    while len(combinada) < POP_SIZE:
        combinada.append(criar_individuo())

    combinada[0] = copy.deepcopy(melhor_global)

    return combinada[:POP_SIZE]


# ---------------- Saida ----------------

def formatar_saida(rotas, distancia, tempo_execucao, tempo_melhor):
    linhas = [
        "======== MELHOR SOLUCAO CLONALG ========",
        f"Nome da instancia: {INSTANCIA.nome}",
        f"Autores: {AUTHORS}",
        f"Numero de veiculos: {len(rotas)}",
        f"Distancia total: {distancia:.4f}",
        f"Tempo total: {int(tempo_execucao)}s",
        f"Tempo da melhor solucao: {int(tempo_melhor)}s",
        "Rotas:",
    ]

    for i, rota in enumerate(rotas, start=1):
        caminho = " -> ".join(["0"] + [str(cliente) for cliente in rota] + ["0"])
        linhas.append(f"Rota {i}: {caminho}")

    linhas.append("")
    return "\n".join(linhas)


def salvar_resultado(rotas, distancia, tempo_execucao, tempo_melhor):
    conteudo = formatar_saida(rotas, distancia, tempo_execucao, tempo_melhor)
    nome_arquivo = f"{INSTANCIA.nome}_resultado_clonalg.txt"

    with open(nome_arquivo, "w", encoding="utf-8") as arquivo:
        arquivo.write(conteudo)

    return nome_arquivo, conteudo


# ---------------- Execucao -------------------

random.seed(RANDOM_SEED)

pop = criar_populacao()
pop_anterior = copy.deepcopy(pop)
inicio = time.time()

print(f"\nInstancia: {INSTANCIA.nome}")
print(f"Clientes: {N_CLIENTES} | Capacidade: {INSTANCIA.capacidade}")
print(f"Configuracao: POP={POP_SIZE}, ITER={ITERACOES}, BETA={BETA_CLONAGEM}, RHO={RHO_MUTACAO}")
print(f"Selecionados={NUM_SELECIONADOS}, Memoria={D_MEMORIA}\n")

melhor_global = min(pop, key=calcular_fitness)
MENOR_FITNESS = calcular_fitness(melhor_global)
MENOR_DISTANCIA = calcular_distancia_total(melhor_global)
MENOR_NUM_VEICULOS = calcular_num_veiculos(melhor_global)
MELHOR_SOLUCAO = melhor_global[:]

for iteracao in range(ITERACOES):
    tempo_decorrido = time.time() - inicio
    if tempo_decorrido >= TIME_LIMIT:
        print(f"\nTempo limite atingido na iteracao {iteracao + 1}.")
        break

    selecionados = selecao(pop)

    clones = clonagem(selecionados)

    pop_mutada = mutar_pop(clones)

    candidato = min(pop_mutada, key=calcular_fitness)
    fitness_candidato = calcular_fitness(candidato)

    if fitness_candidato < MENOR_FITNESS:
        MENOR_FITNESS = fitness_candidato
        MENOR_DISTANCIA = calcular_distancia_total(candidato)
        MENOR_NUM_VEICULOS = calcular_num_veiculos(candidato)
        MELHOR_SOLUCAO = candidato[:]
        TEMPO_MELHOR = tempo_decorrido
        melhor_global = candidato[:]

    pop_anterior = copy.deepcopy(pop)
    pop = memoria(pop_mutada, pop_anterior, melhor_global)

    MELHORES_FITNESSES.append(MENOR_FITNESS)
    MENORES_DISTANCIAS.append(MENOR_DISTANCIA)
    MENORES_VEICULOS.append(MENOR_NUM_VEICULOS)

    if VERBOSE:
        print(f"Iteracao {iteracao + 1}: tempo = {tempo_decorrido:.2f}s")


rotas_finais = decodificar_solucao(MELHOR_SOLUCAO)
tempo_total = time.time() - inicio

print("\nResultados finais:")
print("Numero de veiculos:", len(rotas_finais))
print("Distancia total:", f"{MENOR_DISTANCIA:.4f}")
print("Fitness:", f"{MENOR_FITNESS:.2f}")
print("Solucao viavel:", solucao_viavel(MELHOR_SOLUCAO))
print("Tempo de execucao:", f"{int(tempo_total)}s")
print("Tempo da melhor solucao:", f"{int(TEMPO_MELHOR)}s")

print("\nRotas:")
for i, rota in enumerate(rotas_finais, start=1):
    caminho = " -> ".join(["0"] + [str(cliente) for cliente in rota] + ["0"])
    print(f"Rota {i}: {caminho}")

nome_saida, conteudo_saida = salvar_resultado(rotas_finais, MENOR_DISTANCIA, tempo_total, TEMPO_MELHOR)

print("\nArquivos salvos:")
print(f" - {nome_saida}")
