# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# Algoritmo Genetico para o VRPTW (Vehicle Routing Problem with Time Windows)
# -----------------------------------------------------------------------------------

import copy
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------- Parametros ----------------
INSTANCE_FILE = sys.argv[1] if len(sys.argv) > 1 else "rc208.txt"
AUTHORS = "Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini"

POP_SIZE = 260
GENERATIONS = 20000
CROSSOVER_RATE = 0.88
MUTATION_RATE = 0.28
ELITISM = 8
TOURNAMENT_SIZE = 4
TIME_LIMIT = 460.0
VERBOSE = True
PENALIDADE_INVIAVEL = 1_000_000_000_000.0
PESO_VEICULOS = 1_000_000.0

# Para reproduzir resultados:
RANDOM_SEED = 42

# ---------------- Estruturas da instancia ----------------

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


# ---------------- Dados da instancia ----------------

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
        if not tokens:
            continue

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

# Configuracao geral para a coletanea: reduz o custo por geracao nas instancias
# grandes e preserva diversidade suficiente nas instancias de 100 clientes.
POP_SIZE = min(320, max(160, int(0.75 * N_CLIENTES)))
ELITISM = max(6, int(0.03 * POP_SIZE))

# ---------------- Funcoes do problema ----------------

def distancia_euclidiana(cliente_a, cliente_b):
    return math.sqrt((cliente_a.x - cliente_b.x) ** 2 + (cliente_a.y - cliente_b.y) ** 2)


def distancia_rota(rota):
    if not rota:
        return 0.0

    deposito = INSTANCIA.deposito
    clientes = INSTANCIA.clientes

    distancia = distancia_euclidiana(deposito, clientes[rota[0]])
    for i in range(len(rota) - 1):
        distancia += distancia_euclidiana(clientes[rota[i]], clientes[rota[i + 1]])
    distancia += distancia_euclidiana(clientes[rota[-1]], deposito)

    return distancia


def rota_viavel(rota):
    clientes = INSTANCIA.clientes
    deposito = INSTANCIA.deposito
    carga = sum(clientes[cliente].demanda for cliente in rota)

    if carga > INSTANCIA.capacidade:
        return False

    tempo_atual = 0.0
    cliente_atual = deposito

    for cliente_id in rota:
        cliente = clientes[cliente_id]
        tempo_atual += distancia_euclidiana(cliente_atual, cliente)
        tempo_atual = max(tempo_atual, cliente.ready_time)

        if tempo_atual > cliente.due_date:
            return False

        tempo_atual += cliente.service_time
        cliente_atual = cliente

    tempo_atual += distancia_euclidiana(cliente_atual, deposito)
    return tempo_atual <= deposito.due_date


def decodificar_cromossomo(individuo):
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
            tempo_apos_atendimento = tempo_inicio + cliente.service_time
            retorno_deposito = distancia_euclidiana(cliente, deposito)
            if tempo_apos_atendimento + retorno_deposito > deposito.due_date:
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


def distancia_total(individuo):
    return sum(distancia_rota(rota) for rota in decodificar_cromossomo(individuo))


def numero_veiculos(individuo):
    return len(decodificar_cromossomo(individuo))


def solucao_viavel(individuo):
    return all(rota_viavel(rota) for rota in decodificar_cromossomo(individuo))


def fitness(individuo):
    rotas = decodificar_cromossomo(individuo)
    distancia = sum(distancia_rota(rota) for rota in rotas)
    penalidade = 0.0

    if not all(rota_viavel(rota) for rota in rotas):
        penalidade = PENALIDADE_INVIAVEL

    return penalidade + len(rotas) * PESO_VEICULOS + distancia


# ---------------- Funcoes do AG ----------------

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


def torneio(populacao):
    candidatos = random.sample(populacao, TOURNAMENT_SIZE)
    return copy.copy(min(candidatos, key=fitness))


def crossover_ox(pai1, pai2):
    n = len(pai1)
    p1, p2 = sorted(random.sample(range(n), 2))

    def criar_filho(base, complemento):
        segmento = base[p1:p2 + 1]
        restantes = [gene for gene in complemento if gene not in segmento]
        return restantes[:p1] + segmento + restantes[p1:]

    filho1 = criar_filho(pai1, pai2)
    filho2 = criar_filho(pai2, pai1)

    return filho1, filho2


def mutacao_swap(individuo):
    filho = individuo[:]
    i, j = random.sample(range(len(filho)), 2)
    filho[i], filho[j] = filho[j], filho[i]
    return filho


def mutacao_inversao(individuo):
    filho = individuo[:]
    i, j = sorted(random.sample(range(len(filho)), 2))
    filho[i:j + 1] = reversed(filho[i:j + 1])
    return filho


def mutacao_oropt(individuo):
    filho = individuo[:]
    i = random.randrange(len(filho))
    gene = filho.pop(i)
    j = random.randrange(len(filho) + 1)
    filho.insert(j, gene)
    return filho


def mutacao(individuo):
    filho = individuo[:]

    if random.random() < MUTATION_RATE:
        escolha = random.random()
        if escolha < 0.4:
            filho = mutacao_swap(filho)
        elif escolha < 0.7:
            filho = mutacao_inversao(filho)
        else:
            filho = mutacao_oropt(filho)

    return filho


def nova_geracao(populacao):
    populacao_ordenada = sorted(populacao, key=fitness)
    nova_pop = [ind[:] for ind in populacao_ordenada[:ELITISM]]

    while len(nova_pop) < POP_SIZE:
        pai1 = torneio(populacao)
        pai2 = torneio(populacao)

        if random.random() < CROSSOVER_RATE:
            filho1, filho2 = crossover_ox(pai1, pai2)
        else:
            filho1, filho2 = pai1[:], pai2[:]

        filho1 = mutacao(filho1)
        filho2 = mutacao(filho2)

        nova_pop.append(filho1)
        if len(nova_pop) < POP_SIZE:
            nova_pop.append(filho2)

    return nova_pop


# ---------------- Saida ----------------

def formatar_saida(rotas, distancia, tempo_execucao, tempo_melhor):
    linhas = [
        "======== MELHOR SOLUCAO AG ========",
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
    nome_arquivo = f"{INSTANCIA.nome}_resultado_ag.txt"

    with open(nome_arquivo, "w", encoding="utf-8") as arquivo:
        arquivo.write(conteudo)

    return nome_arquivo, conteudo


# ---------------- Execucao ----------------

random.seed(RANDOM_SEED)

print(f"\nLendo instancia: {INSTANCE_FILE}")
print(
    f"Instancia: {INSTANCIA.nome} | Clientes: {INSTANCIA.n_clientes} | "
    f"Capacidade: {INSTANCIA.capacidade} | Veiculos max: {INSTANCIA.max_veiculos}"
)

populacao = criar_populacao()
inicio = time.time()
melhor_global = min(populacao, key=fitness)
melhor_fitness_global = fitness(melhor_global)
tempo_melhor = 0.0

melhores_fitness = []
melhores_distancias = []
melhores_veiculos = []

for geracao in range(GENERATIONS):
    tempo_decorrido = time.time() - inicio
    if tempo_decorrido >= TIME_LIMIT:
        print(f"\nTempo limite atingido na geracao {geracao}.")
        break

    melhor = min(populacao, key=fitness)

    best_fit = fitness(melhor)
    if best_fit < melhor_fitness_global:
        melhor_global = melhor[:]
        melhor_fitness_global = best_fit
        tempo_melhor = tempo_decorrido

    best_distancia = distancia_total(melhor)
    best_veiculos = numero_veiculos(melhor)

    melhores_fitness.append(best_fit)
    melhores_distancias.append(best_distancia)
    melhores_veiculos.append(best_veiculos)

    if VERBOSE:
        print(f"Geracao {geracao}: tempo = {tempo_decorrido:.2f}s")

    populacao = nova_geracao(populacao)

# ---------------- Resultado final ----------------

candidato_final = min(populacao, key=fitness)
fitness_candidato_final = fitness(candidato_final)
if fitness_candidato_final < melhor_fitness_global:
    melhor_global = candidato_final[:]
    melhor_fitness_global = fitness_candidato_final
    tempo_melhor = time.time() - inicio

melhor_final = melhor_global

rotas_final = decodificar_cromossomo(melhor_final)
distancia_final = distancia_total(melhor_final)
fitness_final = fitness(melhor_final)
tempo_final = time.time() - inicio

print("\n--- Melhor solucao encontrada ---")
print("Numero de veiculos:", len(rotas_final))
print("Distancia total:", f"{distancia_final:.4f}")
print("Fitness:", f"{fitness_final:.2f}")
print("Solucao viavel:", solucao_viavel(melhor_final))
print("Tempo de execucao:", f"{int(tempo_final)}s")
print("Tempo da melhor solucao:", f"{int(tempo_melhor)}s")

print("\n--- Rotas ---")
for i, rota in enumerate(rotas_final, start=1):
    print(f"Rota {i}: {' -> '.join(['0'] + [str(cliente) for cliente in rota] + ['0'])}")

nome_saida, conteudo_saida = salvar_resultado(rotas_final, distancia_final, tempo_final, tempo_melhor)

print("\nResultado salvo com sucesso:")
print(f" - {nome_saida}")
