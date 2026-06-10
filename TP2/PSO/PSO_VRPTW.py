# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# Particle Swarm Optimization (PSO) discreto para o VRPTW
# Vehicle Routing Problem with Time Windows
# -----------------------------------------------------------------------------------

import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ---------------- Parametros ----------------
INSTANCE_FILE = sys.argv[1] if len(sys.argv) > 1 else "rc208.txt"
AUTHORS = "Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini"

SWARM_SIZE = 140
ITERACOES = 20000

W = 0.55
C1 = 1.10
C2 = 1.40
MUTATION_RATE = 0.20
MAX_VELOCITY = 45
TIME_LIMIT = 460.0
PENALIDADE_INVIAVEL = 1_000_000_000_000.0
PESO_VEICULOS = 1_000_000.0

RANDOM_SEED = 42
VERBOSE = True


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


@dataclass
class Particula:
    posicao: List[int]
    velocidade: List[Tuple[int, int]]
    pbest: List[int]
    pbest_fitness: float


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

# Configuracao geral para a coletanea: enxame e numero maximo de trocas
# crescem com a instancia, evitando movimentos curtos demais nos casos grandes.
SWARM_SIZE = min(220, max(100, int(0.45 * N_CLIENTES)))
MAX_VELOCITY = min(80, max(25, int(0.18 * N_CLIENTES)))


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


def decodificar_posicao(posicao):
    clientes = INSTANCIA.clientes
    deposito = INSTANCIA.deposito

    rotas = []
    rota_atual = []
    carga_atual = 0.0
    tempo_atual = 0.0
    cliente_atual = deposito

    for cliente_id in posicao:
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


def distancia_total(posicao):
    return sum(distancia_rota(rota) for rota in decodificar_posicao(posicao))


def numero_veiculos(posicao):
    return len(decodificar_posicao(posicao))


def solucao_viavel(posicao):
    return all(rota_viavel(rota) for rota in decodificar_posicao(posicao))


def fitness(posicao):
    rotas = decodificar_posicao(posicao)
    distancia = sum(distancia_rota(rota) for rota in rotas)
    penalidade = 0.0

    if not all(rota_viavel(rota) for rota in rotas):
        penalidade = PENALIDADE_INVIAVEL

    return penalidade + len(rotas) * PESO_VEICULOS + distancia


def eh_melhor(posicao_a, posicao_b):
    return fitness(posicao_a) < fitness(posicao_b)


# ---------------- Funcoes do PSO discreto ----------------

def criar_posicao_aleatoria():
    posicao = list(range(1, N_CLIENTES + 1))
    random.shuffle(posicao)
    return posicao


def criar_posicao_heuristica():
    clientes = INSTANCIA.clientes
    posicao = list(range(1, N_CLIENTES + 1))

    def centro_janela(cliente_id):
        cliente = clientes[cliente_id]
        return (cliente.ready_time + cliente.due_date) / 2

    posicao.sort(key=centro_janela)

    for _ in range(len(posicao) // 5):
        i, j = random.sample(range(len(posicao)), 2)
        posicao[i], posicao[j] = posicao[j], posicao[i]

    return posicao


def gerar_velocidade_inicial():
    velocidade = []
    quantidade = random.randint(1, min(MAX_VELOCITY, max(1, N_CLIENTES // 3)))

    for _ in range(quantidade):
        i, j = random.sample(range(N_CLIENTES), 2)
        velocidade.append((i, j))

    return velocidade


def criar_particula():
    if random.random() < 0.3:
        posicao = criar_posicao_heuristica()
    else:
        posicao = criar_posicao_aleatoria()

    return Particula(
        posicao=posicao[:],
        velocidade=gerar_velocidade_inicial(),
        pbest=posicao[:],
        pbest_fitness=fitness(posicao),
    )


def criar_enxame():
    return [criar_particula() for _ in range(SWARM_SIZE)]


def diferenca_por_trocas(origem, destino):
    atual = origem[:]
    indice_por_cliente = {cliente: i for i, cliente in enumerate(atual)}
    trocas = []

    for i, cliente_destino in enumerate(destino):
        if atual[i] == cliente_destino:
            continue

        j = indice_por_cliente[cliente_destino]
        cliente_i = atual[i]
        atual[i], atual[j] = atual[j], atual[i]
        indice_por_cliente[cliente_i] = j
        indice_por_cliente[cliente_destino] = i
        trocas.append((i, j))

    return trocas


def amostrar_trocas(trocas, coeficiente):
    if not trocas:
        return []

    quantidade = int(round(coeficiente * random.random() * len(trocas)))
    quantidade = max(0, min(len(trocas), quantidade))

    if quantidade == 0:
        return []

    return trocas[:quantidade]


def limitar_velocidade(velocidade):
    if len(velocidade) <= MAX_VELOCITY:
        return velocidade

    return velocidade[-MAX_VELOCITY:]


def aplicar_velocidade(posicao, velocidade):
    nova_posicao = posicao[:]

    for i, j in velocidade:
        nova_posicao[i], nova_posicao[j] = nova_posicao[j], nova_posicao[i]

    return nova_posicao


def mutacao_local(posicao):
    nova_posicao = posicao[:]
    escolha = random.random()

    if escolha < 0.4:
        i, j = random.sample(range(N_CLIENTES), 2)
        nova_posicao[i], nova_posicao[j] = nova_posicao[j], nova_posicao[i]
    elif escolha < 0.7:
        i, j = sorted(random.sample(range(N_CLIENTES), 2))
        nova_posicao[i:j + 1] = reversed(nova_posicao[i:j + 1])
    else:
        i = random.randrange(N_CLIENTES)
        cliente = nova_posicao.pop(i)
        j = random.randrange(N_CLIENTES)
        nova_posicao.insert(j, cliente)

    return nova_posicao


def atualizar_particula(particula, gbest):
    velocidade_inercial = amostrar_trocas(particula.velocidade, W)
    velocidade_cognitiva = amostrar_trocas(diferenca_por_trocas(particula.posicao, particula.pbest), C1)
    velocidade_social = amostrar_trocas(diferenca_por_trocas(particula.posicao, gbest), C2)

    particula.velocidade = limitar_velocidade(
        velocidade_inercial + velocidade_cognitiva + velocidade_social
    )

    nova_posicao = aplicar_velocidade(particula.posicao, particula.velocidade)

    if random.random() < MUTATION_RATE:
        nova_posicao = mutacao_local(nova_posicao)

    if eh_melhor(nova_posicao, particula.posicao) or random.random() < 0.05:
        particula.posicao = nova_posicao

    fitness_atual = fitness(particula.posicao)
    if fitness_atual < particula.pbest_fitness:
        particula.pbest = particula.posicao[:]
        particula.pbest_fitness = fitness_atual


# ---------------- Saida ----------------

def formatar_saida(rotas, distancia, tempo_execucao, tempo_melhor):
    linhas = [
        "======== MELHOR SOLUCAO PSO ========",
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
    nome_arquivo = f"{INSTANCIA.nome}_resultado_pso.txt"

    with open(nome_arquivo, "w", encoding="utf-8") as arquivo:
        arquivo.write(conteudo)

    return nome_arquivo, conteudo


# ---------------- Execucao ----------------

def executar_pso():
    random.seed(RANDOM_SEED)

    enxame = criar_enxame()
    inicio = time.time()

    gbest = min((particula.pbest for particula in enxame), key=fitness)
    gbest = gbest[:]
    gbest_fitness = fitness(gbest)
    tempo_melhor = 0.0

    melhores_fitness = []
    melhores_distancias = []
    melhores_veiculos = []

    print(f"\nLendo instancia: {INSTANCE_FILE}")
    print(
        f"Instancia: {INSTANCIA.nome} | Clientes: {INSTANCIA.n_clientes} | "
        f"Capacidade: {INSTANCIA.capacidade} | Veiculos max: {INSTANCIA.max_veiculos}"
    )
    print(
        f"Configuracao: SWARM={SWARM_SIZE}, ITER={ITERACOES}, "
        f"W={W}, C1={C1}, C2={C2}, VMAX={MAX_VELOCITY}"
    )

    for iteracao in range(ITERACOES):
        tempo_decorrido = time.time() - inicio
        if tempo_decorrido >= TIME_LIMIT:
            print(f"\nTempo limite atingido na iteracao {iteracao}.")
            break

        for particula in enxame:
            atualizar_particula(particula, gbest)

            if particula.pbest_fitness < gbest_fitness:
                gbest = particula.pbest[:]
                gbest_fitness = particula.pbest_fitness
                tempo_melhor = time.time() - inicio

        best_distancia = distancia_total(gbest)
        best_veiculos = numero_veiculos(gbest)

        melhores_fitness.append(gbest_fitness)
        melhores_distancias.append(best_distancia)
        melhores_veiculos.append(best_veiculos)

        if VERBOSE:
            print(f"Iteracao {iteracao}: tempo = {tempo_decorrido:.2f}s")

    return gbest, melhores_fitness, melhores_distancias, melhores_veiculos, time.time() - inicio, tempo_melhor


if __name__ == "__main__":
    melhor_final, melhores_fitness, melhores_distancias, melhores_veiculos, tempo_final, tempo_melhor = executar_pso()

    rotas_final = decodificar_posicao(melhor_final)
    distancia_final = distancia_total(melhor_final)
    fitness_final = fitness(melhor_final)

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

    print("\nArquivos salvos:")
    print(f" - {nome_saida}")
