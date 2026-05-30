# -----------------------------------------------------------------------------------
# Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
# Algoritmo de Colônia de Formigas para o Problema do Caixeiro Viajante (TSP)
# --------------------------------------------------------------------------------

import random
import matplotlib.pyplot as plt

POP_SIZE = 20
ITERACOES = 50

ALPHA = 0.5
BETA = 0.5
RHO = 0.5
VALOR_FEROMONIO = 0.000001 

MELHOR_CAMINHO = None
MENOR_DISTANCIA = float('inf')
MELHORES_CAMINHOS = []
MENORES_DISTANCIAS = []

#------------- Lendo a Matriz de Distâncias -------------

def ler_matriz_distancias(caminho_arquivo):
    matriz = []

    with open(caminho_arquivo, 'r') as f:
        for linha in f:
            linha = linha.strip()

            if not linha:
                continue

            if linha.startswith('#'):
                continue

            valores = list(map(int, linha.split()))

            matriz.append(valores)

    return matriz

MATRIZ_DISTANCIAS = ler_matriz_distancias("/home/gabriel/Desktop/Faculdade/7periodo/BioInspirados/Algoritmos-Bioinspirados-main/lesson_6/lau15_dist.txt")
N_CIDADES = len(MATRIZ_DISTANCIAS)

# ---------------- Funções do problema ----------------

def calcular_distancia_total(rota):
    distancia = 0

    for i in range(len(rota) - 1):
        cidade_atual = rota[i]
        proxima_cidade = rota[i + 1]
        distancia += MATRIZ_DISTANCIAS[cidade_atual][proxima_cidade]

    # retorna para a cidade inicial
    distancia += MATRIZ_DISTANCIAS[rota[-1]][rota[0]]

    return distancia


def rota_valida(rota):
    return len(rota) == N_CIDADES and len(set(rota)) == N_CIDADES


# ---------------- Funções do ACO ----------------

def criar_individuo():
    individuo = []
    individuo.append(random.randint(0, N_CIDADES - 1))
    return individuo


def criar_populacao():
    return [criar_individuo() for _ in range(POP_SIZE)]

def criar_matriz_feromonios():
    return [[VALOR_FEROMONIO for _ in range(N_CIDADES)] for _ in range(N_CIDADES)]

def escolher_proxima_cidade(individuo, feromonios):
    cidade_atual = individuo[-1]
    cidades = list(range(N_CIDADES))
    cidades_disponiveis = [cidade for cidade in range(N_CIDADES) if cidade not in individuo]

    if not cidades_disponiveis:
        return None

    probabilidades = [0] * N_CIDADES
    
    for cidade in cidades_disponiveis:
        somatorio = 0
        for c in cidades_disponiveis:
            feromonio = feromonios[cidade_atual][c]
            distancia = MATRIZ_DISTANCIAS[cidade_atual][c]
            inverso = 1 / distancia
            somatorio = somatorio + (feromonio**ALPHA * inverso**BETA)


        feromonio = feromonios[cidade_atual][cidade]
        distancia = MATRIZ_DISTANCIAS[cidade_atual][cidade]
        inverso = 1 / distancia
        
        probabilidades[cidade] = (feromonio**ALPHA * inverso**BETA)/somatorio


    # ---------------- Roleta -------------------
    sorteio = random.uniform(0,1)
    aux = 0
    for i in range(len(probabilidades)):
        aux = aux + probabilidades[i]
        if sorteio <= aux:
            #print(f"Sorteio: {sorteio}, Aux: {aux}, Cidade escolhida: {i}")
            individuo.append(cidades[i])
            break


def caminho_completo(individuo, feromonios, lista_individuos):
    while len(individuo) < N_CIDADES:
        escolher_proxima_cidade(individuo, feromonios)
    individuo.append(individuo[0])  # Retorna para a cidade inicial
    print(f"Indivíduo completo: {individuo}")
    print(f"Distância total: {calcular_distancia_total(individuo)}")
    lista_individuos.append(individuo)

def atualiza_feromonios(feromonios, individuos):
    for i in range(N_CIDADES):
        for j in range(N_CIDADES):
            feromonios[i][j] = (1 - RHO) * feromonios[i][j]
    
    for individuo in individuos:
        distancia = calcular_distancia_total(individuo)
        for i in range(len(individuo) - 1):
            cidade_atual = individuo[i]
            proxima_cidade = individuo[i + 1]
            feromonios[cidade_atual][proxima_cidade] += 1 / distancia
            feromonios[proxima_cidade][cidade_atual] += 1 / distancia


# ----------------- Função Principal ----------------

feromonios = criar_matriz_feromonios()
for iteracao in range(ITERACOES):
    individuos = criar_populacao()
    lista_individuos = []
    print(f"Iteração {iteracao + 1}")
    for individuo in individuos:
        caminho_completo(individuo, feromonios, lista_individuos)

    atualiza_feromonios(feromonios, lista_individuos)

    for individuo in lista_individuos:
        distancia = calcular_distancia_total(individuo)
        if distancia < MENOR_DISTANCIA:
            MENOR_DISTANCIA = distancia
            MELHOR_CAMINHO = individuo

    MELHORES_CAMINHOS.append(MELHOR_CAMINHO)
    MENORES_DISTANCIAS.append(MENOR_DISTANCIA)

print("Melhor caminho encontrado:", MELHOR_CAMINHO)
print("Melhor distância encontrada:", MENOR_DISTANCIA)
print("Menores distâncias ao longo das iterações:", MENORES_DISTANCIAS)

#------------------ Gráfico da evolução da melhor distância ----------------

plt.figure(figsize=(14, 6))
plt.plot(range(1, ITERACOES + 1), MENORES_DISTANCIAS, linestyle='-')
plt.axhline(y=291, linestyle='--', label='Ótimo (291)')

plt.title('Evolução da melhor distância')

plt.xlabel('Iteração')
plt.ylabel('Melhor distância')
plt.grid(True)

# Texto com informações
texto_info = (
    f"Solução ótima esperada: 291\n"
    f"Menor distancia: {MENOR_DISTANCIA}\n"
    f"População: {POP_SIZE}\n"
    f"Iterações: {ITERACOES}\n"
    f"Alpha: {ALPHA}\n"
    f"Beta: {BETA}\n"
    f"Rho: {RHO}"
)

plt.text(
    0.8, 0.95, texto_info,
    transform=plt.gca().transAxes,  # usa coordenadas relativas (0 a 1)
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='square', facecolor='white', alpha=0.8)
)

plt.savefig("lesson_6/grafico_distancia_lau15.png", dpi=300)
plt.show()
