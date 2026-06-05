"""
-----------------------------------------------------------------------------------
Authors: Felipe Girardi Siqueira, Lucas Daniel Lana Maciel, Gabriel Vaz Bernardini
Algoritmo Evolução Diferencial
-----------------------------------------------------------------------------------
"""

import os
import random
import sys
import time

import numpy as np

# parâmetros globais de configuração
SEED = 42
ELITE_SIZE = 12
LS_FRAC = 0.50
N_REMOVE_PERTURB = 20
PRESERVE_RATE = 0.45
SHUFFLE_FACTOR_MIN = 0.0
SHUFFLE_FACTOR_MAX = 1.5
SEG_SIZES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

# ─────────────────────────────────────────────────────────────────
# 1. LEITURA DE INSTÂNCIA
# ─────────────────────────────────────────────────────────────────

def ler_instancia(caminho):
    """Lê a instância VRPTW no formato esperado e retorna os dados."""
    with open(caminho, 'r', encoding='utf-8') as arquivo:
        linhas = [linha.strip() for linha in arquivo if linha.strip()]

    nome_instancia = linhas[0]
    capacidade = None
    for i, linha in enumerate(linhas):
        if linha.upper().startswith('NUMBER'):
            capacidade = int(linhas[i + 1].split()[1])
            break

    clientes = []
    lendo_clientes = False
    for linha in linhas:
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
                    pass

    return nome_instancia, capacidade, clientes


def construir_distancias(clientes):
    """Retorna matriz de distâncias Euclidianas entre todos os clientes."""
    coordenadas = np.array([[c['x'], c['y']] for c in clientes], dtype=np.float64)
    diferenca = coordenadas[:, None, :] - coordenadas[None, :, :]
    return np.sqrt((diferenca**2).sum(axis=2))

# ─────────────────────────────────────────────────────────────────
# 2. CONSTRUÇÃO DE SOLUÇÃO INICIAL
# ─────────────────────────────────────────────────────────────────

def greedy_permutacao(clientes, dist, fator_embaralhamento=0.0):
    """Constrói permutação de clientes com heurística gulosa e ruído opcional."""
    nao_visitados = list(range(1, len(clientes)))
    permutacao = []
    atual = 0

    while nao_visitados:
        candidatos = [
            (dist[atual, j] * (1 + random.uniform(0, fator_embaralhamento)), j)
            for j in nao_visitados
        ]
        _, proximo = min(candidatos)
        permutacao.append(proximo)
        nao_visitados.remove(proximo)
        atual = proximo

    return permutacao


def permutacao_aleatoria(n):
    perm = list(range(1, n + 1))
    random.shuffle(perm)
    return perm

# ─────────────────────────────────────────────────────────────────
# 3. DECODIFICADOR DE PERMUTAÇÃO PARA ROTAS
# ─────────────────────────────────────────────────────────────────

def decodificar(permutacao, clientes, capacidade, dist):
    """Transforma uma permutação em um conjunto de rotas que respeitam as janelas."""
    limite_deposito = clientes[0]['due_date']
    rotas = []
    rota_atual = [0]
    carga = 0.0
    tempo = 0.0

    for cid in permutacao:
        cliente = clientes[cid]
        anterior = rota_atual[-1]
        inicio = max(tempo + dist[anterior, cid], cliente['ready_time'])

        if (inicio <= cliente['due_date'] and
                carga + cliente['demand'] <= capacidade and
                inicio + cliente['service_time'] + dist[cid, 0] <= limite_deposito):
            rota_atual.append(cid)
            carga += cliente['demand']
            tempo = inicio + cliente['service_time']
        else:
            rotas.append(rota_atual + [0])
            rota_atual = [0, cid]
            carga = cliente['demand']
            tempo = max(dist[0, cid], cliente['ready_time']) + cliente['service_time']

    rotas.append(rota_atual + [0])
    return rotas


def rotas_para_permutacao(rotas):
    return [cid for rota in rotas for cid in rota[1:-1]]


def custo_insercao_tw(cid, pos, rota, clientes, dist):
    """
    Custo de inserção considerando delta de distância
    + penalidade por folga de janela consumida.
    """
    u, v = rota[pos - 1], rota[pos]
    delta_dist = dist[u, cid] + dist[cid, v] - dist[u, v]

    # folga de janela do cliente seguinte após inserção
    cliente = clientes[cid]
    # quanto da janela do próximo nó é consumida
    cliente_v = clientes[v] if v != 0 else clientes[0]
    folga_v = max(0.0, cliente_v['due_date'] - cliente_v['ready_time'])
    penalidade_tw = delta_dist / (folga_v + 1e-6)  # penaliza se folga pequena

    return delta_dist + 0.3 * penalidade_tw


def encontrar_melhor_insercao(cid, cliente, rotas, clientes, capacidade, dist):
    melhor = None
    for idx, rota in enumerate(rotas):
        carga = sum(clientes[x]['demand'] for x in rota if x != 0)
        if carga + cliente['demand'] > capacidade:
            continue
        for pos in range(1, len(rota)):
            candidato = rota[:pos] + [cid] + rota[pos:]
            if rota_valida(candidato, clientes, capacidade, dist):
                custo = custo_insercao_tw(cid, pos, rota, clientes, dist)
                if melhor is None or custo < melhor[0]:
                    melhor = (custo, idx, pos)
    return melhor


def inserir_cliente_guloso(cid, cliente, rotas, clientes, capacidade, dist):
    """Insere cliente na melhor posição válida em um conjunto de rotas."""
    melhor = encontrar_melhor_insercao(cid, cliente, rotas, clientes, capacidade, dist)
    if melhor is not None:
        _, idx, pos = melhor
        rotas[idx] = rotas[idx][:pos] + [cid] + rotas[idx][pos:]
        return True

    rotas.append([0, cid, 0])
    return False

# ─────────────────────────────────────────────────────────────────
# 4. FUNÇÕES DE AVALIAÇÃO
# ─────────────────────────────────────────────────────────────────

def custo_rota(rota, dist):
    r = np.asarray(rota)
    return float(dist[r[:-1], r[1:]].sum())


def avaliar(rotas, dist):
    return len(rotas), sum(custo_rota(rota, dist) for rota in rotas)


def rota_valida(rota, clientes, capacidade, dist):
    """Verifica se a rota é viável para capacidade, janelas e retorno."""
    limite_deposito = clientes[0]['due_date']
    carga = 0.0
    tempo = 0.0

    for i in range(len(rota) - 1):
        atual, proximo = rota[i], rota[i + 1]
        cliente = clientes[proximo] if proximo != 0 else clientes[0]
        tempo = max(tempo + dist[atual, proximo], cliente['ready_time'])
        if tempo > cliente['due_date']:
            return False
        carga += cliente['demand']
        if carga > capacidade:
            return False
        tempo += cliente['service_time']

    return tempo <= limite_deposito


def rotas_viaveis(rotas, clientes, capacidade, dist):
    return all(rota_valida(rota, clientes, capacidade, dist) for rota in rotas)


def pontuacao(n_veiculos, distancia_total):
    return n_veiculos * 10000 + distancia_total

# ─────────────────────────────────────────────────────────────────
# 5. BUSCA LOCAL
# ─────────────────────────────────────────────────────────────────

def dois_opt(rota, clientes, capacidade, dist, prazo):
    """Aplica melhoria 2-opt em uma rota até não haver mais ganhos."""
    melhor = rota[:]
    n = len(melhor)
    melhorou = True

    while melhorou and time.time() < prazo:
        melhorou = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                ganho = (
                    dist[melhor[i - 1], melhor[i]] + dist[melhor[j], melhor[j + 1]]
                    - dist[melhor[i - 1], melhor[j]] - dist[melhor[i], melhor[j + 1]]
                )
                if ganho > 1e-6:
                    candidato = melhor[:i] + melhor[i:j + 1][::-1] + melhor[j + 1:]
                    if rota_valida(candidato, clientes, capacidade, dist):
                        melhor = candidato
                        melhorou = True
                        break
            if melhorou:
                break

    return melhor


def dois_opt_todas(rotas, clientes, dist, capacidade, prazo):
    return [dois_opt(rota, clientes, capacidade, dist, prazo) for rota in rotas]


def carga_rota(rota, clientes):
    return sum(clientes[c]['demand'] for c in rota if c != 0)


def ganho_remover_segmento(a, b, segmento, dist):
    return dist[a, segmento[0]] + dist[segmento[-1], b] - dist[a, b]


def encontrar_melhor_posicao_insercao_segmento(rota_j, segmento, dist, ganho_remover, ganho_minimo=1e-6):
    melhor_posicao = -1
    melhor_ganho = ganho_minimo
    for pos in range(1, len(rota_j)):
        u, v = rota_j[pos - 1], rota_j[pos]
        ganho_inserir = dist[u, segmento[0]] + dist[segmento[-1], v] - dist[u, v]
        ganho_liquido = ganho_remover - ganho_inserir
        if ganho_liquido > melhor_ganho:
            melhor_ganho = ganho_liquido
            melhor_posicao = pos
    return melhor_posicao, melhor_ganho


def tentar_mover_segmento(rotas, i, j, inicio, tamanho, interna, segmento, clientes, capacidade, dist):
    rota_i = rotas[i]
    rota_j = rotas[j]
    a = rota_i[inicio]
    b = rota_i[inicio + tamanho + 1]
    ganho_remover = ganho_remover_segmento(a, b, segmento, dist)

    posicao, _ = encontrar_melhor_posicao_insercao_segmento(rota_j, segmento, dist, ganho_remover)
    if posicao < 0:
        return False

    nova_rota_i = [rota_i[0]] + interna[:inicio] + interna[inicio + tamanho:] + [rota_i[-1]]
    nova_rota_j = rota_j[:posicao] + segmento + rota_j[posicao:]
    if not rota_valida(nova_rota_i, clientes, capacidade, dist):
        return False
    if not rota_valida(nova_rota_j, clientes, capacidade, dist):
        return False

    rotas[i] = nova_rota_i
    rotas[j] = nova_rota_j
    return True

def fundir_rotas_pequenas(rotas, clientes, capacidade, dist, prazo, max_clientes=3):
    """Tenta absorver rotas pequenas em outras rotas existentes."""
    rotas = [r[:] for r in rotas]
    alterou = True
    while alterou and time.time() < prazo:
        alterou = False
        # ordena: tenta eliminar as menores primeiro
        rotas.sort(key=lambda r: len(r))
        for i, rota_pequena in enumerate(rotas):
            interna = rota_pequena[1:-1]
            if len(interna) > max_clientes:
                continue
            # tenta inserir todos os clientes dessa rota nas demais
            absorbed = True
            temp_rotas = [r[:] for j, r in enumerate(rotas) if j != i]
            for cid in interna:
                if not inserir_cliente_guloso(cid, clientes[cid], temp_rotas, clientes, capacidade, dist):
                    absorbed = False
                    break
            if absorbed:
                rotas = temp_rotas
                alterou = True
                break
    return rotas

def chave_rota_tw(rota, clientes):
    """Ordena por tempo médio de ready_time para agrupar janelas compatíveis."""
    tempos = [clientes[c]['ready_time'] for c in rota if c != 0]
    return (len(rota), np.mean(tempos) if tempos else 0)

def or_opt(rotas, clientes, dist, capacidade, prazo, tamanhos_segmento=SEG_SIZES):
    """Move segmentos curtos entre rotas para reduzir custo geral."""
    rotas = [rota[:] for rota in rotas]
    alterou = True

    while alterou and time.time() < prazo:
        alterou = False
        rotas.sort(key=lambda r: chave_rota_tw(r, clientes))

        for i in range(len(rotas)):
            interna = rotas[i][1:-1]
            if not interna:
                continue

            for tamanho in tamanhos_segmento:
                if len(interna) < tamanho:
                    continue

                for inicio in range(len(interna) - tamanho + 1):
                    segmento = interna[inicio:inicio + tamanho]
                    carga_segmento = sum(clientes[c]['demand'] for c in segmento)
                    rota_i = rotas[i]

                    for j in range(len(rotas)):
                        if i == j:
                            continue

                        rota_j = rotas[j]
                        carga_j = carga_rota(rota_j, clientes)
                        if carga_j + carga_segmento > capacidade:
                            continue

                        if tentar_mover_segmento(rotas, i, j, inicio, tamanho, interna, segmento, clientes, capacidade, dist):
                            alterou = True
                            break

                    if alterou:
                        break
                if alterou:
                    break
            if alterou:
                break

    return [rota for rota in rotas if len(rota) > 2]


def busca_local(rotas, clientes, dist, capacidade, prazo):
    rotas = fundir_rotas_pequenas(rotas, clientes, capacidade, dist, prazo)
    rotas = dois_opt_todas(rotas, clientes, dist, capacidade, prazo)
    rotas = or_opt(rotas, clientes, dist, capacidade, prazo)
    rotas = fundir_rotas_pequenas(rotas, clientes, capacidade, dist, prazo)  # segunda passagem
    return rotas

# ─────────────────────────────────────────────────────────────────
# 6. PERTURBAÇÃO ILS
# ─────────────────────────────────────────────────────────────────

def perturbar_dupla_ponte(rotas):
    """Embaralha rotas e permuta internamente algumas delas."""
    if len(rotas) < 2:
        return rotas

    novas_rotas = [rota[:] for rota in rotas]
    random.shuffle(novas_rotas)

    for i, rota in enumerate(novas_rotas):
        if random.random() < 0.3 and len(rota) > 3:
            interna = rota[1:-1]
            random.shuffle(interna)
            novas_rotas[i] = [0] + interna + [0]

    return novas_rotas


def perturbar_remocao_aleatoria(rotas, clientes, dist, capacidade, n_remover=3):
    """Remove clientes aleatórios e tenta reinseri-los de forma gulosa."""
    rotas = [rota[:] for rota in rotas]
    clientes_em_rotas = [cid for rota in rotas for cid in rota[1:-1]]

    if len(clientes_em_rotas) <= n_remover:
        return rotas

    removidos = random.sample(clientes_em_rotas, n_remover)

    for cid in removidos:
        for idx, rota in enumerate(rotas):
            if cid in rota:
                rotas[idx] = [x for x in rota if x != cid]
                break

    rotas = [rota for rota in rotas if len(rota) > 2]

    for cid in removidos:
        cliente = clientes[cid]
        inserir_cliente_guloso(cid, cliente, rotas, clientes, capacidade, dist)

    return rotas

# ─────────────────────────────────────────────────────────────────
# 7. CRUZAMENTO RBX (apenas entre soluções elite — rápido)
# ─────────────────────────────────────────────────────────────────

def cruzamento_rbx(rotas_a, rotas_b, clientes, capacidade, dist):
    """Cria um filho usando rotas inteiras de uma solução e completa com outra."""
    selecionadas = [rota for rota in rotas_a if random.random() < PRESERVE_RATE]
    if not selecionadas:
        selecionadas = [random.choice(rotas_a)]

    cobertos = {cid for rota in selecionadas for cid in rota[1:-1]}
    perm_b = rotas_para_permutacao(rotas_b)
    restantes = [cid for cid in perm_b if cid not in cobertos]

    filho = [rota[:] for rota in selecionadas]

    for cid in restantes:
        cliente = clientes[cid]
        inserir_cliente_guloso(cid, cliente, filho, clientes, capacidade, dist)

    return filho


def gerar_solucao_inicial(iteracao, elite, clientes, capacidade, dist, n_remover_perturb):
    """Retorna uma solução inicial para a iteração atual."""
    if iteracao == 0:
        perm = greedy_permutacao(clientes, dist, fator_embaralhamento=0.0)
        return decodificar(perm, clientes, capacidade, dist)

    if elite and random.random() < 0.4 and len(elite) >= 2:
        a, b = random.sample(elite, 2)
        return cruzamento_rbx(a[1], b[1], clientes, capacidade, dist)

    if elite and random.random() < 0.5:
        base = random.choice(elite)[1]
        if random.random() < 0.5:
            return perturbar_remocao_aleatoria(base, clientes, dist, capacidade, n_remover=n_remover_perturb)
        return perturbar_dupla_ponte(base)

    sf = random.uniform(SHUFFLE_FACTOR_MIN, SHUFFLE_FACTOR_MAX)
    perm = greedy_permutacao(clientes, dist, fator_embaralhamento=sf)
    return decodificar(perm, clientes, capacidade, dist)


def calcular_prazo_busca(restante, frac_busca_local):
    """Calcula quanto tempo reservar para busca local nesta iteração."""
    return min(restante * frac_busca_local, max(5.0, restante * 0.3))


def adicionar_a_elite(elite, pont, rotas, elite_size):
    """Mantém o conjunto elite ordenado e limitado."""
    elite.append((pont, [rota[:] for rota in rotas]))
    elite.sort(key=lambda x: x[0])
    return elite[:elite_size]

# ─────────────────────────────────────────────────────────────────
# 8. LOOP PRINCIPAL: ILS + AG
# ─────────────────────────────────────────────────────────────────

def executar(clientes, capacidade, dist,
             elite_size=ELITE_SIZE,
             frac_busca_local=LS_FRAC,
             n_remover_perturb=N_REMOVE_PERTURB,
             limite_tempo=450):
    """Executa o algoritmo híbrido ILS + AG e retorna a melhor solução"""
    n = len(clientes) - 1
    inicio = time.time()
    prazo_total = inicio + limite_tempo

    melhor_rotas = None
    melhor_nv = melhor_td = float('inf')
    melhor_pontuacao = float('inf')

    elite = []
    iteracao = 0
    historico = []

    print(f"  Iniciando ILS+AG | Clientes: {n} | Limite: {limite_tempo}s")

    while time.time() < prazo_total - 3:
        tempo_passado = time.time() - inicio
        restante = prazo_total - time.time()

        rotas = gerar_solucao_inicial(
            iteracao, elite, clientes, capacidade, dist, n_remover_perturb
        )

        prazo_busca = calcular_prazo_busca(restante, frac_busca_local)
        prazo_local = time.time() + prazo_busca
        rotas = busca_local(rotas, clientes, dist, capacidade, prazo_local)

        nv, td = avaliar(rotas, dist)
        pont = pontuacao(nv, td)

        elite = adicionar_a_elite(elite, pont, rotas, elite_size)

        if pont < melhor_pontuacao:
            melhor_pontuacao = pont
            melhor_rotas = [rota[:] for rota in rotas]
            melhor_nv, melhor_td = nv, td
            historico.append((iteracao, melhor_nv, melhor_td))
            print(f"  Iter {iteracao:4d} | ★ Veículos: {melhor_nv} | Dist: {melhor_td:.4f} | {tempo_passado:.1f}s")
        elif iteracao % 5 == 0:
            print(f"  Iter {iteracao:4d} | Veículos: {nv} | Dist: {td:.4f} | melhor={melhor_nv}/{melhor_td:.4f} | {tempo_passado:.1f}s")

        iteracao += 1

    restante = prazo_total - time.time()
    if melhor_rotas and restante > 2:
        print("  BL final...")
        final = busca_local(melhor_rotas, clientes, dist, capacidade, time.time() + restante - 1)
        nv_f, td_f = avaliar(final, dist)
        if pontuacao(nv_f, td_f) <= melhor_pontuacao:
            melhor_rotas, melhor_nv, melhor_td = final, nv_f, td_f

    return melhor_rotas, melhor_nv, melhor_td, time.time() - inicio, historico

# ─────────────────────────────────────────────────────────────────
# 9. SAÍDA
# ─────────────────────────────────────────────────────────────────

def formatar_saida(nome_instancia, autores, algoritmo, rotas, nv, td, tempo_total):
    linhas = [
        f"======== MELHOR SOLUCAO {algoritmo} ========",
        f"Nome da instancia : {nome_instancia}",
        f"Autores : {autores}",
        f"Numero de veiculos: {nv}",
        f"Distancia total: {td:.4f}",
        f"Tempo total: {tempo_total:.0f}s",
        "Rotas:",
    ]
    linhas += [f"Rota {i + 1}: {' -> '.join(map(str, rota))}" for i, rota in enumerate(rotas)]
    return "\n".join(linhas)

def salvar_saida(nome_instancia, algoritmo, conteudo, diretorio_saida="."):
    caminho = os.path.join(diretorio_saida, f"{nome_instancia}_resultado_{algoritmo.lower()}.txt")
    with open(caminho, 'w', encoding='utf-8') as arquivo:
        arquivo.write(conteudo)
    return caminho

# ─────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    # ════════════════════════════════════
    AUTHORS        = "Autor A e Autor B"  # <- nomes do grupo
    ALGORITHM_NAME = "AG"
    TIME_LIMIT     = 450   # 8 min - 30s de margem
    OUTPUT_DIR     = "."
    # ════════════════════════════════════

    random.seed(SEED)
    np.random.seed(SEED)

    if len(sys.argv) < 2:
        print("Uso: python vrptw_ag.py <instancia.txt> [instancia2.txt ...]")
        sys.exit(1)

    # Reporter
    try:
        from vrptw_report import Reporter
        rep = Reporter(authors=AUTHORS, algorithm=ALGORITHM_NAME)
        use_reporter = True
    except ImportError:
        rep = None
        use_reporter = False

    for caminho in sys.argv[1:]:
        print(f"\n{'=' * 55}\nInstância: {caminho}\n{'=' * 55}")
        nome_instancia, capacidade, clientes = ler_instancia(caminho)
        dist = construir_distancias(clientes)

        rotas, nv, td, tempo_total, historico = executar(
            clientes, capacidade, dist,
            elite_size=ELITE_SIZE,
            frac_busca_local=LS_FRAC,
            n_remover_perturb=N_REMOVE_PERTURB,
            limite_tempo=TIME_LIMIT,
        )

        viavel = rotas_viaveis(rotas, clientes, capacidade, dist)

        if use_reporter:
            rep.record(nome_instancia, rotas, nv, td, tempo_total,
                       clientes, dist, historico, viavel)
        else:
            print(f"\n  Viável: {viavel}")
            print(f"  Veículos: {nv} | Distância: {td:.4f} | Tempo: {tempo_total:.1f}s")

        saida_texto = formatar_saida(nome_instancia, AUTHORS, ALGORITHM_NAME,
                                    rotas, nv, td, tempo_total)
        caminho_saida = salvar_saida(nome_instancia, ALGORITHM_NAME, saida_texto, OUTPUT_DIR)
        print(f"  Arquivo salvo: {caminho_saida}")

    # Sumário e relatório HTML
    if use_reporter and rep.results:
        rep.print_summary()
        rep.save_html(os.path.join(OUTPUT_DIR, "resultados.html"))

if __name__ == "__main__":
    main()