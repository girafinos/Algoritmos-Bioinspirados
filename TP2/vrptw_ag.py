"""
AG + ILS Híbrido para VRPTW
Estratégia:
  - Múltiplos inícios independentes (ILS) com perturbação greedy
  - Busca local intensa em cada início (2-opt + or-opt)
  - AG opera sobre o conjunto de melhores soluções encontradas
  - O cruzamento RBX combina rotas de soluções já boas (elite)
"""

import random, time, sys, os
import numpy as np

# ─────────────────────────────────────────────────────────────────
# 1. LEITURA
# ─────────────────────────────────────────────────────────────────

def parse_instance(filepath):
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    instance_name = lines[0]
    capacity = None
    for i, line in enumerate(lines):
        if line.upper().startswith('NUMBER'):
            capacity = int(lines[i+1].split()[1])
            break
    customers = []
    reading = False
    for line in lines:
        if line.upper().startswith('CUST'):
            reading = True; continue
        if reading:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    customers.append({
                        'id': int(parts[0]), 'x': float(parts[1]), 'y': float(parts[2]),
                        'demand': float(parts[3]), 'ready_time': float(parts[4]),
                        'due_date': float(parts[5]), 'service_time': float(parts[6]),
                    })
                except ValueError:
                    pass
    return instance_name, capacity, customers

def build_dist(customers):
    coords = np.array([[c['x'], c['y']] for c in customers], dtype=np.float64)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(axis=2))

# ─────────────────────────────────────────────────────────────────
# 2. CONSTRUÇÃO DE SOLUÇÕES
# ─────────────────────────────────────────────────────────────────

def greedy_permutation(customers, dist, shuffle_factor=0.0):
    unvisited = list(range(1, len(customers)))
    perm, current = [], 0
    while unvisited:
        dists = [(dist[current, j] * (1 + random.uniform(0, shuffle_factor)), j)
                 for j in unvisited]
        _, nxt = min(dists)
        perm.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return perm

def random_permutation(n):
    perm = list(range(1, n + 1))
    random.shuffle(perm)
    return perm

# ─────────────────────────────────────────────────────────────────
# 3. DECODER
# ─────────────────────────────────────────────────────────────────

def decode(permutation, customers, capacity, dist):
    depot_due = customers[0]['due_date']
    routes, route = [], [0]
    load = t = 0.0
    for cid in permutation:
        c = customers[cid]
        prev = route[-1]
        start = max(t + dist[prev, cid], c['ready_time'])
        if (start <= c['due_date'] and
                load + c['demand'] <= capacity and
                start + c['service_time'] + dist[cid, 0] <= depot_due):
            route.append(cid); load += c['demand']
            t = start + c['service_time']
        else:
            routes.append(route + [0])
            s2 = max(dist[0, cid], c['ready_time'])
            route = [0, cid]; load = c['demand']
            t = s2 + c['service_time']
    routes.append(route + [0])
    return routes

def routes_to_perm(routes):
    return [cid for r in routes for cid in r[1:-1]]

# ─────────────────────────────────────────────────────────────────
# 4. AVALIAÇÃO
# ─────────────────────────────────────────────────────────────────

def route_cost(route, dist):
    r = np.asarray(route)
    return float(dist[r[:-1], r[1:]].sum())

def evaluate(routes, dist):
    return len(routes), sum(route_cost(r, dist) for r in routes)

def route_ok(route, customers, capacity, dist):
    depot_due = customers[0]['due_date']
    load = t = 0.0
    for i in range(len(route) - 1):
        a, b = route[i], route[i+1]
        c = customers[b] if b != 0 else customers[0]
        t = max(t + dist[a, b], c['ready_time'])
        if t > c['due_date']: return False
        load += c['demand']
        if load > capacity: return False
        t += c['service_time']
    return t <= depot_due

def is_feasible(routes, customers, capacity, dist):
    return all(route_ok(r, customers, capacity, dist) for r in routes)

def fitness(nv, td): return nv * 10000 + td

# ─────────────────────────────────────────────────────────────────
# 5. BUSCA LOCAL
# ─────────────────────────────────────────────────────────────────

def two_opt(route, customers, capacity, dist, deadline):
    best = route[:]
    n = len(best)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                gain = (dist[best[i-1], best[i]] + dist[best[j], best[j+1]]
                      - dist[best[i-1], best[j]] - dist[best[i], best[j+1]])
                if gain > 1e-6:
                    cand = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if route_ok(cand, customers, capacity, dist):
                        best = cand; improved = True
    return best

def two_opt_all(routes, customers, dist, capacity, deadline):
    return [two_opt(r, customers, capacity, dist, deadline) for r in routes]

def or_opt(routes, customers, dist, capacity, deadline, seg_sizes=(1, 2, 3)):
    routes = [r[:] for r in routes]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        routes.sort(key=len)
        for i in range(len(routes)):
            inner_i = routes[i][1:-1]
            if not inner_i: continue
            for seg_sz in seg_sizes:
                if len(inner_i) < seg_sz: continue
                for p in range(len(inner_i) - seg_sz + 1):
                    seg = inner_i[p:p+seg_sz]
                    seg_load = sum(customers[s]['demand'] for s in seg)
                    ri = routes[i]
                    a, b = ri[p], ri[p+seg_sz+1]
                    gain_remove = dist[a, seg[0]] + dist[seg[-1], b] - dist[a, b]
                    for j in range(len(routes)):
                        if i == j: continue
                        rj = routes[j]
                        load_j = sum(customers[c]['demand'] for c in rj if c != 0)
                        if load_j + seg_load > capacity: continue
                        best_pos, best_gain = -1, 1e-6
                        for ins in range(1, len(rj)):
                            u, v = rj[ins-1], rj[ins]
                            net = gain_remove - (dist[u, seg[0]] + dist[seg[-1], v] - dist[u, v])
                            if net > best_gain:
                                best_gain = net; best_pos = ins
                        if best_pos >= 0:
                            new_ri = [ri[0]] + inner_i[:p] + inner_i[p+seg_sz:] + [ri[-1]]
                            new_rj = rj[:best_pos] + seg + rj[best_pos:]
                            if route_ok(new_ri, customers, capacity, dist) and \
                               route_ok(new_rj, customers, capacity, dist):
                                routes[i] = new_ri
                                routes[j] = new_rj
                                changed = True; break
                    if changed: break
                if changed: break
            if changed: break
    return [r for r in routes if len(r) > 2]

def local_search(routes, customers, dist, capacity, deadline):
    r = two_opt_all(routes, customers, dist, capacity, deadline)
    r = or_opt(r, customers, dist, capacity, deadline)
    return r

# ─────────────────────────────────────────────────────────────────
# 6. PERTURBAÇÃO ILS (double-bridge / random destroy)
# ─────────────────────────────────────────────────────────────────

def perturb_double_bridge(routes):
    """
    Double-bridge: embaralha a ordem das rotas e inverte algumas delas.
    Escapa de ótimos locais sem destruir a estrutura de rotas.
    """
    if len(routes) < 2:
        return routes
    r = [route[:] for route in routes]
    random.shuffle(r)
    # Inverte aleatoriamente algumas rotas internamente
    for i in range(len(r)):
        if random.random() < 0.3 and len(r[i]) > 3:
            inner = r[i][1:-1]
            random.shuffle(inner)
            r[i] = [0] + inner + [0]
    return r

def perturb_random_remove(routes, customers, dist, capacity, n_remove=3):
    """
    Remove n_remove clientes aleatórios e reinsere guloso.
    Perturbação mais cirúrgica que o double-bridge.
    """
    routes = [r[:] for r in routes]
    all_clients = [cid for r in routes for cid in r[1:-1]]
    if len(all_clients) <= n_remove:
        return routes

    to_remove = random.sample(all_clients, n_remove)

    # Remove os clientes das rotas
    for cid in to_remove:
        for i, r in enumerate(routes):
            if cid in r:
                routes[i] = [x for x in r if x != cid]
                break
    routes = [r for r in routes if len(r) > 2]

    # Reinsere cada cliente na melhor posição disponível
    for cid in to_remove:
        c = customers[cid]
        best = None
        for ri, route in enumerate(routes):
            load = sum(customers[x]['demand'] for x in route if x != 0)
            if load + c['demand'] > capacity:
                continue
            for ins in range(1, len(route)):
                candidate = route[:ins] + [cid] + route[ins:]
                if route_ok(candidate, customers, capacity, dist):
                    cost = route_cost(candidate, dist) - route_cost(route, dist)
                    if best is None or cost < best[0]:
                        best = (cost, ri, ins)
        if best:
            _, ri, ins = best
            routes[ri] = routes[ri][:ins] + [cid] + routes[ri][ins:]
        else:
            routes.append([0, cid, 0])

    return routes

# ─────────────────────────────────────────────────────────────────
# 7. CRUZAMENTO RBX (apenas entre soluções elite — rápido)
# ─────────────────────────────────────────────────────────────────

def rbx(routes_a, routes_b, customers, capacity, dist):
    """
    Opera diretamente em rotas (não em permutações).
    Seleciona rotas inteiras de routes_a, completa com clientes
    de routes_b na ordem em que aparecem.
    """
    selected = [r for r in routes_a if random.random() < 0.4]
    if not selected:
        selected = [random.choice(routes_a)]

    covered = {cid for r in selected for cid in r[1:-1]}

    # Clientes de b que não estão cobertos, na ordem de b
    perm_b = routes_to_perm(routes_b)
    remaining = [cid for cid in perm_b if cid not in covered]

    child_routes = [r[:] for r in selected]

    # Reinsere restantes guloso (poucos clientes → rápido)
    for cid in remaining:
        c = customers[cid]
        best = None
        for ri, route in enumerate(child_routes):
            load = sum(customers[x]['demand'] for x in route if x != 0)
            if load + c['demand'] > capacity: continue
            for ins in range(1, len(route)):
                cand = route[:ins] + [cid] + route[ins:]
                if route_ok(cand, customers, capacity, dist):
                    cost = route_cost(cand, dist) - route_cost(route, dist)
                    if best is None or cost < best[0]:
                        best = (cost, ri, ins)
        if best:
            _, ri, ins = best
            child_routes[ri] = child_routes[ri][:ins] + [cid] + child_routes[ri][ins:]
        else:
            child_routes.append([0, cid, 0])

    return child_routes

# ─────────────────────────────────────────────────────────────────
# 8. LOOP PRINCIPAL: ILS + AG
# ─────────────────────────────────────────────────────────────────

def run(customers, capacity, dist,
        elite_size=8,
        ls_frac=0.55,          # fração do tempo para busca local por iteração
        n_remove_perturb=4,    # clientes removidos na perturbação
        time_limit=450):

    n = len(customers) - 1
    t0 = time.time()
    deadline_total = t0 + time_limit

    best_routes = None
    best_nv = best_td = float('inf')
    best_sc = float('inf')

    elite = []   # lista de (score, routes) — soluções elite para o cruzamento
    iteration = 0
    history = []  # [(iteracao, nv, td)] para gráfico de convergência

    print(f"  Iniciando ILS+AG | Clientes: {n} | Limite: {time_limit}s")

    while time.time() < deadline_total - 3:
        elapsed = time.time() - t0
        remaining = deadline_total - time.time()

        # ── Geração de solução inicial desta iteração ──────────────
        if iteration == 0:
            # Primeira iteração: greedy puro
            perm = greedy_permutation(customers, dist, shuffle_factor=0.0)
            routes = decode(perm, customers, capacity, dist)

        elif elite and random.random() < 0.4 and len(elite) >= 2:
            # Cruzamento entre dois membros da elite
            a, b = random.sample(elite, 2)
            routes = rbx(a[1], b[1], customers, capacity, dist)

        elif elite and random.random() < 0.5:
            # Perturbação de um membro da elite
            base = random.choice(elite)[1]
            if random.random() < 0.5:
                routes = perturb_random_remove(base, customers, dist, capacity,
                                               n_remove=n_remove_perturb)
            else:
                routes = perturb_double_bridge(base)

        else:
            # Novo início aleatório / greedy perturbado
            sf = random.uniform(0.3, 1.5)
            perm = greedy_permutation(customers, dist, shuffle_factor=sf)
            routes = decode(perm, customers, capacity, dist)

        # ── Busca local ────────────────────────────────────────────
        # Dá mais tempo nas primeiras iterações, menos quando o tempo aperta
        ls_time = min(remaining * ls_frac, max(5.0, remaining * 0.3))
        ls_deadline = time.time() + ls_time
        routes = local_search(routes, customers, dist, capacity, ls_deadline)

        nv, td = evaluate(routes, dist)
        sc = fitness(nv, td)

        # ── Atualiza elite ─────────────────────────────────────────
        elite.append((sc, [r[:] for r in routes]))
        elite.sort(key=lambda x: x[0])
        elite = elite[:elite_size]

        # ── Atualiza melhor global ─────────────────────────────────
        if sc < best_sc:
            best_sc = sc
            best_routes = [r[:] for r in routes]
            best_nv, best_td = nv, td
            history.append((iteration, best_nv, best_td))
            print(f"  Iter {iteration:4d} | ★ Veículos: {best_nv} | "
                  f"Dist: {best_td:.4f} | {elapsed:.1f}s")
        elif iteration % 5 == 0:
            print(f"  Iter {iteration:4d} | Veículos: {nv} | "
                  f"Dist: {td:.4f} | melhor={best_nv}/{best_td:.4f} | {elapsed:.1f}s")

        iteration += 1

    # ── BL final no melhor ─────────────────────────────────────────
    remaining = deadline_total - time.time()
    if best_routes and remaining > 2:
        print("  BL final...")
        final = local_search(best_routes, customers, dist, capacity,
                             time.time() + remaining - 1)
        nv_f, td_f = evaluate(final, dist)
        if fitness(nv_f, td_f) <= best_sc:
            best_routes, best_nv, best_td = final, nv_f, td_f

    return best_routes, best_nv, best_td, time.time() - t0, history

# ─────────────────────────────────────────────────────────────────
# 9. SAÍDA
# ─────────────────────────────────────────────────────────────────

def format_output(instance_name, authors, alg, routes, nv, td, elapsed):
    lines = [
        f"======== MELHOR SOLUCAO {alg} ========",
        f"Nome da instancia : {instance_name}",
        f"Autores : {authors}",
        f"Numero de veiculos: {nv}",
        f"Distancia total: {td:.4f}",
        f"Tempo total: {elapsed:.0f}s",
        "Rotas:",
    ] + [f"Rota {i+1}: {' -> '.join(map(str, r))}" for i, r in enumerate(routes)]
    return "\n".join(lines)

def save_output(name, alg, content, out_dir="."):
    path = os.path.join(out_dir, f"{name}_resultado_{alg.lower()}.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path

# ─────────────────────────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    # ════════════════════════════════════
    AUTHORS        = "Autor A e Autor B"  # <- nomes do grupo
    ALGORITHM_NAME = "AG"
    TIME_LIMIT     = 450   # 8 min - 30s de margem
    OUTPUT_DIR     = "."
    SEED           = 42

    ELITE_SIZE       = 8
    LS_FRAC          = 0.55
    N_REMOVE_PERTURB = 4
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

    for filepath in sys.argv[1:]:
        print(f"\n{'='*55}\nInstância: {filepath}\n{'='*55}")
        instance_name, capacity, customers = parse_instance(filepath)
        dist = build_dist(customers)

        routes, nv, td, elapsed, history = run(
            customers, capacity, dist,
            elite_size=ELITE_SIZE,
            ls_frac=LS_FRAC,
            n_remove_perturb=N_REMOVE_PERTURB,
            time_limit=TIME_LIMIT,
        )

        feasible = is_feasible(routes, customers, capacity, dist)

        if use_reporter:
            rep.record(instance_name, routes, nv, td, elapsed,
                       customers, dist, history, feasible)
        else:
            print(f"\n  Viável: {feasible}")
            print(f"  Veículos: {nv} | Distância: {td:.4f} | Tempo: {elapsed:.1f}s")

        out = format_output(instance_name, AUTHORS, ALGORITHM_NAME,
                            routes, nv, td, elapsed)
        path = save_output(instance_name, ALGORITHM_NAME, out, OUTPUT_DIR)
        print(f"  Arquivo salvo: {path}")

    # Sumário e relatório HTML
    if use_reporter and rep.results:
        rep.print_summary()
        rep.save_html(os.path.join(OUTPUT_DIR, "resultados.html"))

if __name__ == "__main__":
    main()