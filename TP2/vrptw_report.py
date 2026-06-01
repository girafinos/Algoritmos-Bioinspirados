"""
vrptw_report.py — Módulo de análise e relatório para o AG VRPTW
Uso:
    from vrptw_report import Reporter
    rep = Reporter()
    rep.record(instance_name, routes, nv, td, elapsed, customers, dist, history)
    rep.print_summary()
    rep.save_html("resultados.html")

O 'history' é uma lista de (iteracao, nv, td) coletada durante o run.
"""

import os
import math
import base64
import io
import html as html_lib
from datetime import datetime

# Benchmarks Solomon clássicos (ótimos conhecidos para instâncias c1/c2/r1/r2/rc1/rc2)
SOLOMON_BEST = {
    "c101": (10, 828.94), "c102": (10, 828.94), "c103": (10, 828.06),
    "c104": (10, 824.78), "c105": (10, 828.94), "c106": (10, 828.94),
    "c107": (10, 828.94), "c108": (10, 828.94), "c109": (10, 828.94),
    "c201": (3, 591.56),  "c202": (3, 591.56),  "c203": (3, 591.17),
    "c204": (3, 590.60),  "c205": (3, 588.88),  "c206": (3, 588.49),
    "c207": (3, 588.29),  "c208": (3, 588.32),
    "r101": (19, 1645.79),"r102": (17, 1486.12),"r103": (13, 1292.68),
    "r104": (9,  1007.24),"r105": (14, 1377.11),"r106": (12, 1251.98),
    "r107": (10, 1104.66),"r108": (9,   960.88),"r109": (11, 1194.73),
    "r110": (10, 1118.59),"r111": (10, 1096.72),"r112": (9,  982.14),
    "r201": (4, 1252.37), "r202": (3, 1191.70), "r203": (3, 939.50),
    "r204": (2,  825.52), "r205": (3, 994.43),  "r206": (3, 906.14),
    "r207": (2,  890.61), "r208": (2, 726.75),  "r209": (3, 909.16),
    "r210": (3,  939.37), "r211": (2, 892.71),
    "rc101":(14,1696.94), "rc102":(12,1554.75), "rc103":(11,1261.67),
    "rc104":(10,1135.48), "rc105":(13,1629.44), "rc106":(11,1424.73),
    "rc107":(11,1230.48), "rc108":(10,1139.82),
    "rc201":(4, 1406.91), "rc202":(3, 1367.09), "rc203":(3, 1049.62),
    "rc204":(3,  798.41), "rc205":(4, 1297.19), "rc206":(3, 1146.32),
    "rc207":(3,  1061.14),"rc208":(3,  828.14),
}


# ─────────────────────────────────────────────────────────────────
# TERMINAL
# ─────────────────────────────────────────────────────────────────

def _bar(value, max_value, width=20, char="█", bg="░"):
    filled = int(round(value / max_value * width)) if max_value > 0 else 0
    return char * filled + bg * (width - filled)

def print_progress(iteration, elapsed, time_limit, best_nv, best_td, current_nv, current_td):
    """Barra de progresso inline para chamar dentro do loop."""
    pct = min(elapsed / time_limit, 1.0)
    remaining = max(0, time_limit - elapsed)
    bar = _bar(pct, 1.0, width=25)
    print(
        f"\r  [{bar}] {pct*100:4.0f}% | "
        f"{remaining:4.0f}s restantes | "
        f"melhor: {best_nv}v / {best_td:,.1f}km | "
        f"atual: {current_nv}v / {current_td:,.1f}km   ",
        end="", flush=True
    )

def print_instance_result(instance_name, nv, td, elapsed, feasible, best_known=None):
    """Resultado final de uma instância no terminal."""
    status = "✓ VIÁVEL" if feasible else "✗ INVIÁVEL"
    gap_v = gap_d = "—"
    if best_known:
        bv, bd = best_known
        gap_v = f"{nv - bv:+d}"
        gap_d = f"{(td - bd) / bd * 100:+.1f}%"

    sep = "─" * 55
    print(f"\n\n  {sep}")
    print(f"  Instância : {instance_name}")
    print(f"  Status    : {status}")
    print(f"  Veículos  : {nv}  (gap vs benchmark: {gap_v})")
    print(f"  Distância : {td:.4f}  (gap: {gap_d})")
    print(f"  Tempo     : {elapsed:.1f}s")
    print(f"  {sep}")


# ─────────────────────────────────────────────────────────────────
# MATPLOTLIB — convergência e mapa de rotas
# ─────────────────────────────────────────────────────────────────

def _plot_convergence_b64(history, instance_name):
    """Retorna imagem PNG em base64 do gráfico de convergência."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        iters = [h[0] for h in history]
        tds   = [h[2] for h in history]
        nvs   = [h[1] for h in history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
        fig.patch.set_facecolor("#0f1117")

        for ax in (ax1, ax2):
            ax.set_facecolor("#1a1d27")
            ax.tick_params(colors="#aab0c0", labelsize=8)
            ax.spines["bottom"].set_color("#2e3245")
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_color("#2e3245")
            ax.spines["right"].set_visible(False)
            ax.yaxis.label.set_color("#aab0c0")

        ax1.plot(iters, tds, color="#4ecca3", linewidth=1.2, alpha=0.9)
        ax1.fill_between(iters, tds, min(tds), alpha=0.12, color="#4ecca3")
        ax1.set_ylabel("Distância total", fontsize=8)

        ax2.step(iters, nvs, color="#f7a541", linewidth=1.2, where="post")
        ax2.set_ylabel("Nº veículos", fontsize=8)
        ax2.set_xlabel("Iteração", fontsize=8, color="#aab0c0")
        ax2.set_yticks(sorted(set(nvs)))

        fig.suptitle(f"Convergência — {instance_name}",
                     color="#e8ecf4", fontsize=10, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        return None


def _plot_routes_b64(routes, customers, instance_name):
    """Retorna imagem PNG em base64 do mapa de rotas."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#aab0c0", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#2e3245")

        xs = [c['x'] for c in customers]
        ys = [c['y'] for c in customers]

        # Clientes
        ax.scatter(xs[1:], ys[1:], s=18, color="#4a5270", zorder=3, linewidths=0)
        # Depósito
        ax.scatter(xs[0], ys[0], s=80, marker="*", color="#f7a541",
                   zorder=5, linewidths=0)

        colors = cm.get_cmap("tab20", max(len(routes), 1))
        for idx, route in enumerate(routes):
            rx = [customers[n]['x'] for n in route]
            ry = [customers[n]['y'] for n in route]
            col = colors(idx)
            ax.plot(rx, ry, color=col, linewidth=1.0, alpha=0.75, zorder=2)
            # Pontos da rota
            ax.scatter(rx[1:-1], ry[1:-1], s=22, color=col,
                       zorder=4, linewidths=0)

        ax.set_title(f"Rotas — {instance_name}  ({len(routes)} veículos)",
                     color="#e8ecf4", fontsize=9, fontweight="bold", pad=8)
        ax.set_aspect("equal")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────

_HTML_STYLE = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  :root {
    --bg:      #0f1117;
    --surface: #1a1d27;
    --border:  #2e3245;
    --text:    #e8ecf4;
    --muted:   #7a8099;
    --green:   #4ecca3;
    --amber:   #f7a541;
    --red:     #ff6b6b;
    --blue:    #5b9cf6;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'IBM Plex Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 2.5rem 2rem;
    min-height: 100vh;
  }

  h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--green);
    letter-spacing: 0.04em;
    margin-bottom: 0.3rem;
  }

  .subtitle {
    font-size: 0.82rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 2.5rem;
  }

  /* ── SUMMARY CARDS ── */
  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin-bottom: 2.5rem;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
  }

  .card-label {
    font-size: 0.72rem;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
  }

  .card-value {
    font-size: 1.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    color: var(--green);
  }

  .card-value.amber { color: var(--amber); }
  .card-value.blue  { color: var(--blue);  }

  /* ── TABLE ── */
  .table-wrap {
    overflow-x: auto;
    margin-bottom: 3rem;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
  }

  thead th {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    padding: 0.6rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }

  tbody tr {
    border-bottom: 1px solid var(--border);
    transition: background 0.15s;
  }

  tbody tr:hover { background: rgba(255,255,255,0.03); }

  tbody td {
    padding: 0.65rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    white-space: nowrap;
  }

  .td-name  { color: var(--text);  font-weight: 600; }
  .td-ok    { color: var(--green); }
  .td-warn  { color: var(--amber); }
  .td-bad   { color: var(--red);   }
  .td-muted { color: var(--muted); }

  .badge {
    display: inline-block;
    padding: 0.1rem 0.5rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
  }

  .badge-ok  { background: rgba(78,204,163,0.15); color: var(--green); }
  .badge-bad { background: rgba(255,107,107,0.15); color: var(--red);  }

  /* ── SECTION TITLE ── */
  .section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 1.2rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
  }

  /* ── INSTANCE BLOCKS ── */
  .instance-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 2.5rem;
  }

  @media (max-width: 800px) {
    .instance-grid { grid-template-columns: 1fr; }
  }

  .chart-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }

  .chart-card img {
    width: 100%;
    display: block;
  }

  .instance-section { margin-bottom: 3rem; }

  .inst-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem;
    font-weight: 600;
    color: var(--amber);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
  }

  .inst-header span {
    font-size: 0.75rem;
    color: var(--muted);
    font-weight: 400;
  }
</style>
"""

def _gap_class(gap_pct):
    if gap_pct is None: return "td-muted"
    if gap_pct <= 2:    return "td-ok"
    if gap_pct <= 10:   return "td-warn"
    return "td-bad"


class Reporter:
    def __init__(self, authors="", algorithm="AG"):
        self.authors = authors
        self.algorithm = algorithm
        self.results = []  # list of dicts

    def record(self, instance_name, routes, nv, td, elapsed,
               customers, dist, history=None, feasible=True):
        """Registra resultado de uma instância."""
        key = instance_name.lower().replace(" ", "")
        best_known = SOLOMON_BEST.get(key)

        gap_v = gap_d = None
        if best_known:
            bv, bd = best_known
            gap_v = nv - bv
            gap_d = (td - bd) / bd * 100 if bd > 0 else None

        self.results.append({
            "name":       instance_name,
            "nv":         nv,
            "td":         td,
            "elapsed":    elapsed,
            "feasible":   feasible,
            "gap_v":      gap_v,
            "gap_d":      gap_d,
            "best_known": best_known,
            "routes":     routes,
            "customers":  customers,
            "dist":       dist,
            "history":    history or [],
        })

        # Imprime resultado no terminal
        print_instance_result(instance_name, nv, td, elapsed, feasible, best_known)

    # ── TERMINAL SUMMARY ──────────────────────────────────────────

    def print_summary(self):
        if not self.results:
            return

        sep  = "═" * 75
        sep2 = "─" * 75
        print(f"\n\n  {sep}")
        print(f"  {'RESUMO FINAL':^73}")
        print(f"  {sep}")

        header = f"  {'Instância':<14} {'Veíc':>5} {'Δv':>5} {'Distância':>12} {'Δdist%':>8} {'Tempo':>7} {'Status':>8}"
        print(header)
        print(f"  {sep2}")

        for r in self.results:
            gv = f"{r['gap_v']:+d}" if r['gap_v'] is not None else "  —"
            gd = f"{r['gap_d']:+.1f}%" if r['gap_d'] is not None else "     —"
            st = "✓" if r['feasible'] else "✗"
            print(f"  {r['name']:<14} {r['nv']:>5} {gv:>5} "
                  f"{r['td']:>12.4f} {gd:>8} {r['elapsed']:>6.0f}s {st:>8}")

        print(f"  {sep2}")
        n_ok  = sum(1 for r in self.results if r['feasible'])
        n_tot = len(self.results)
        avg_gd = [r['gap_d'] for r in self.results if r['gap_d'] is not None]
        avg_str = f"{sum(avg_gd)/len(avg_gd):+.1f}%" if avg_gd else "—"
        print(f"  Instâncias: {n_tot}  |  Viáveis: {n_ok}/{n_tot}  |  Gap médio distância: {avg_str}")
        print(f"  {sep}\n")

    # ── HTML ──────────────────────────────────────────────────────

    def save_html(self, filepath="resultados.html"):
        ts = datetime.now().strftime("%d/%m/%Y %H:%M")
        n_ok  = sum(1 for r in self.results if r['feasible'])
        n_tot = len(self.results)
        avg_nv = sum(r['nv'] for r in self.results) / n_tot if n_tot else 0
        avg_gd = [r['gap_d'] for r in self.results if r['gap_d'] is not None]
        avg_gap = f"{sum(avg_gd)/len(avg_gd):+.1f}%" if avg_gd else "—"

        # ── Cards de sumário
        cards_html = f"""
        <div class="cards">
          <div class="card">
            <div class="card-label">Instâncias</div>
            <div class="card-value">{n_tot}</div>
          </div>
          <div class="card">
            <div class="card-label">Viáveis</div>
            <div class="card-value {'amber' if n_ok < n_tot else ''}">{n_ok}/{n_tot}</div>
          </div>
          <div class="card">
            <div class="card-label">Veículos médio</div>
            <div class="card-value amber">{avg_nv:.1f}</div>
          </div>
          <div class="card">
            <div class="card-label">Gap dist médio</div>
            <div class="card-value blue">{avg_gap}</div>
          </div>
        </div>
        """

        # ── Tabela de resultados
        rows = ""
        for r in self.results:
            gv_str = f"{r['gap_v']:+d}"  if r['gap_v'] is not None else "—"
            gd_str = f"{r['gap_d']:+.1f}%" if r['gap_d'] is not None else "—"
            gd_cls = _gap_class(r['gap_d'])
            bk = r['best_known']
            bk_str = f"{bk[0]}v / {bk[1]:.2f}" if bk else "—"
            badge = '<span class="badge badge-ok">✓</span>' if r['feasible'] \
                    else '<span class="badge badge-bad">✗</span>'
            rows += f"""
            <tr>
              <td class="td-name">{html_lib.escape(r['name'])}</td>
              <td>{r['nv']}</td>
              <td class="{'td-ok' if r['gap_v'] is not None and r['gap_v'] <= 0 else 'td-warn'}">{gv_str}</td>
              <td>{r['td']:.4f}</td>
              <td class="{gd_cls}">{gd_str}</td>
              <td class="td-muted">{bk_str}</td>
              <td>{r['elapsed']:.0f}s</td>
              <td>{badge}</td>
            </tr>"""

        table_html = f"""
        <div class="section-title">Tabela de resultados</div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Instância</th>
                <th>Veículos</th>
                <th>Δv</th>
                <th>Distância</th>
                <th>Δdist%</th>
                <th>Benchmark</th>
                <th>Tempo</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """

        # ── Blocos por instância (gráficos)
        instance_blocks = ""
        for r in self.results:
            img_conv  = _plot_convergence_b64(r['history'], r['name'])
            img_routes = _plot_routes_b64(r['routes'], r['customers'], r['name'])

            conv_html  = f'<div class="chart-card"><img src="data:image/png;base64,{img_conv}" alt="convergência"></div>' \
                         if img_conv else ""
            route_html = f'<div class="chart-card"><img src="data:image/png;base64,{img_routes}" alt="rotas"></div>' \
                         if img_routes else ""

            gv = f"{r['gap_v']:+d}" if r['gap_v'] is not None else "—"
            gd = f"{r['gap_d']:+.1f}%" if r['gap_d'] is not None else "—"

            instance_blocks += f"""
            <div class="instance-section">
              <div class="inst-header">
                {html_lib.escape(r['name'])}
                <span>{r['nv']} veículos · {r['td']:.4f} · Δv={gv} · Δdist={gd} · {r['elapsed']:.0f}s</span>
              </div>
              <div class="instance-grid">
                {conv_html}
                {route_html}
              </div>
            </div>"""

        html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VRPTW — Resultados {self.algorithm}</title>
  {_HTML_STYLE}
</head>
<body>
  <h1>▸ VRPTW / {html_lib.escape(self.algorithm)}</h1>
  <div class="subtitle">
    {html_lib.escape(self.authors or "—")} &nbsp;·&nbsp; gerado em {ts}
  </div>

  {cards_html}
  {table_html}

  <div class="section-title">Análise por instância</div>
  {instance_blocks}
</body>
</html>"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  Relatório salvo: {filepath}")
        return filepath