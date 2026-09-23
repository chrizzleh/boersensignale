"""Wochenreport als HTML (E-Mail, GitHub Pages) und PDF (Archiv)."""
from __future__ import annotations

import base64
import io
import logging
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Environment  # noqa: E402

log = logging.getLogger(__name__)
NAVY, GREEN, RED, GREY = "#14285a", "#1f7a3a", "#b3261e", "#6b7280"


def _pct(v, digits=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    return f"{v * 100:+.{digits}f} %".replace(".", ",")


def _num(v, digits=2):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    s = f"{v:,.{digits}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def _date(s):
    if not s:
        return "–"
    y, m, d = str(s)[:10].split("-")
    return f"{d}.{m}.{y}"


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chart_depots(rep: dict, idx_w: pd.DataFrame, cfg) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    year = rep["asof"][:4]
    colors = {"konservativ": NAVY, "spekulativ": "#c77d0a"}
    for name, d in rep["depots"].items():
        h = pd.DataFrame(d["history"])
        if h.empty:
            continue
        h["date"] = pd.to_datetime(h["date"])
        h = h[h["date"].dt.year == int(year)]
        if h.empty:
            continue
        base = cfg["start_capital_eur"]
        ax.plot(h["date"], (h["value"] / base - 1) * 100, lw=2.2, color=colors.get(name, GREY),
                label=f"Musterdepot {name}")
    for label, tk, c in [("DAX", "^GDAXI", "#9aa3b5"), ("S&P 500", "^GSPC", "#c9b7d9"), ("Nikkei 225", "^N225", "#b9d3c2")]:
        if tk in idx_w:
            s = idx_w[tk].loc[:rep["asof"]].dropna()
            prev = s[s.index.year < int(year)]
            s = s[s.index.year == int(year)]
            if len(prev) and len(s):
                ax.plot(s.index, (s / prev.iloc[-1] - 1) * 100, lw=1.2, color=c, label=label)
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_ylabel("% seit Jahresbeginn")
    ax.legend(fontsize=7, ncol=5, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    return _png(fig)


def chart_signals(rep: dict, idx_w: pd.DataFrame) -> str:
    h = pd.DataFrame(rep["signal_history"])
    h["date"] = pd.to_datetime(h["date"])
    h = h[h["date"] >= h["date"].max() - pd.Timedelta(days=365 * 5)]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    s = idx_w["ACWI"].reindex(h["date"]) if "ACWI" in idx_w else idx_w["^GSPC"].reindex(h["date"])
    ax.plot(h["date"], s.values, color=NAVY, lw=1.3)
    lo, hi = s.min(), s.max()
    for _, r in h.iterrows():
        c = GREEN if r["q"] >= 1 else ("#e0a526" if r["q"] > 0 else RED)
        ax.axvspan(r["date"] - pd.Timedelta(days=3.5), r["date"] + pd.Timedelta(days=3.5), color=c, alpha=0.13, lw=0)
    ax.set_ylim(lo * 0.95, hi * 1.03)
    ax.set_title("MSCI ACWI mit Aktienquote (grün 100 %, gelb 50 %, rot 0 %)", fontsize=8, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)


TEMPLATE = r"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Musterdepot {{ d(r.asof) }}</title>
<style>
 @page { size: A4; margin: 14mm 12mm; }
 body { font-family: "Segoe UI", Calibri, Arial, sans-serif; color:#1b1b1b; max-width: 860px; margin: 0 auto; padding: 12px; font-size: 13px; line-height: 1.45; background:#fff; }
 h1 { font-size: 22px; margin: 0; } h2 { font-size: 16px; margin: 22px 0 6px; color: {{ NAVY }}; border-bottom: 2px solid {{ NAVY }}; padding-bottom: 3px; }
 .head { display:flex; justify-content:space-between; align-items:baseline; border-bottom: 3px solid {{ NAVY }}; padding-bottom: 6px; }
 table { border-collapse: collapse; width: 100%; margin: 6px 0; font-size: 12px; }
 th { background: {{ NAVY }}; color: #fff; font-weight: 600; padding: 4px 6px; text-align: left; }
 td { padding: 3px 6px; border-bottom: 1px solid #e3e6ee; }
 td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
 .buy { color: {{ GREEN }}; font-weight: 700; } .sell { color: {{ RED }}; font-weight: 700; }
 .pos { color: {{ GREEN }}; } .neg { color: {{ RED }}; }
 .sig td { font-size: 14px; padding: 6px; } .muted { color: {{ GREY }}; font-size: 11px; }
 .box { background:#f3f5fa; border-left: 4px solid {{ NAVY }}; padding: 8px 12px; margin: 10px 0; }
 img { width: 100%; } .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 14px; }
 .pb { page-break-before: always; }
</style></head><body>
<div class="head"><h1>Musterdepot · Lang-Signale</h1><div>Ausgabe vom <b>{{ d(r.asof) }}</b></div></div>

<table class="sig">
<tr><th></th><th>aktuelles Signal</th><th>Signalwechsel am</th><th>vorhergehendes Signal</th><th class="n">Aktienanteil</th></tr>
{% for key, lab in [("system1","System 1 (technisch)"),("system2","System 2 (Makro)")] %}{% set s = r[key] %}
<tr><td>{{ lab }}</td><td class="{{ 'buy' if s.value=='Kauf' else 'sell' }}">{{ s.value }}</td><td>{{ d(s.since) }}</td>
<td>{{ d(s.previous_since) }} ({{ s.previous or '–' }})</td>{% if loop.first %}<td class="n" rowspan="2" style="font-size:20px"><b>{{ '%d' % (r.quota*100) }} %</b></td>{% endif %}</tr>
{% endfor %}</table>
{% if r.quota != r.quota_prev %}<div class="box"><b>Handlungsbedarf:</b> Die Aktienquote ändert sich von {{ '%d' % (r.quota_prev*100) }} % auf {{ '%d' % (r.quota*100) }} %.</div>
{% elif has_actions %}<div class="box"><b>Handlungsbedarf:</b> Depotumschichtungen, siehe Transaktionen.</div>
{% else %}<div class="box">Handlungsbedarf: <b>keiner</b>. Kontrollintervall wöchentlich.</div>{% endif %}

{% if r.commentary %}<h2>Marktkommentar</h2>{{ r.commentary }}{% endif %}

<h2>Indikatoren im Detail</h2>
<div class="grid"><div>
<table><tr><th>System 2</th><th>Signal</th><th>seit</th></tr>
{% for k in ["Index-Trend","Zinsstruktur","Makro 5"] %}{% set c = r.components[k] %}<tr><td><b>{{ k }}</b></td><td class="{{ 'buy' if c.value=='Kauf' else 'sell' }}">{{ c.value }}</td><td>{{ d(c.since) }}</td></tr>{% endfor %}
{% for k in ["Anleihen","Öl","Dollar","Rohstoffe","Saison"] %}{% set c = r.components[k] %}<tr><td>&nbsp;&nbsp;↳ {{ k }}</td><td class="{{ 'buy' if c.value=='Kauf' else 'sell' }}">{{ c.value }}</td><td>{{ d(c.since) }}</td></tr>{% endfor %}
</table></div><div>
<table><tr><th>System 1</th><th>Signal</th><th>seit</th></tr>
{% for k in ["Marktbreite","Leitindizes-Trend","Hoch-Tief","Volatilität","Momentum Welt"] %}{% set c = r.components[k] %}<tr><td>{{ k }}</td><td class="{{ 'buy' if c.value=='Kauf' else 'sell' }}">{{ c.value }}</td><td>{{ d(c.since) }}</td></tr>{% endfor %}
</table>
<table><tr><th>Frühwarnindex</th><th>Signal</th><th class="n">bis Verkaufsschwelle</th></tr>
{% for k, v in r.details.index_trend_dist.items() %}{% set c = r.components['IT ' + k] %}<tr><td>{{ k }}</td><td class="{{ 'buy' if c.value=='Kauf' else 'sell' }}">{{ c.value }}</td><td class="n">{{ p(v, 1) }}</td></tr>{% endfor %}
</table></div></div>
<p class="muted">Marktbreite: {{ '%d' % (r.details.breadth*100) }} % der beobachteten Indizes über der 40-Wochen-Linie ·
Neue 9-Monats-Hochs/-Tiefs (4 Wochen): {{ r.details.new_highs_4w }} / {{ r.details.new_lows_4w }} ·
Kombinierte Zinsstruktur (32-W-Schnitt): {{ n(r.details.yield_curve) }} %-Pkt. · VIX {{ n(r.details.vix,1) }} ·
Brent {{ n(r.details.brent,1) }} USD · EUR/USD {{ n(r.details.eurusd,4) }}</p>

<h2>Indizes</h2>
<table><tr><th>Index</th><th class="n">Stand</th><th class="n">Woche</th><th class="n">4 Wochen</th><th class="n">seit Jahresanfang</th><th class="n">12 Monate</th></tr>
{% for i in r.indices %}<tr><td>{{ i.name }}</td><td class="n">{{ n(i.last) }}</td>
{% for k in ["w1","w4","ytd","w52"] %}<td class="n {{ 'pos' if (i[k] or 0) >= 0 else 'neg' }}">{{ p(i[k]) }}</td>{% endfor %}</tr>{% endfor %}
</table><p class="muted">In Heimatwährung, Wochenschlusskurse.</p>
<img src="{{ chart_depots }}" alt="Depotentwicklung">

{% for name, dep in r.depots.items() %}
<h2>{{ dep.label or name }}</h2>
<table><tr><th>Aktie</th><th>Ticker</th><th class="n">RSL</th><th>Kaufdatum</th><th class="n">Kurs bei Kauf*</th><th class="n">Stück</th><th class="n">Kurs aktuell</th><th class="n">G/V {{ r.asof[:4] }}</th></tr>
{% for pz in dep.positions %}<tr><td>{{ pz.name }}</td><td>{{ pz.ticker }}</td><td class="n">{{ n(pz.rsl*100) }}</td><td>{{ d(pz.buy_date) }}</td>
<td class="n">{{ n(pz.year_start_price_eur,3) }} €</td><td class="n">{{ '%d' % pz.shares }}</td><td class="n">{{ n(pz.last_price_eur,3) }} €</td>
<td class="n {{ 'pos' if pz.gain_ytd>=0 else 'neg' }}">{{ p(pz.gain_ytd) }}</td></tr>{% else %}<tr><td colspan="8" class="muted">keine Positionen – Depot vollständig in Cash</td></tr>{% endfor %}
<tr><td colspan="6"><b>Barbestand</b></td><td class="n" colspan="2">{{ n(dep.cash) }} €</td></tr>
<tr><td colspan="6"><b>Aktueller Depotwert</b></td><td class="n" colspan="2">{{ n(dep.invested) }} €</td></tr>
<tr><td colspan="6"><b>Gesamtvermögen</b></td><td class="n" colspan="2"><b>{{ n(dep.total) }} €</b></td></tr>
<tr><td colspan="6"><b>Gewinn/Verlust seit 01.01.{{ r.asof[:4] }}</b></td><td class="n {{ 'pos' if dep.ytd>=0 else 'neg' }}" colspan="2"><b>{{ p(dep.ytd) }}</b></td></tr>
{% for y, v in dep.yearly_perf.items()|reverse %}<tr><td colspan="6">Performance in {{ y }}</td><td class="n" colspan="2">{{ p(v) }}</td></tr>{% endfor %}
</table>
<p class="muted">* Kursanpassung zu Jahresbeginn bei allen Aktien, die vor {{ r.asof[:4] }} gekauft wurden. Mit {{ n(cap,0) }} € wird das Depot zu Jahresbeginn neu gestartet. RSL = Relative Stärke nach Levy × 100.</p>
{% set acts = r.actions[name] %}
<h3 style="font-size:14px;margin:10px 0 4px">Aktuelle Depottransaktionen</h3>
{% if acts %}<table><tr><th>Aktie</th><th>Ticker</th><th class="n">Kurs</th><th>Aktion</th><th>Limit</th><th class="n">Stück</th><th>Begründung</th></tr>
{% for a in acts %}<tr><td>{{ a.name }}</td><td>{{ a.ticker }}</td><td class="n">{{ n(a.price_eur,3) }} €</td>
<td class="{{ 'buy' if a.action=='Kauf' else 'sell' }}">{{ a.action }}</td><td>Market</td><td class="n">{{ '%d' % a.shares }}</td><td class="muted">{{ a.reason }}{% if a.gain is not none %} · G/V {{ p(a.gain) }}{% endif %}</td></tr>{% endfor %}</table>
<p class="muted">Kurse = Wochenschlusskurse. Order zu Wochenbeginn an der liquidesten Börse (meist Heimatbörse), bei größeren Orders limitieren.</p>
{% else %}<p class="muted">Keine Transaktionen in dieser Woche.</p>{% endif %}
{% if dep.closed_this_year %}<details><summary class="muted">Abgeschlossene Transaktionen {{ r.asof[:4] }} ({{ dep.closed_this_year|length }})</summary>
<table><tr><th>Aktie</th><th class="n">Kauf</th><th class="n">Kurs</th><th class="n">Verkauf</th><th class="n">Kurs</th><th class="n">G/V</th></tr>
{% for c in dep.closed_this_year|reverse %}<tr><td>{{ c.name }}</td><td class="n">{{ d(c.buy_date) }}</td><td class="n">{{ n(c.buy_price_eur,3) }} €</td><td class="n">{{ d(c.sell_date) }}</td><td class="n">{{ n(c.sell_price_eur,3) }} €</td><td class="n {{ 'pos' if c.gain>=0 else 'neg' }}">{{ p(c.gain) }}</td></tr>{% endfor %}
</table></details>{% endif %}
{% endfor %}

<h2>Relative Stärke – Spitzengruppe</h2>
<div class="grid"><div><table><tr><th>Standardwerte</th><th class="n">RSL</th></tr>
{% for t in r.rsl_top_large %}<tr><td>{{ t.name }} <span class="muted">{{ t.ticker }}</span></td><td class="n">{{ n(t.rsl*100) }}</td></tr>{% endfor %}</table></div>
<div><table><tr><th>Gesamtuniversum</th><th class="n">RSL</th></tr>
{% for t in r.rsl_top[:15] %}<tr><td>{{ t.name }} <span class="muted">{{ t.ticker }}</span></td><td class="n">{{ n(t.rsl*100) }}</td></tr>{% endfor %}</table></div></div>
<p class="muted">Universum: {{ r.universe_size }} Aktien mit ausreichender Kurshistorie.</p>
<img src="{{ chart_signals }}" alt="Signalhistorie">

<p class="muted" style="margin-top:18px">Automatisch erzeugt am {{ r.generated }} auf Basis öffentlich beschriebener Regeln von Uwe Lang (Relative Stärke nach Levy,
Index-Trend, Zinsstruktur, Makro-Indikatoren). Eigenständiger Nachbau, keine Verbindung zu Swissinvest/Börsensignale. Keine Anlageberatung.</p>
</body></html>"""


def render_html(rep: dict, idx_w: pd.DataFrame, cfg, commentary_html: str | None = None) -> str:
    env = Environment(autoescape=False)
    tpl = env.from_string(TEMPLATE)
    rep = dict(rep)
    if commentary_html:
        rep["commentary"] = commentary_html
    has_actions = any(rep["actions"].get(k) for k in rep["actions"])
    return tpl.render(r=rep, d=_date, p=_pct, n=_num, cap=cfg["start_capital_eur"], has_actions=has_actions,
                      chart_depots=chart_depots(rep, idx_w, cfg), chart_signals=chart_signals(rep, idx_w),
                      NAVY=NAVY, GREEN=GREEN, RED=RED, GREY=GREY)


def to_pdf(html: str, path: Path) -> Path | None:
    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(path))
        return path
    except Exception as e:  # noqa: BLE001
        log.warning("PDF nicht erzeugt: %s", e)
        return None


def send_mail(rep: dict, html: str, pdf: Path | None) -> None:
    """Versand über SMTP, wenn die Secrets MAIL_USER, MAIL_PASSWORD, MAIL_TO gesetzt sind."""
    user, pw, to = os.getenv("MAIL_USER"), os.getenv("MAIL_PASSWORD"), os.getenv("MAIL_TO")
    if not (user and pw and to):
        log.info("Kein E-Mail-Versand (MAIL_USER/MAIL_PASSWORD/MAIL_TO nicht gesetzt)")
        return
    q = int(rep["quota"] * 100)
    n_act = sum(len(v) for v in rep["actions"].values())
    msg = EmailMessage()
    msg["Subject"] = (f"Musterdepot {_date(rep['asof'])}: S1 {rep['system1']['value']}, S2 {rep['system2']['value']}, "
                      f"Quote {q} %" + (f", {n_act} Transaktion(en)" if n_act else ""))
    msg["From"], msg["To"] = user, to
    msg.set_content("Der Wochenreport ist als HTML und PDF beigefügt.")
    msg.add_alternative(html, subtype="html")
    if pdf and Path(pdf).exists():
        msg.add_attachment(Path(pdf).read_bytes(), maintype="application", subtype="pdf",
                           filename=Path(pdf).name)
    host = os.getenv("MAIL_HOST", "smtp.gmail.com")
    with smtplib.SMTP_SSL(host, int(os.getenv("MAIL_PORT", "465"))) as s:
        s.login(user, pw)
        s.send_message(msg)
    log.info("E-Mail an %s verschickt", to)
