"""Wöchentlicher Lauf: Daten laden, Signale rechnen, Depots fortschreiben, Report erzeugen.

python run_weekly.py              # normaler Wochenlauf
python run_weekly.py --backtest   # Signalhistorie ab 2005 + Abgleich mit den Originalsignalen
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from md import data, portfolio, report, signals, universe

ROOT = Path(__file__).parent
STATE, OUT, REPORTS = ROOT / "state", ROOT / "output", ROOT / "reports"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
log = logging.getLogger("musterdepot")
for _n in ("fontTools", "weasyprint", "yfinance", "matplotlib"):
    logging.getLogger(_n).setLevel(logging.WARNING)


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def market_tickers(cfg) -> list[str]:
    t = set(cfg["index_trend"]["indices"].values()) | set(cfg["system1"]["breadth_indices"].values())
    t |= set(cfg["system1"]["trend_indices"]) | set(cfg["report_indices"].values())
    t |= {cfg["system1"]["vix_ticker"], cfg["system1"]["momentum_ticker"], cfg["commodities"]["ticker"],
          cfg["commodities"]["fallback"], "BZ=F", "EURUSD=X", "^TNX", "^IRX"}
    return sorted(t)


def get_universe(force: bool = False) -> pd.DataFrame:
    path = ROOT / "data" / "universe.csv"
    if path.exists() and not force:
        age = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).days
        u = pd.read_csv(path)
        if age < 30 and len(u) > 300:
            return u
    u = universe.build_universe()
    path.parent.mkdir(exist_ok=True)
    u.to_csv(path, index=False)
    log.info("Universum neu aufgebaut: %d Titel", len(u))
    return u


def last_complete_friday(today: date) -> pd.Timestamp:
    # Samstag/Sonntag: der gerade vergangene Freitag. Unter der Woche: Freitag der Vorwoche.
    d = today - timedelta(days=(today.weekday() - 4) % 7)
    return pd.Timestamp(d)


def load_all(cfg, start_market="2003-01-01", start_stocks=None, force_universe=False):
    today = date.today()
    start_stocks = start_stocks or str(today - timedelta(days=int(365 * 4.2)))
    uni = get_universe(force_universe)
    log.info("Lade %d Markt- und %d Devisenreihen …", len(market_tickers(cfg)), len(data.FX_TICKERS))
    mkt = data.yahoo_close(market_tickers(cfg) + data.FX_TICKERS, start_market)
    log.info("Lade %d Aktien ab %s …", len(uni), start_stocks)
    stk = data.yahoo_close(uni["ticker"].tolist(), start_stocks)
    missing = sorted(set(uni["ticker"]) - set(stk.columns[stk.notna().any()]))
    log.info("Aktien mit Daten: %d, ohne Daten: %d (%s …)", stk.shape[1] - len(missing), len(missing), missing[:15])
    macro = data.fill_macro_from_yahoo(data.macro_bundle(start_market), mkt)
    for k, v in macro.items():
        log.info("Makro %-7s %s", k, "FEHLT" if v is None else f"{len(v)} Werte bis {v.index[-1].date()}")
    return uni, mkt, stk, macro


def prepare(cfg, uni, mkt, stk, cutoff):
    fx = mkt[[c for c in data.FX_TICKERS if c in mkt]]
    idx_w = data.weekly(mkt.drop(columns=[c for c in fx.columns if c != "EURUSD=X"])).loc[:cutoff]
    stk_eur = data.prices_in_eur(stk, fx)
    stk_eur_w = data.weekly(stk_eur).loc[:cutoff]
    stk_loc_w = data.weekly(stk).loc[:cutoff]
    # Aktien auf den Index-Kalender bringen
    stk_eur_w = stk_eur_w.reindex(idx_w.index)
    stk_loc_w = stk_loc_w.reindex(idx_w.index)
    return idx_w, stk_eur_w, stk_loc_w


def perf_table(idx_w: pd.DataFrame, cfg, asof) -> list[dict]:
    rows = []
    for name, tk in cfg["report_indices"].items():
        if tk not in idx_w:
            continue
        s = idx_w[tk].loc[:asof].dropna()
        if len(s) < 60:
            continue
        ytd_base = s[s.index.year < asof.year]
        f = lambda n: float(s.iloc[-1] / s.iloc[-1 - n] - 1)  # noqa: E731
        rows.append({"name": name, "last": float(s.iloc[-1]), "w1": f(1), "w4": f(4), "w52": f(52),
                     "ytd": float(s.iloc[-1] / ytd_base.iloc[-1] - 1) if len(ytd_base) else None})
    return rows


def signal_meta(sig: pd.Series) -> dict:
    sw = signals.switch_dates(sig)
    cur = sig.dropna().iloc[-1] if sig.notna().any() else np.nan
    lab = lambda v: "Kauf" if v == 1 else ("Verkauf" if v == -1 else "–")  # noqa: E731
    return {"value": lab(cur), "since": str(sw[-1][0].date()) if sw else None,
            "previous": lab(sw[-2][1]) if len(sw) > 1 else None,
            "previous_since": str(sw[-2][0].date()) if len(sw) > 1 else None}


def run_depots(cfg, sig_df, uni, stk_eur_w, stk_loc_w, asof, rsl, vol):
    """Depots aus dem gespeicherten Zustand bis `asof` fortschreiben (holt verpasste Wochen nach)."""
    STATE.mkdir(exist_ok=True)
    cap = cfg["start_capital_eur"]
    results, week_actions = {}, {}
    for name, dcfg in cfg["depots"].items():
        path = STATE / f"depot_{name}.json"
        if path.exists():
            depot = json.loads(path.read_text(encoding="utf-8"))
            start = pd.Timestamp(depot["history"][-1]["date"]) + pd.Timedelta(days=1) if depot["history"] else None
        else:  # Erststart: ab Jahresbeginn simulieren, damit das Depot sofort einen Verlauf hat
            first = sig_df.loc[str(asof.year)].index[0]
            depot = portfolio.new_depot(name, cap, first)
            start = first
        weeks = sig_df.loc[start:asof].index if start is not None else []
        acts = []
        for d in weeks:
            q = float(sig_df.at[d, "Aktienquote"]) if pd.notna(sig_df.at[d, "Aktienquote"]) else 0.0
            i = sig_df.index.get_loc(d)
            pq = float(sig_df["Aktienquote"].iloc[i - 1]) if i > 0 else q
            tab = portfolio.ranking(d, rsl, vol, stk_eur_w, uni, cfg["rsl"]["min_history_weeks"],
                                    cfg["rsl"]["min_price_eur"])
            a = portfolio.step(depot, d, tab, stk_eur_w.loc[d], stk_loc_w.loc[d], q, pq, dcfg, cap)
            acts.extend(a)
        depot.setdefault("transactions", []).extend(acts)
        path.write_text(json.dumps(depot, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        tab = portfolio.ranking(asof, rsl, vol, stk_eur_w, uni, cfg["rsl"]["min_history_weeks"],
                                cfg["rsl"]["min_price_eur"])
        results[name] = {**portfolio.snapshot(depot, tab), "history": depot["history"],
                         "closed_this_year": [c for c in depot["closed"] if c["sell_date"][:4] == str(asof.year)]}
        week_actions[name] = [a for a in depot["transactions"] if a["date"] == str(asof.date())]
    return results, week_actions, tab


def weekly_run(cfg):
    asof = last_complete_friday(date.today())
    uni, mkt, stk, macro = load_all(cfg)
    idx_w, stk_eur_w, stk_loc_w = prepare(cfg, uni, mkt, stk, asof)
    asof = idx_w.index[-1]
    res = signals.compute_all(idx_w, stk_eur_w, macro, cfg)
    sig, det = res["signals"], res["details"]
    rsl = portfolio.rsl_panel(stk_eur_w, cfg["rsl"]["weeks"])
    vol = portfolio.vol_panel(stk_eur_w)
    depots, actions, tab = run_depots(cfg, sig, uni, stk_eur_w, stk_loc_w, asof, rsl, vol)

    comp = {c: signal_meta(sig[c]) for c in sig.columns if c not in ("Aktienquote",)}
    top = tab.head(25).reset_index().rename(columns={"index": "ticker"})
    top_large = tab[tab["tags"].fillna("").str.contains("large")].head(15).reset_index().rename(columns={"index": "ticker"})
    it_dist = det["index_trend_dist"].loc[asof].to_dict()
    out = {
        "asof": str(asof.date()),
        "generated": pd.Timestamp.now(tz="Europe/Berlin").isoformat(timespec="minutes"),
        "system1": comp["System 1"], "system2": comp["System 2"],
        "quota": float(sig["Aktienquote"].iloc[-1]),
        "quota_prev": float(sig["Aktienquote"].iloc[-2]),
        "components": comp,
        "details": {
            "breadth": float(det["breadth"].loc[asof]), "trend_share": float(det["trend_share"].loc[asof]),
            "new_highs_4w": int(det["highs"].loc[asof]), "new_lows_4w": int(det["lows"].loc[asof]),
            "yield_curve": float(det["yield_curve"].loc[asof]), "index_trend_dist": it_dist,
            "vix": float(det["vix"].loc[asof]), "brent": float(det["brent"].loc[asof]),
            "eurusd": float(det["eurusd"].loc[asof]),
        },
        "indices": perf_table(idx_w, cfg, asof),
        "depots": depots, "actions": actions,
        "rsl_top": top[["ticker", "name", "rsl", "pct", "price_eur", "tags"]].to_dict("records"),
        "rsl_top_large": top_large[["ticker", "name", "rsl", "pct", "price_eur", "tags"]].to_dict("records"),
        "signal_history": [{"date": str(d.date()), "s1": v1, "s2": v2, "q": q}
                           for d, v1, v2, q in zip(sig.index, sig["System 1"], sig["System 2"], sig["Aktienquote"])
                           if d.year >= asof.year - 10 and pd.notna(v2)],
        "universe_size": int(tab.shape[0]),
    }
    OUT.mkdir(exist_ok=True)
    js = json.dumps(out, ensure_ascii=False, indent=1, default=lambda o: None if o is None or (isinstance(o, float) and np.isnan(o)) else float(o))
    (OUT / "latest.json").write_text(js, encoding="utf-8")
    sig.to_csv(OUT / "signals_history.csv", float_format="%.3f")
    REPORTS.mkdir(exist_ok=True)
    html = report.render_html(json.loads(js), idx_w, cfg)
    (REPORTS / f"{out['asof']}.html").write_text(html, encoding="utf-8")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "index.html").write_text(html, encoding="utf-8")
    pdf = report.to_pdf(html, REPORTS / f"{out['asof']}.pdf")
    log.info("Report geschrieben: %s (PDF: %s)", out["asof"], bool(pdf))
    report.send_mail(json.loads(js), html, pdf)
    return out


def backtest(cfg):
    uni, mkt, stk, macro = load_all(cfg, start_market="2000-01-01", start_stocks="2012-01-01")
    asof = last_complete_friday(date.today())
    idx_w, stk_eur_w, _ = prepare(cfg, uni, mkt, stk, asof)
    res = signals.compute_all(idx_w, stk_eur_w, macro, cfg)
    sig = res["signals"]
    OUT.mkdir(exist_ok=True)
    sig.to_csv(OUT / "backtest_signals.csv", float_format="%.3f")
    lines = ["# Backtest der Timing-Signale", ""]
    lab = {1.0: "Kauf", -1.0: "Verkauf"}
    for col in ["System 1", "System 2", "Index-Trend", "Zinsstruktur", "Makro 5", "Anleihen", "Öl", "Dollar",
                "Rohstoffe", "Marktbreite", "Hoch-Tief"]:
        sw = [(str(d.date()), lab.get(v, "?")) for d, v in signals.switch_dates(sig[col]) if d.year >= 2014]
        lines.append(f"## {col} ({len(sw)} Wechsel seit 2014)")
        lines.append(", ".join(f"{d} {v}" for d, v in sw))
        lines.append("")
    lines.append("## Referenz aus dem Original")
    for k, v in cfg["reference_signals"].items():
        lines.append(f"- {k}: " + ", ".join(f"{d} {s}" for d, s in v))
    # Wie wäre ein MSCI-Welt-Anleger mit der Aktienquote gefahren?
    w = idx_w.get("ACWI")
    if w is not None:
        r = w.pct_change().fillna(0)
        q = sig["Aktienquote"].shift(1).fillna(0)
        for start in ("2010", "2015", "2020"):
            rr, qq = r.loc[start:], q.loc[start:]
            bh = (1 + rr).prod() - 1
            tm = (1 + rr * qq).prod() - 1
            dd = lambda x: float(((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min())  # noqa: E731
            lines.append(f"- ACWI ab {start}: Kaufen+Halten {bh:+.1%} (max. DD {dd(rr):.1%}), "
                         f"mit Aktienquote {tm:+.1%} (max. DD {dd(rr * qq):.1%}), Ø Quote {qq.mean():.0%}")
    (OUT / "backtest.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--refresh-universe", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg()
    if a.refresh_universe:
        get_universe(force=True)
    if a.backtest:
        backtest(cfg)
    else:
        weekly_run(cfg)
