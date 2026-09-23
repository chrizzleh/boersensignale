"""Aktienauswahl nach Relativer Stärke (Levy) und Führung der Musterdepots."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ Relative Stärke
def rsl_panel(px_eur_w: pd.DataFrame, weeks: int) -> pd.DataFrame:
    return px_eur_w / px_eur_w.rolling(weeks, min_periods=weeks).mean()


def vol_panel(px_eur_w: pd.DataFrame, weeks: int = 26) -> pd.DataFrame:
    r = np.log(px_eur_w).diff()
    return r.rolling(weeks, min_periods=weeks // 2).std() * math.sqrt(52)


def ranking(date, rsl: pd.DataFrame, vol: pd.DataFrame, px_eur: pd.DataFrame, universe: pd.DataFrame,
            min_hist: int, min_price: float) -> pd.DataFrame:
    """Rangliste zu einem Stichtag. pct = Perzentil in der RSL-Rangliste (1,0 = stärkster Wert)."""
    hist_ok = px_eur.loc[:date].notna().sum() >= min_hist
    r = rsl.loc[date]
    fresh = px_eur.loc[:date].tail(3).notna().any()  # Kurs in den letzten 3 Wochen vorhanden
    ok = r.notna() & hist_ok.reindex(r.index, fill_value=False) & fresh.reindex(r.index, fill_value=False)
    ok &= px_eur.loc[date].reindex(r.index) >= min_price
    t = pd.DataFrame({"rsl": r[ok], "vol": vol.loc[date].reindex(r[ok].index),
                      "price_eur": px_eur.loc[date].reindex(r[ok].index)})
    t["pct"] = t["rsl"].rank(pct=True)
    t = t.join(universe.set_index("ticker")[["name", "tags", "index"]], how="left")
    return t.sort_values("rsl", ascending=False)


# ------------------------------------------------------------------ Depot
def new_depot(name: str, capital: float, date: pd.Timestamp, initial: list | None = None,
              prices_eur: pd.Series | None = None, year_reset: bool = True) -> dict:
    """Neues Depot. `initial` = bereits vorhandene Positionen (z. B. eigene Aktien)."""
    positions = []
    for ip in [x for x in (initial or []) if float(x.get("shares") or 0) > 0]:
        px_now = float(prices_eur.get(ip["ticker"], np.nan)) if prices_eur is not None else np.nan
        buy = float(ip.get("buy_price_eur") or px_now)
        positions.append({"ticker": ip["ticker"], "name": ip.get("name", ip["ticker"]), "shares": float(ip["shares"]),
                          "buy_date": str(ip.get("buy_date") or date.date()), "buy_price_eur": buy,
                          "year_start_price_eur": px_now if np.isfinite(px_now) else buy,
                          "last_price_eur": px_now if np.isfinite(px_now) else buy, "buy_rsl": None,
                          "fees_eur": 0.0, "initial": True})
    start_value = capital + sum(p["shares"] * p["last_price_eur"] for p in positions)
    return {"name": name, "cash": capital, "positions": positions, "year": int(date.year),
            "year_start_value": start_value if not year_reset else capital, "yearly_perf": {}, "history": [],
            "closed": [], "last_sold": {}, "start_date": str(date.date()), "capital": capital,
            "start_value": start_value, "year_reset": year_reset, "realized_eur": 0.0, "fees_eur": 0.0}


def _value(depot, prices: pd.Series) -> float:
    v = depot["cash"]
    for p in depot["positions"]:
        px = prices.get(p["ticker"], np.nan)
        if not np.isfinite(px):
            px = p.get("last_price_eur", p["buy_price_eur"])
        p["last_price_eur"] = float(px)
        v += p["shares"] * px
    return v


def _year_roll(depot, date, prices, capital):
    """Neustart mit 100.000 EUR zu Jahresbeginn, Positionen werden anteilig mitskaliert."""
    if int(date.year) == depot["year"]:
        return
    value = _value(depot, prices)
    if not depot.get("year_reset", True):  # echtes Depot: kein Neustart, nur Jahresbilanz festhalten
        depot["yearly_perf"][str(depot["year"])] = value / depot["year_start_value"] - 1
        for p in depot["positions"]:
            p["year_start_price_eur"] = p["last_price_eur"]
        depot["year"], depot["year_start_value"] = int(date.year), value
        return
    depot["yearly_perf"][str(depot["year"])] = value / depot["year_start_value"] - 1
    f = capital / value
    depot["cash"] *= f
    for p in depot["positions"]:
        p["shares"] *= f
        p["year_start_price_eur"] = p["last_price_eur"]
    depot["year"], depot["year_start_value"] = int(date.year), capital


def step(depot: dict, date: pd.Timestamp, table: pd.DataFrame, prices_eur: pd.Series,
         prices_local: pd.Series, quota: float, prev_quota: float, dcfg: dict, capital: float) -> list[dict]:
    """Eine Wochenentscheidung. Gibt die Transaktionen der Woche zurück."""
    _year_roll(depot, date, prices_eur, capital)
    total = _value(depot, prices_eur)
    actions = []
    fee = float(dcfg.get("fee_eur", 0.0))

    def sell(p, reason, shares=None):
        px = p["last_price_eur"]
        n = p["shares"] if shares is None else shares
        depot["cash"] += n * px - fee
        pnl = n * (px - p["buy_price_eur"]) - fee - p.get("fees_eur", 0.0) * (n / p["shares"])
        depot["realized_eur"] = depot.get("realized_eur", 0.0) + pnl
        depot["fees_eur"] = depot.get("fees_eur", 0.0) + fee
        actions.append({"date": str(date.date()), "action": "Verkauf", "ticker": p["ticker"], "name": p["name"],
                        "shares": round(n, 2), "price_eur": round(px, 3),
                        "price_local": float(prices_local.get(p["ticker"], np.nan)), "reason": reason,
                        "gain": px / p["buy_price_eur"] - 1, "pnl_eur": round(pnl, 2)})
        if shares is None or shares >= p["shares"]:
            depot["closed"].append({**{k: p.get(k) for k in ("ticker", "name", "buy_date", "buy_price_eur", "buy_rsl")},
                                    "shares": round(p["shares"], 2), "sell_date": str(date.date()),
                                    "sell_price_eur": round(px, 3), "gain": px / p["buy_price_eur"] - 1,
                                    "pnl_eur": round(pnl, 2), "reason": reason})
            depot["positions"].remove(p)
            depot["last_sold"][p["ticker"]] = str(date.date())
        else:
            p["shares"] -= n

    # 1) Verkaufsregeln je Position
    for p in list(depot["positions"]):
        row = table.loc[p["ticker"]] if p["ticker"] in table.index else None
        if quota == 0:
            sell(p, "Beide Systeme auf Verkauf")
        elif row is None:
            sell(p, "keine Kursdaten mehr")
        elif row["rsl"] < dcfg["sell_rsl_below"]:
            sell(p, f"RSL {row['rsl']:.2f} unter {dcfg['sell_rsl_below']:.2f}")
        elif row["pct"] < dcfg["sell_below_pct"] and not (
                p.get("initial") and dcfg.get("initial_sell_rule", "trend") == "trend"):
            sell(p, f"aus der Spitzengruppe gefallen (Rang-Perzentil {row['pct']:.0%})")

    # 2) Aktienquote reduzieren: schwächste Werte zuerst
    slots = int(round(dcfg["max_positions"] * quota))
    if len(depot["positions"]) > slots:
        weak = sorted(depot["positions"], key=lambda p: table["rsl"].get(p["ticker"], 0))
        for p in weak[: len(depot["positions"]) - slots]:
            sell(p, f"Aktienquote auf {quota:.0%} gesenkt")

    # 3) Käufe nur bei vollem Kaufsignal (beide Systeme), wie im Original
    if quota >= 1.0 and len(depot["positions"]) < slots:
        tags = set(dcfg["universe_tags"])
        cand = table[table["tags"].fillna("").apply(lambda s: bool(tags & set(s.split(","))))]
        vol_cut = table["vol"].quantile(dcfg["max_vol_pct"]) if dcfg["max_vol_pct"] < 1 else np.inf
        cand = cand[(cand["rsl"] >= dcfg["min_rsl"]) & (cand["pct"] >= 1 - dcfg["buy_top_pct"])
                    & (cand["vol"] <= vol_cut)]
        held = {p["ticker"] for p in depot["positions"]}
        held_names = {str(p["name"]).split()[0].lower() for p in depot["positions"]}
        recent = {t for t, d in depot["last_sold"].items() if (date - pd.Timestamp(d)).days < 28}
        size = total / dcfg["max_positions"]
        for tk, r in cand.iterrows():
            if len(depot["positions"]) >= slots:
                break
            first = str(r["name"]).split()[0].lower()
            if tk in held or tk in recent or first in held_names:
                continue
            px = r["price_eur"]
            n = math.floor(min(size, depot["cash"] - fee) / px)
            if n < 1:
                continue
            depot["cash"] -= n * px + fee
            depot["fees_eur"] = depot.get("fees_eur", 0.0) + fee
            depot["positions"].append({"ticker": tk, "name": r["name"], "shares": n, "buy_date": str(date.date()),
                                       "buy_price_eur": float(px), "year_start_price_eur": float(px),
                                       "last_price_eur": float(px), "buy_rsl": float(r["rsl"]), "fees_eur": fee})
            held_names.add(first)
            actions.append({"date": str(date.date()), "action": "Kauf", "ticker": tk, "name": r["name"],
                            "shares": n, "price_eur": round(float(px), 3),
                            "price_local": float(prices_local.get(tk, np.nan)),
                            "reason": f"RSL {r['rsl']:.2f}, Rang-Perzentil {r['pct']:.1%}", "gain": None,
                            "amount_eur": round(n * float(px), 2)})

    value = _value(depot, prices_eur)
    depot["history"].append({"date": str(date.date()), "value": round(value, 2), "cash": round(depot["cash"], 2),
                             "quota": quota})
    return actions


def _status(p, table, dcfg):
    """Halten / Beobachten: wie nah ist die Position an einer Verkaufsregel?"""
    if dcfg is None or p["ticker"] not in table.index:
        return "Halten"
    r = table.loc[p["ticker"]]
    if r["rsl"] < dcfg["sell_rsl_below"] + 0.03 or r["pct"] < dcfg["sell_below_pct"] + 0.08:
        return "Beobachten"
    return "Halten"


def snapshot(depot: dict, table: pd.DataFrame, dcfg: dict | None = None) -> dict:
    invested = sum(p["shares"] * p["last_price_eur"] for p in depot["positions"])
    total = depot["cash"] + invested
    pos = []
    for p in sorted(depot["positions"], key=lambda p: -table["rsl"].get(p["ticker"], 0)):
        pos.append({**p, "rsl": float(table["rsl"].get(p["ticker"], np.nan)),
                    "pct": float(table["pct"].get(p["ticker"], np.nan)),
                    "gain_total": p["last_price_eur"] / p["buy_price_eur"] - 1,
                    "gain_ytd": p["last_price_eur"] / p["year_start_price_eur"] - 1,
                    "value": p["shares"] * p["last_price_eur"],
                    "pnl_eur": p["shares"] * (p["last_price_eur"] - p["buy_price_eur"]),
                    "status": _status(p, table, dcfg)})
    unrealized = sum(x["pnl_eur"] for x in pos)
    start_value = depot.get("start_value", depot.get("capital", depot["year_start_value"]))
    return {"cash": depot["cash"], "invested": invested, "total": total,
            "ytd": total / depot["year_start_value"] - 1, "positions": pos,
            "yearly_perf": depot["yearly_perf"], "equity_share": invested / total if total else 0,
            "realized_eur": depot.get("realized_eur", 0.0), "unrealized_eur": unrealized,
            "fees_eur": depot.get("fees_eur", 0.0), "start_value": start_value,
            "start_date": depot.get("start_date"), "since_start": total / start_value - 1 if start_value else None,
            "year_reset": depot.get("year_reset", True)}
