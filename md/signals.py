"""Markt-Timing nach Uwe Lang: Einzelindikatoren, System 1 (technisch, schnell), System 2 (Makro, langsam).

Alle Signale sind Wochenreihen mit +1 = Kauf, -1 = Verkauf, NaN = noch nicht bestimmbar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ Bausteine
def hold_state(buy: pd.Series, sell: pd.Series) -> pd.Series:
    """Zustandsautomat: bleibt im letzten Signal, bis das Gegensignal kommt."""
    state, out = np.nan, []
    for b, s in zip(buy.fillna(False), sell.fillna(False)):
        if b and not s:
            state = 1.0
        elif s and not b:
            state = -1.0
        out.append(state)
    return pd.Series(out, index=buy.index)


def majority(frame: pd.DataFrame, prev_hold: bool = True) -> pd.Series:
    """Mehrheitsentscheid; bei Gleichstand bleibt das vorherige Signal."""
    tot = frame.fillna(0).sum(axis=1)
    raw = np.sign(tot).replace(0, np.nan)
    raw[frame.notna().sum(axis=1) == 0] = np.nan
    return raw.ffill() if prev_hold else raw


def confirm(raw: pd.Series, weeks: int) -> pd.Series:
    """Signalwechsel erst, wenn das neue Signal `weeks` Wochen in Folge anliegt."""
    state, streak, out = np.nan, 0, []
    for v in raw:
        if np.isnan(v):
            out.append(state)
            continue
        if np.isnan(state):
            state = v
        elif v != state:
            streak += 1
            if streak >= weeks:
                state, streak = v, 0
        else:
            streak = 0
        out.append(state)
    return pd.Series(out, index=raw.index)


# ------------------------------------------------------------------ System 2
def index_trend(idx_w: pd.DataFrame, cfg) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Frühwarnindizes: je Index Verkauf bei neuem 26-Wochen-Tief, Kauf bei neuem 26-Wochen-Hoch.
    Gesamtsignal = Mehrheit der drei."""
    per, dist = {}, {}
    for name, tk in cfg["indices"].items():
        s = idx_w[tk].dropna()
        low = s.shift(1).rolling(cfg["sell_low_weeks"]).min()
        high = s.shift(1).rolling(cfg["buy_high_weeks"]).max()
        per[name] = hold_state(s > high, s < low).reindex(idx_w.index).ffill()
        dist[name] = (low / s - 1).reindex(idx_w.index)  # nötiger Rückgang bis zur Verkaufsschwelle
    per = pd.DataFrame(per)
    votes = per.fillna(0).sum(axis=1)
    need = cfg["votes_needed"]
    buy = (per == 1).sum(axis=1) >= need
    sell = (per == -1).sum(axis=1) >= need
    total = hold_state(buy, sell)
    return total, per, pd.DataFrame(dist)


def yield_curve(us10, us1, eu10, eu1, idx, cfg) -> tuple[pd.Series, pd.Series]:
    w = lambda s: s.resample("W-FRI").last().ffill().reindex(idx).ffill()  # noqa: E731
    spreads = []
    if us10 is not None and us1 is not None:
        spreads.append(w(us10) - w(us1))
    if eu10 is not None and eu1 is not None:
        spreads.append(w(eu10) - w(eu1))
    comb = pd.concat(spreads, axis=1).mean(axis=1)
    smooth = comb.rolling(cfg["smooth_weeks"], min_periods=cfg["smooth_weeks"] // 2).mean()
    sig = pd.Series(np.where(smooth > cfg["threshold"], 1.0, -1.0), index=idx)
    sig[smooth.isna()] = np.nan
    return sig, smooth


def bonds(us10, eu10, idx, cfg) -> pd.Series:
    """Kauf, wenn US- oder Euro-Rendite ein 39-Wochen-Tief erreicht.
    Verkauf, wenn beide seit dem letzten Kauf ein 39-Wochen-Hoch erreicht haben."""
    n = cfg["window_weeks"]
    w = lambda s: s.resample("W-FRI").last().ffill().reindex(idx).ffill()  # noqa: E731
    a, b = w(us10), w(eu10)
    lo = lambda s: s <= s.shift(1).rolling(n).min()  # noqa: E731
    hi = lambda s: s >= s.shift(1).rolling(n).max()  # noqa: E731
    la, lb, ha, hb = lo(a), lo(b), hi(a), hi(b)
    state, fa, fb, out = np.nan, False, False, []
    for i in range(len(idx)):
        if la.iloc[i] or lb.iloc[i]:
            state, fa, fb = 1.0, False, False
        else:
            fa |= bool(ha.iloc[i])
            fb |= bool(hb.iloc[i])
            if fa and fb:
                state, fa, fb = -1.0, False, False
        out.append(state)
    return pd.Series(out, index=idx)


def oil(brent_w: pd.Series, cfg) -> pd.Series:
    """Brent 5-Wochen-Tief -> Kauf (billige Energie), 6-Wochen-Hoch -> Verkauf."""
    s = brent_w
    buy = s <= s.shift(1).rolling(cfg["buy_low_weeks"]).min()
    sell = s >= s.shift(1).rolling(cfg["sell_high_weeks"]).max()
    return hold_state(buy, sell)


def dollar(eurusd_w: pd.Series, cfg) -> pd.Series:
    """US-Dollar 15-Wochen-Hoch ggü. Euro (EUR/USD 15-Wochen-Tief) -> Kauf; umgekehrt Verkauf."""
    n, s = cfg["window_weeks"], eurusd_w
    buy = s <= s.shift(1).rolling(n).min()
    sell = s >= s.shift(1).rolling(n).max()
    return hold_state(buy, sell)


def commodities(cmd_w: pd.Series) -> pd.Series:
    """Verkauf, wenn Rohstoffe über Vorjahr liegen UND der Jahresanstieg größer ist als vor einem Jahr."""
    yoy = cmd_w / cmd_w.shift(52) - 1
    sell = (yoy > 0) & (yoy > yoy.shift(52))
    sig = pd.Series(np.where(sell, -1.0, 1.0), index=cmd_w.index)
    sig[yoy.shift(52).isna()] = np.nan
    return sig


def season(idx: pd.DatetimeIndex, cfg) -> pd.Series:
    return pd.Series(np.where(idx.month.isin(cfg["weak_months"]), -1.0, 1.0), index=idx)


# ------------------------------------------------------------------ System 1
def breadth(idx_w: pd.DataFrame, tickers: list[str], ma: int) -> pd.Series:
    cols = [t for t in tickers if t in idx_w]
    sub = idx_w[cols]
    above = sub > sub.rolling(ma, min_periods=int(ma * 0.8)).mean()
    valid = sub.rolling(ma, min_periods=int(ma * 0.8)).mean().notna()
    return above.sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)


def hilo(stocks_w: pd.DataFrame, n: int, sum_w: int) -> tuple[pd.Series, pd.Series]:
    prev_max = stocks_w.shift(1).rolling(n, min_periods=n).max()
    prev_min = stocks_w.shift(1).rolling(n, min_periods=n).min()
    highs = (stocks_w > prev_max).sum(axis=1)
    lows = (stocks_w < prev_min).sum(axis=1)
    return highs.rolling(sum_w).sum(), lows.rolling(sum_w).sum()


def compute_all(idx_w: pd.DataFrame, stocks_w: pd.DataFrame, macro: dict, cfg: dict) -> dict:
    """Berechnet alle Indikatoren und beide Systeme. Liefert DataFrame + Zusatzinfos für den Report."""
    ix = idx_w.index
    w = lambda s: None if s is None else s.resample("W-FRI").last().ffill().reindex(ix).ffill()  # noqa: E731

    it, it_per, it_dist = index_trend(idx_w, cfg["index_trend"])
    yc, yc_val = yield_curve(macro.get("us10"), macro.get("us1"), macro.get("eu10"), macro.get("eu1"), ix, cfg["yield_curve"])
    bd = bonds(macro["us10"], macro["eu10"], ix, cfg["bonds"])
    brent = w(macro.get("brent")) if macro.get("brent") is not None else idx_w.get("BZ=F")
    if "BZ=F" in idx_w:  # aktuellere Yahoo-Daten bevorzugen, FRED für die Historie
        brent = idx_w["BZ=F"].combine_first(brent) if brent is not None else idx_w["BZ=F"]
    ol = oil(brent, cfg["oil"])
    eurusd = idx_w["EURUSD=X"].combine_first(w(macro.get("eurusd"))) if macro.get("eurusd") is not None else idx_w["EURUSD=X"]
    dl = dollar(eurusd, cfg["dollar"])
    cmd = idx_w.get(cfg["commodities"]["ticker"])
    if cmd is None or cmd.dropna().empty:
        cmd = idx_w.get(cfg["commodities"]["fallback"])
    cm = commodities(cmd)
    ss = season(ix, cfg["season"])
    macro5 = pd.DataFrame({"Anleihen": bd, "Öl": ol, "Dollar": dl, "Rohstoffe": cm, "Saison": ss})
    m5 = majority(macro5)
    sys2_votes = pd.DataFrame({"Index-Trend": it, "Zinsstruktur": yc, "Makro 5": m5})
    sys2 = majority(sys2_votes)

    s1 = cfg["system1"]
    br = breadth(idx_w, list(s1["breadth_indices"].values()), s1["breadth_ma_weeks"])
    br_sig = pd.Series(np.where(br > s1["breadth_threshold"], 1.0, -1.0), index=ix)
    lead = idx_w[[t for t in s1["trend_indices"] if t in idx_w]]
    tr = (lead > lead.rolling(s1["trend_ma_weeks"]).mean()).sum(axis=1) / lead.shape[1]
    tr_sig = pd.Series(np.where(tr > 0.5, 1.0, np.where(tr < 0.5, -1.0, np.nan)), index=ix).ffill()
    hi_n, lo_n = hilo(stocks_w, s1["hilo_weeks"], s1["hilo_sum_weeks"])
    hl_sig = pd.Series(np.where(hi_n > lo_n, 1.0, -1.0), index=ix)
    hl_sig[(hi_n + lo_n) == 0] = np.nan
    vix = idx_w.get(s1["vix_ticker"])
    vx_sig = pd.Series(np.where(vix < vix.rolling(s1["vix_ma_weeks"]).mean(), 1.0, -1.0), index=ix)
    mom = idx_w.get(s1["momentum_ticker"])
    mo_sig = pd.Series(np.where(mom / mom.shift(s1["momentum_weeks"]) - 1 > 0, 1.0, -1.0), index=ix)
    sys1_votes = pd.DataFrame({"Marktbreite": br_sig, "Leitindizes-Trend": tr_sig, "Hoch-Tief": hl_sig,
                               "Volatilität": vx_sig, "Momentum Welt": mo_sig})
    sys1_raw = majority(sys1_votes, prev_hold=True)
    sys1 = confirm(sys1_raw, s1["confirm_weeks"])

    q = cfg["equity_quota"]
    n_buy = (sys1 == 1).astype(int) + (sys2 == 1).astype(int)
    quota = n_buy.map({2: q["both_buy"], 1: q["one_buy"], 0: q["none"]})

    df = pd.concat([sys1.rename("System 1"), sys2.rename("System 2"), quota.rename("Aktienquote"),
                    sys1_votes, sys2_votes, macro5, it_per.add_prefix("IT ")], axis=1)
    return {
        "signals": df,
        "details": {
            "breadth": br, "trend_share": tr, "highs": hi_n, "lows": lo_n, "yield_curve": yc_val,
            "index_trend_dist": it_dist, "vix": vix, "brent": brent, "eurusd": eurusd,
        },
    }


def switch_dates(sig: pd.Series) -> list[tuple[pd.Timestamp, float]]:
    s = sig.dropna()
    chg = s[s != s.shift(1)]
    return list(chg.items())
