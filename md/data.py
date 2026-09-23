"""Datenbeschaffung: Yahoo Finance (Kurse), FRED (US-Zinsen), EZB (Euro-Zinsen)."""
from __future__ import annotations

import io
import logging
import time

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)
UA = {"User-Agent": "Mozilla/5.0 (musterdepot research script)"}


# ------------------------------------------------------------------ Yahoo
def yahoo_close(tickers: list[str], start: str, chunk: int = 150, retries: int = 3) -> pd.DataFrame:
    """Tägliche Schlusskurse (dividendenbereinigt) als DataFrame, Spalten = Ticker."""
    import yfinance as yf

    tickers = sorted(set(t for t in tickers if t))
    frames = []
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        for attempt in range(retries):
            try:
                df = yf.download(part, start=start, auto_adjust=True, progress=False,
                                 threads=True, group_by="column")
                if df is None or df.empty:
                    raise RuntimeError("leere Antwort")
                close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": part[0]})
                frames.append(close)
                break
            except Exception as e:  # noqa: BLE001
                log.warning("Yahoo-Chunk %s fehlgeschlagen (%s), Versuch %d", i, e, attempt + 1)
                time.sleep(5 * (attempt + 1))
        time.sleep(1)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def weekly(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Wochenschlusskurse (Freitag). Lücken innerhalb einer Woche werden mit dem letzten Kurs gefüllt."""
    return df.resample("W-FRI").last().ffill(limit=2)


# ------------------------------------------------------------------ FRED
def fred(series_id: str, start: str = "1990-01-01") -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    last = None
    for attempt in range(2):
        try:
            r = requests.get(url, headers=UA, timeout=25)
            r.raise_for_status()
            break
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3)
    else:
        raise last
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.to_numeric(df["value"], errors="coerce")
    s.index = df["date"]
    return s.dropna().rename(series_id)


# ------------------------------------------------------------------ EZB
def ecb(flow_key: str, start: str = "2000-01-01") -> pd.Series:
    """z. B. 'YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y'."""
    url = f"https://data-api.ecb.europa.eu/service/data/{flow_key}?format=csvdata&startPeriod={start}"
    r = requests.get(url, headers=UA, timeout=90)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    s.index = pd.to_datetime(df["TIME_PERIOD"])
    return s.dropna().sort_index().rename(flow_key.split("/")[-1])


def safe(fn, *args, default=None, **kw):
    try:
        return fn(*args, **kw)
    except Exception as e:  # noqa: BLE001
        log.warning("%s%s fehlgeschlagen: %s", fn.__name__, args, e)
        return default


# ------------------------------------------------------------------ Makro-Paket
def macro_bundle(start: str = "2003-01-01") -> dict[str, pd.Series]:
    """Alle Makroreihen als Tagesreihen. Mit Fallbacks, falls eine Quelle ausfällt."""
    out: dict[str, pd.Series] = {}
    out["us10"] = safe(fred, "DGS10", start)
    fred_ok = out["us10"] is not None  # FRED blockt gelegentlich Cloud-IPs -> dann nicht weiter warten
    out["us1"] = safe(fred, "DGS1", start) if fred_ok else None
    out["eu10"] = safe(ecb, "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y", start)
    out["eu1"] = safe(ecb, "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y", start)
    if out["eu10"] is None and fred_ok:  # Fallback: deutsche 10J-Rendite, monatlich
        out["eu10"] = safe(fred, "IRLTLT01DEM156N", start)
    if out["eu1"] is None and fred_ok:
        out["eu1"] = safe(fred, "IR3TIB01DEM156N", start)
    out["brent"] = safe(fred, "DCOILBRENTEU", start) if fred_ok else None
    out["eurusd"] = safe(fred, "DEXUSEU", start) if fred_ok else None
    return out


def yahoo_yield(s: pd.Series | None) -> pd.Series | None:
    """^TNX/^IRX kommen je nach Zeitraum als Prozent oder Prozent x 10."""
    if s is None or s.dropna().empty:
        return None
    s = s.dropna()
    return s.where(s < 25, s / 10)


def fill_macro_from_yahoo(macro: dict, mkt: pd.DataFrame) -> dict:
    """Ersatzquellen, wenn FRED nicht erreichbar ist."""
    alt = {"us10": yahoo_yield(mkt.get("^TNX")), "us1": yahoo_yield(mkt.get("^IRX")),
           "brent": mkt.get("BZ=F"), "eurusd": mkt.get("EURUSD=X")}
    for k, v in alt.items():
        if macro.get(k) is None and v is not None and not v.dropna().empty:
            macro[k] = v.dropna()
            log.info("Makro %s aus Yahoo ersetzt", k)
    return macro


def fx_to_eur(fx_close: pd.DataFrame, ccy: str) -> pd.Series | None:
    """Faktor, mit dem ein Kurs in Landeswährung multipliziert wird, um EUR zu erhalten."""
    if ccy == "EUR":
        return None
    col = f"EUR{ccy}=X"
    if col not in fx_close:
        return None
    return 1.0 / fx_close[col]


SUFFIX_CCY = {
    "": "USD", "DE": "EUR", "F": "EUR", "PA": "EUR", "AS": "EUR", "MI": "EUR", "MC": "EUR",
    "BR": "EUR", "VI": "EUR", "HE": "EUR", "LS": "EUR", "IR": "EUR", "SW": "CHF", "L": "GBP",
    "T": "JPY", "HK": "HKD", "AX": "AUD", "KS": "KRW", "TW": "TWD", "SI": "SGD", "ST": "SEK",
    "CO": "DKK", "OL": "NOK", "TO": "CAD", "SS": "CNY", "SZ": "CNY", "NS": "INR",
}


def currency_of(ticker: str) -> str:
    if ticker.startswith("^") or "=" in ticker:
        return "IDX"
    suf = ticker.rsplit(".", 1)[1] if "." in ticker else ""
    return SUFFIX_CCY.get(suf, "USD")


def prices_in_eur(close: pd.DataFrame, fx_close: pd.DataFrame) -> pd.DataFrame:
    """Rechnet alle Aktienkurse in EUR um (London notiert in Pence)."""
    out = {}
    fx = fx_close.reindex(close.index).ffill()
    for t in close.columns:
        ccy = currency_of(t)
        s = close[t]
        if ccy == "GBP":
            s = s / 100.0  # Pence -> Pfund
        if ccy in ("EUR", "IDX"):
            out[t] = s
            continue
        f = fx_to_eur(fx, ccy)
        out[t] = s * f if f is not None else s * np.nan
    return pd.DataFrame(out)


FX_TICKERS = [f"EUR{c}=X" for c in ["USD", "CHF", "GBP", "JPY", "HKD", "AUD", "KRW", "TWD",
                                     "SGD", "SEK", "DKK", "NOK", "CAD", "CNY", "INR"]]
