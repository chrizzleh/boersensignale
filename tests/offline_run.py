"""Offline-Test mit synthetischen Kursen (keine Netzverbindung nötig)."""
import sys, shutil
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parents[1]))
import run_weekly as rw
from md import data, universe

rng = np.random.default_rng(1)
def fake_close(tickers, start, **kw):
    idx = pd.bdate_range(start, pd.Timestamp.today())
    out = {}
    for t in tickers:
        mu = rng.normal(0.0003, 0.0004); sig = rng.uniform(0.008, 0.03)
        base = 1.1 if t == "EURUSD=X" else (rng.uniform(20, 300))
        if t.startswith("EUR") and t.endswith("=X"): sig = 0.004; mu = 0
        s = base * np.exp(np.cumsum(rng.normal(mu, sig, len(idx))))
        # Crash-Phase einbauen
        s[(idx > "2020-02-15") & (idx < "2020-04-01")] *= np.linspace(1, .7, ((idx > "2020-02-15") & (idx < "2020-04-01")).sum())
        ser = pd.Series(s, index=idx)
        if rng.random() < 0.03: ser[:] = np.nan
        out[t] = ser
    return pd.DataFrame(out)
def fake_series(sid, start="2003-01-01"):
    idx = pd.bdate_range(start, pd.Timestamp.today())
    lvl = {"DGS10": 3, "DGS1": 2, "DCOILBRENTEU": 70, "DEXUSEU": 1.15}.get(sid, 2.0)
    return pd.Series(lvl + np.cumsum(rng.normal(0, 0.03, len(idx))), index=idx)
data.yahoo_close = fake_close
data.fred = fake_series
data.ecb = lambda k, start="2003-01-01": fake_series(k, start)
def fake_uni():
    rows = [{"ticker": k, "name": v, "index": "statisch", "tags": tag} for tag, d in universe.STATIC.items() for k, v in d.items()]
    return pd.DataFrame(rows)
universe.build_universe = fake_uni
for p in ["state", "output", "reports", "docs", "data/universe.csv"]:
    pp = rw.ROOT / p
    shutil.rmtree(pp, ignore_errors=True) if pp.is_dir() else pp.unlink(missing_ok=True)
cfg = rw.load_cfg()
if "--backtest" in sys.argv:
    rw.backtest(cfg)
else:
    out = rw.weekly_run(cfg)
    print("S1", out["system1"], "\nS2", out["system2"], "\nQuote", out["quota"])
    for k, d in out["depots"].items(): print(k, round(d["total"]), d["ytd"], len(d["positions"]), "Pos,", len(d["closed_this_year"]), "geschlossen")
