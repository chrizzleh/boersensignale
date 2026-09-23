"""Aktienuniversum: global inkl. Japan/Asien.

Primär werden die Indexlisten von Wikipedia gelesen (läuft in GitHub Actions), Ergebnis landet in
data/universe.csv. Fällt eine Quelle aus, greift die statische Liste unten.
Tags: large = Standardwerte (konservatives Depot), mid = Nebenwerte, tech = Technologie-Indizes.
"""
from __future__ import annotations

import io
import logging
import re

import pandas as pd
import requests

log = logging.getLogger(__name__)
UA = {"User-Agent": "Mozilla/5.0 (musterdepot research script)"}

# (Name, Wikipedia-Seite, mögliche Tickerspalten, Suffix, Tag, Mindestanzahl)
WIKI = [
    ("S&P 500", "List_of_S%26P_500_companies", ["Symbol"], "", "large", 400),
    ("Nasdaq-100", "Nasdaq-100", ["Ticker", "Symbol"], "", "tech", 80),
    ("DAX", "DAX", ["Ticker", "Symbol"], ".DE", "large", 30),
    ("MDAX", "MDAX", ["Symbol", "Ticker"], ".DE", "mid", 40),
    ("TecDAX", "TecDAX", ["Symbol", "Ticker"], ".DE", "tech", 20),
    ("SDAX", "SDAX", ["Symbol", "Ticker"], ".DE", "mid", 40),
    ("Euro Stoxx 50", "EURO_STOXX_50", ["Ticker", "Symbol"], "", "large", 40),
    ("SMI", "Swiss_Market_Index", ["Ticker", "Symbol"], ".SW", "large", 15),
    ("FTSE 100", "FTSE_100_Index", ["Ticker", "EPIC"], ".L", "large", 80),
    ("CAC 40", "CAC_40", ["Ticker", "Symbol"], ".PA", "large", 30),
    ("AEX", "AEX_index", ["Ticker symbol", "Ticker", "Symbol"], ".AS", "large", 20),
    ("Hang Seng", "Hang_Seng_Index", ["Ticker", "Stock code", "Code", "SEHK"], ".HK", "large", 50),
]

# Statische Ergänzung / Fallback: Japan, Asien-Pazifik, Österreich, DACH-Standardwerte, US-Schwergewichte
STATIC = {
    "large": {
        # Japan (Nikkei-Schwergewichte)
        "7203.T": "Toyota", "6758.T": "Sony", "8306.T": "Mitsubishi UFJ", "6861.T": "Keyence",
        "9984.T": "SoftBank Group", "6098.T": "Recruit", "8035.T": "Tokyo Electron", "9983.T": "Fast Retailing",
        "4063.T": "Shin-Etsu Chemical", "6501.T": "Hitachi", "8316.T": "Sumitomo Mitsui FG", "7974.T": "Nintendo",
        "4568.T": "Daiichi Sankyo", "6367.T": "Daikin", "8058.T": "Mitsubishi Corp", "8001.T": "Itochu",
        "8031.T": "Mitsui & Co", "9432.T": "NTT", "9433.T": "KDDI", "6902.T": "Denso", "7267.T": "Honda",
        "4502.T": "Takeda", "6981.T": "Murata", "6954.T": "Fanuc", "7741.T": "Hoya", "6273.T": "SMC",
        "4519.T": "Chugai Pharma", "8766.T": "Tokio Marine", "6594.T": "Nidec", "7011.T": "Mitsubishi Heavy",
        "6503.T": "Mitsubishi Electric", "6857.T": "Advantest", "6723.T": "Renesas", "4661.T": "Oriental Land",
        "7751.T": "Canon", "6752.T": "Panasonic", "5108.T": "Bridgestone", "2914.T": "Japan Tobacco",
        "4543.T": "Terumo", "6146.T": "Disco", "8411.T": "Mizuho FG", "7733.T": "Olympus", "6702.T": "Fujitsu",
        "6701.T": "NEC", "4901.T": "Fujifilm", "7270.T": "Subaru", "7201.T": "Nissan", "3382.T": "Seven & i",
        "9020.T": "JR East", "8802.T": "Mitsubishi Estate", "5401.T": "Nippon Steel", "6301.T": "Komatsu",
        # Asien-Pazifik
        "2330.TW": "TSMC", "2317.TW": "Hon Hai", "2454.TW": "MediaTek", "005930.KS": "Samsung Electronics",
        "000660.KS": "SK Hynix", "005380.KS": "Hyundai Motor", "035420.KS": "Naver", "051910.KS": "LG Chem",
        "D05.SI": "DBS Group", "O39.SI": "OCBC", "BHP.AX": "BHP", "CBA.AX": "Commonwealth Bank",
        "CSL.AX": "CSL", "RIO.AX": "Rio Tinto (AU)", "WES.AX": "Wesfarmers", "INFY": "Infosys ADR",
        "HDB": "HDFC Bank ADR", "IBN": "ICICI Bank ADR", "BABA": "Alibaba ADR", "PDD": "PDD Holdings ADR",
        "0700.HK": "Tencent", "9988.HK": "Alibaba HK", "1299.HK": "AIA Group", "0005.HK": "HSBC Holdings (HK)",
        "0941.HK": "China Mobile", "3690.HK": "Meituan", "1810.HK": "Xiaomi", "0388.HK": "HKEX", "1211.HK": "BYD",
        # Österreich (ATX)
        "EBS.VI": "Erste Group", "OMV.VI": "OMV", "VER.VI": "Verbund", "VOE.VI": "voestalpine",
        "RBI.VI": "Raiffeisen Bank Intl", "ANDR.VI": "Andritz", "WIE.VI": "Wienerberger", "BG.VI": "BAWAG",
        "SBO.VI": "Schoeller-Bleckmann", "POST.VI": "Österreichische Post", "VIG.VI": "Vienna Insurance",
        "UQA.VI": "Uniqa", "CAI.VI": "CA Immo", "LNZ.VI": "Lenzing", "EVN.VI": "EVN", "ATS.VI": "AT&S",
        # DAX
        "ADS.DE": "Adidas", "AIR.DE": "Airbus", "ALV.DE": "Allianz", "BAS.DE": "BASF", "BAYN.DE": "Bayer",
        "BEI.DE": "Beiersdorf", "BMW.DE": "BMW", "BNR.DE": "Brenntag", "CBK.DE": "Commerzbank",
        "CON.DE": "Continental", "1COV.DE": "Covestro", "DTG.DE": "Daimler Truck", "DBK.DE": "Deutsche Bank",
        "DB1.DE": "Deutsche Börse", "DHL.DE": "DHL Group", "DTE.DE": "Deutsche Telekom", "EOAN.DE": "E.ON",
        "FRE.DE": "Fresenius", "HNR1.DE": "Hannover Rück", "HEI.DE": "Heidelberg Materials",
        "HEN3.DE": "Henkel", "IFX.DE": "Infineon", "MBG.DE": "Mercedes-Benz", "MRK.DE": "Merck KGaA",
        "MTX.DE": "MTU Aero", "MUV2.DE": "Munich Re", "P911.DE": "Porsche AG", "PAH3.DE": "Porsche SE",
        "QIA.DE": "Qiagen", "RHM.DE": "Rheinmetall", "RWE.DE": "RWE", "SAP.DE": "SAP", "SRT3.DE": "Sartorius",
        "SIE.DE": "Siemens", "ENR.DE": "Siemens Energy", "SHL.DE": "Siemens Healthineers", "SY1.DE": "Symrise",
        "VOW3.DE": "Volkswagen", "VNA.DE": "Vonovia", "ZAL.DE": "Zalando", "G1A.DE": "GEA",
        # SMI
        "NESN.SW": "Nestlé", "NOVN.SW": "Novartis", "ROG.SW": "Roche", "UBSG.SW": "UBS", "ZURN.SW": "Zurich Insurance",
        "ABBN.SW": "ABB", "CFR.SW": "Richemont", "LONN.SW": "Lonza", "SIKA.SW": "Sika", "GIVN.SW": "Givaudan",
        "ALC.SW": "Alcon", "HOLN.SW": "Holcim", "PGHN.SW": "Partners Group", "SREN.SW": "Swiss Re",
        "SCMN.SW": "Swisscom", "GEBN.SW": "Geberit", "SLHN.SW": "Swiss Life", "KNIN.SW": "Kühne+Nagel",
        "LOGN.SW": "Logitech", "SOON.SW": "Sonova",
        # US-Schwergewichte (Fallback, falls S&P-Liste nicht geladen wird)
        "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon", "GOOGL": "Alphabet",
        "META": "Meta Platforms", "BRK-B": "Berkshire Hathaway", "LLY": "Eli Lilly", "AVGO": "Broadcom",
        "JPM": "JPMorgan", "V": "Visa", "XOM": "Exxon Mobil", "UNH": "UnitedHealth", "MA": "Mastercard",
        "COST": "Costco", "HD": "Home Depot", "PG": "Procter & Gamble", "JNJ": "Johnson & Johnson",
        "ORCL": "Oracle", "WMT": "Walmart", "NFLX": "Netflix", "BAC": "Bank of America", "CRM": "Salesforce",
        "ABBV": "AbbVie", "KO": "Coca-Cola", "CVX": "Chevron", "AMD": "AMD", "PEP": "PepsiCo", "MRK": "Merck & Co",
        "ADBE": "Adobe", "TMO": "Thermo Fisher", "CSCO": "Cisco", "ACN": "Accenture", "LIN": "Linde",
        "MCD": "McDonald's", "ABT": "Abbott", "GE": "GE Aerospace", "CAT": "Caterpillar", "IBM": "IBM",
        "BA": "Boeing", "GS": "Goldman Sachs", "ISRG": "Intuitive Surgical", "NOW": "ServiceNow",
        "INTU": "Intuit", "QCOM": "Qualcomm", "TXN": "Texas Instruments", "UBER": "Uber", "AMAT": "Applied Materials",
        "PLTR": "Palantir", "TSLA": "Tesla", "VLO": "Valero Energy",
    },
    "mid": {
        "HAG.DE": "Hensoldt", "RAA.DE": "Rational", "KGX.DE": "Kion", "LEG.DE": "LEG Immobilien",
        "EVK.DE": "Evonik", "TLX.DE": "Talanx", "HOT.DE": "Hochtief", "FRA.DE": "Fraport", "LHA.DE": "Lufthansa",
        "BOSS.DE": "Hugo Boss", "PUM.DE": "Puma", "KBX.DE": "Knorr-Bremse", "SDF.DE": "K+S", "AIXA.DE": "Aixtron",
        "EVT.DE": "Evotec", "NEM.DE": "Nemetschek", "BC8.DE": "Bechtle", "COK.DE": "Cancom", "SHA0.DE": "Schaeffler",
        "TKA.DE": "Thyssenkrupp", "JEN.DE": "Jenoptik", "S92.DE": "SMA Solar", "NDX1.DE": "Nordex",
        "WAF.DE": "Siltronic", "SMHN.DE": "Süss MicroTec", "RENK.DE": "Renk", "AOX.DE": "Aroundtown",
    },
    "tech": {},
}


def _read_tables(page: str) -> list[pd.DataFrame]:
    url = f"https://en.wikipedia.org/wiki/{page}"
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))


def _clean(sym: str, suffix: str, index: str) -> str | None:
    sym = str(sym).strip()
    if not sym or sym.lower() == "nan":
        return None
    sym = re.sub(r"\[.*?\]", "", sym).strip()
    if index == "Hang Seng":
        m = re.search(r"(\d{1,5})", sym)
        return f"{int(m.group(1)):04d}.HK" if m else None
    sym = sym.split(":")[-1].strip()
    sym = sym.replace(" ", "")
    if index in ("S&P 500", "Nasdaq-100"):
        return sym.replace(".", "-")
    if index == "FTSE 100":
        return sym.rstrip(".").replace(".", "-") + ".L"  # BT.A -> BT-A.L, RR. -> RR.L
    if "." in sym and index not in ("FTSE 100",):
        return sym  # hat schon einen Börsensuffix (z. B. Euro Stoxx 50)
    return sym + suffix


def scrape_index(index, page, cols, suffix, tag, min_rows) -> pd.DataFrame:
    tables = _read_tables(page)
    for t in tables:
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [" ".join(map(str, c)).strip() for c in t.columns]
        colmap = {str(c).strip(): c for c in t.columns}
        tcol = next((colmap[c] for c in cols if c in colmap), None)
        if tcol is None or len(t) < min_rows:
            continue
        ncol = next((colmap[c] for c in ("Security", "Company", "Name", "Constituent", "Company name")
                     if c in colmap), None)
        rows = []
        for _, r in t.iterrows():
            tk = _clean(r[tcol], suffix, index)
            if tk:
                rows.append({"ticker": tk, "name": str(r[ncol]) if ncol is not None else tk,
                             "index": index, "tag": tag})
        if len(rows) >= min_rows:
            return pd.DataFrame(rows)
    raise RuntimeError(f"keine passende Tabelle für {index}")


def build_universe() -> pd.DataFrame:
    parts = []
    for spec in WIKI:
        try:
            df = scrape_index(*spec)
            log.info("Universum %-14s %4d Titel", spec[0], len(df))
            parts.append(df)
        except Exception as e:  # noqa: BLE001
            log.warning("Universum %s nicht geladen: %s", spec[0], e)
    for tag, d in STATIC.items():
        parts.append(pd.DataFrame([{"ticker": k, "name": v, "index": "statisch", "tag": tag} for k, v in d.items()]))
    u = pd.concat(parts, ignore_index=True)
    # Ein Titel kann in mehreren Indizes sein: bevorzugt 'large', Namen aus Wikipedia behalten
    prio = {"large": 0, "tech": 1, "mid": 2}
    u["p"] = u["tag"].map(prio)
    u["from_static"] = (u["index"] == "statisch").astype(int)
    tags = u.groupby("ticker")["tag"].apply(lambda s: ",".join(sorted(set(s), key=prio.get)))
    u = u.sort_values(["ticker", "p", "from_static"]).drop_duplicates("ticker")
    u["tags"] = u["ticker"].map(tags)
    return u[["ticker", "name", "index", "tags"]].reset_index(drop=True)
