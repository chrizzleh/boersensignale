# Musterdepot nach Uwe-Lang-Regeln

Eigenständiger Nachbau des Börsensignale-Musterdepots auf Basis öffentlich beschriebener Regeln.
Läuft jeden Samstagmorgen automatisch in GitHub Actions, rechnet die Signale auf Freitagsschlusskursen,
führt zwei Musterdepots und erzeugt einen Wochenreport (HTML + PDF, optional per E-Mail).

Keine Verbindung zu Swissinvest/Börsensignale. Keine Anlageberatung.

## Was berechnet wird

**System 2 (Makro, langsam)**: Mehrheit aus drei Stimmen
- Index-Trend: Nasdaq Composite, Dow Jones Utility, DAX. Je Index Verkauf bei neuem 26-Wochen-Tief, Kauf bei neuem 26-Wochen-Hoch. Mehrheit 2 von 3.
- Zinsstruktur: (US 10J − US 1J + Euro 10J − Euro 1J) / 2, geglättet über 32 Wochen. Positiv = Kauf.
- Makro 5 (Mehrheit): Anleihen (39-Wochen-Tief/Hoch der 10J-Renditen), Öl (Brent 5-W-Tief Kauf / 6-W-Hoch Verkauf),
  Dollar (15-W-Hoch/Tief ggü. Euro), Rohstoffe (S&P GSCI statt CRB), Saison (Mai–September Verkauf).

**System 1 (technisch, schnell)**: Mehrheit aus Marktbreite (Indizes über 40-Wochen-Linie), Trend der Leitindizes,
Hoch-Tief-Signal (neue 9-Monats-Hochs vs. -Tiefs im Universum), Volatilität (VIX) und Welt-Momentum (ACWI).
Wechsel erst nach zwei bestätigenden Wochen.

**Aktienquote**: beide Kauf = 100 %, eines = 50 %, keines = 0 %. Neukäufe nur bei 100 %.

**Aktienauswahl**: Relative Stärke nach Levy (Kurs ÷ Schnitt der letzten 27 Wochenschlüsse) über ein globales Universum
(S&P 500, Nasdaq-100, DAX/MDAX/SDAX/TecDAX, Euro Stoxx 50, SMI, ATX, FTSE 100, CAC 40, AEX, Hang Seng, Nikkei, Asien-Pazifik).
Verkauf, wenn ein Wert aus der Spitzengruppe fällt oder unter seinen 27-Wochen-Schnitt rutscht.

Alle Schwellen stehen in `config.yaml`.

## Einrichtung

1. Dateien ins Repo hochladen (inkl. Ordner `.github`).
2. Repo → **Actions** → Workflows aktivieren, falls GitHub fragt.
3. Der erste Lauf startet durch den Upload. Er baut das Universum, rechnet den Backtest und simuliert die Depots ab Jahresbeginn.
4. Optional E-Mail: Repo → Settings → Secrets and variables → Actions → *New repository secret*
   - `MAIL_USER` = deine Gmail-Adresse
   - `MAIL_PASSWORD` = Gmail-App-Passwort (nicht dein normales Passwort)
   - `MAIL_TO` = Empfängeradresse
5. Optional Webseite: Settings → Pages → Branch `main`, Ordner `/docs`. Dann liegt der aktuelle Report unter
   `https://<name>.github.io/<repo>/`.

## Dateien

| Pfad | Inhalt |
|---|---|
| `output/latest.json` | alle Zahlen der aktuellen Woche |
| `output/run_log.txt` | Protokoll des letzten Laufs (bei Fehlern hier schauen) |
| `output/backtest.md` | Signalwechsel seit 2014 und Abgleich mit dem Original |
| `reports/JJJJ-MM-TT.pdf` | Wochenreport-Archiv |
| `state/depot_*.json` | Depotbestände und Historie |
