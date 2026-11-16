"""
Previsao da seca do Rio Negro em Manaus - jul-dez/2024
Hackathon Ocean (Samsung Ocean / UEA) - Equipe Outliers
Integrantes: Hugo Carvalho de Moraes, Gabriel Assis da Silva e Juan Victor Sequeira

Script unico: recebe o dataset de treino (cota diaria ate 30/06/2024) e os dados externos,
e gera o dataset predito de cota de 01/07 a 31/12/2024 e os dois graficos.

COMO EXECUTAR (Python 3.10+):
    pip install -r requirements.txt
    python previsao_rio_negro_2024.py

Opcoes:
    --treino      CSV de treino com colunas data,cota            (padrao: dados/train.csv)
    --externos    pasta com os dados externos (noaa/ e ana/)      (padrao: dados/externos)
    --saida       pasta onde gravar CSV, graficos e resumo        (padrao: saida)
    --observado   CSV data,cota com a cota observada de jul-dez/2024, usado SO para desenhar
                  o observado nos graficos e calcular a Nota Final (padrao: dados/avaliacao/...)
    --sem-observado   ignora o arquivo observado (graficos so com a previsao)

SAIDAS (na pasta --saida):
    previsao_cota_2024.csv     184 linhas, colunas data,cota (01/07 a 31/12/2024)
    grafico1_2024.png          ano de 2024 inteiro: observado, previsto, corte em 30/06/2024
    grafico2_vazante.png       zoom set-nov/2024: previsto com incerteza, 09/10/2024, linhas de 17 m e 14 m
    resumo_execucao.txt        backtest, probabilidade de cota < 14 m em outubro e Nota Final

MODELO (resumo): a variacao da cota desde 30/jun e prevista como a curva media dos ultimos 15 anos,
deslocada por tres regressoes Ridge sobre 2 componentes principais das curvas (clima; clima + rios do sul;
Manaus + ENSO + Atlantico). Previsao = media das tres + cota de 30/06/2024. Treino 1973-2023.
Nenhuma informacao posterior a 30/06/2024 entra no modelo. Semente fixa; o resultado e deterministico.
"""
import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# =============================================================================================
# 1. Constantes
# =============================================================================================
CUTOFF = pd.Timestamp("2024-06-30")                     # ultima data de informacao permitida
H_DAYS = [d.strftime("%m-%d") for d in pd.date_range("2023-07-01", "2023-12-31")]   # 184 dias
ON_MASK = np.array([d[:2] in ("10", "11") for d in H_DAYS])                          # out-nov
OCT_MASK = np.array([d[:2] == "10" for d in H_DAYS])                                 # outubro
SENT = [-99.99, -99.9, -999, -999.0, -999.9, -9.99]     # valores sentinela = ausente
STATIONS = {"labrea": ("13870000", "purus"), "humaita": ("15630000", "madeira")}
WIN = 400                                               # dias antes da origem usados no tratamento

# Variaveis e membros do modelo
MAN = ["man__manaus__raw__c0", "man__manaus__raw__slp30", "man__manaus__padall__amj"]
ENSO = ["noaa__nino34__anom__last", "noaa__nino12__anom__last", "noaa__nino4__anom__last", "noaa__soi__anom__last"]
ATL = ["noaa__natl__anom__jfm", "noaa__grad_atl__anom__jfm", "noaa__tna__anom__last"]
FW4 = ["noaa__natl__anom__last", "noaa__nino12__anom__tr6", "ana__labrea__pad30__lvl10", "ana__humaita__pad30__pk"]
MEMBROS = {"clima": (FW4[:2], 1.0), "clima+rios": (FW4, 3.0), "a_priori": (MAN + ENSO + ATL, 3.0)}
START, NC, CENTER = 1973, 2, 15                         # inicio do treino, componentes, anos da curva-base
ANOS_BT = list(range(1995, 2024))                       # anos do backtest


# =============================================================================================
# 2. Serie alvo (train.csv)
# =============================================================================================
def load_train(path):
    df = pd.read_csv(path, parse_dates=["data"])
    df = df[df["data"] <= CUTOFF]
    assert df["data"].max() <= CUTOFF
    return df


def year_matrix(df):
    """Matriz ano x dia (MM-DD) com a cota."""
    t = df.assign(ano=df["data"].dt.year, md=df["data"].dt.strftime("%m-%d"))
    return t.pivot_table(index="ano", columns="md", values="cota")


def delta_matrix(piv):
    """Variacao da cota em relacao a 30/jun, para os anos com o semestre completo (ano x 184)."""
    full = piv.dropna(subset=["06-30"] + H_DAYS)
    return full[H_DAYS].sub(full["06-30"], axis=0)


# =============================================================================================
# 3. Estacoes da ANA: tratamento causal (so olha para tras a partir de cada origem)
# =============================================================================================
def read_station(raw_dir, label):
    """Leitura, uma leitura por dia (prefere consistido; mediana) e calendario diario. Serie em cm."""
    code = STATIONS[label][0]
    df = pd.read_csv(Path(raw_dir) / "ana" / f"cota_{code}_{label}.csv", parse_dates=["data"])
    df["cota_cm"] = pd.to_numeric(df["cota_cm"], errors="coerce")
    df = df[df["data"] <= CUTOFF].dropna(subset=["cota_cm"])
    assert df["data"].max() <= CUTOFF
    amp = df.groupby("data")["cota_cm"].agg(lambda s: s.max() - s.min())
    susp = amp.index[amp > 20]                                        # leituras duplicadas discordantes
    best = df.groupby("data")["nivel_consistencia"].transform("max")
    df = df[df["nivel_consistencia"] == best]
    s = df.groupby("data")["cota_cm"].median()
    s = s.reindex(pd.date_range(s.index.min(), CUTOFF, freq="D"))
    flag = pd.Series(False, index=s.index)
    flag[flag.index.isin(susp)] = True
    return s, flag


def treat(x, susp, L, lo, hi, G=7):
    """Limpeza de um trecho que TERMINA na origem. x em cm -> metros.
    Remove faixa impossivel e picos isolados, corta em degrau de regua, interpola falhas <= G dias."""
    x = np.asarray(x, float).copy()
    susp = np.asarray(susp, bool)
    x[(x < lo) | (x > hi)] = np.nan
    d1 = np.diff(x, prepend=np.nan)
    d2 = np.diff(x, append=np.nan)
    spike = (np.abs(d1) > L) & (np.abs(d2) > L) & (np.sign(d1) != np.sign(d2))
    sus = susp & ((np.abs(d1) > L) | (np.abs(d2) > L))
    x[spike | sus] = np.nan
    d = np.abs(np.diff(x, prepend=np.nan))
    steps = np.where(d > L)[0]
    if len(steps):
        x[: steps[-1]] = np.nan                                       # degrau: usa so o trecho recente
    s = pd.Series(x)
    if G > 0:
        s = s.interpolate(method="linear", limit=G, limit_area="inside")
        isn = pd.Series(x).isna()
        run = isn.groupby((~isn).cumsum()).transform("sum")
        s[isn & (run > G)] = np.nan                                   # nao preenche falha longa
    s = s.ffill(limit=3) if s.iloc[-1] != s.iloc[-1] else s           # so no fim, sem dado a direita
    return s.values / 100.0


def _mean(a, frac=0.6):
    return np.nan if np.isfinite(a).mean() < frac else float(np.nanmean(a))


def origin_stats(v):
    """Estatisticas na origem. v: vetor em metros que termina em 30/jun (v[-1] = origem)."""
    r = v[::-1]
    out = {}
    for k in (1, 5, 10, 30):
        out[f"lvl{k}"] = _mean(r[:k])
    for j in (7, 15, 30, 60, 90):
        a, b = r[:3], r[j:j + 3]
        out[f"slp{j}"] = (np.nanmean(a) - np.nanmean(b)) if np.isfinite(a).any() and np.isfinite(b).any() else np.nan
    out["amj"] = _mean(r[:91], 0.8)
    seg = r[:181]
    if np.isfinite(seg).mean() >= 0.8:
        out["pk"] = float(np.nanmax(seg))
        out["dpk"] = float(np.nanargmax(seg))
    else:
        out["pk"] = out["dpk"] = np.nan
    return out


def station_table(raw_dir, label, last_year=2024):
    """Tabela ano x estatisticas de uma estacao, com o tratamento refeito a cada origem (causal)."""
    s, flag = read_station(raw_dir, label)
    vals, fl, idx = s.values, flag.values, s.index
    ad = np.abs(np.diff(vals, prepend=np.nan))
    rows = {}
    for y in range(idx[0].year + 1, last_year + 1):
        t0 = pd.Timestamp(year=y, month=6, day=30)
        if t0 > idx[-1]:
            break
        e = idx.get_loc(t0) + 1
        h, hd = vals[:e], ad[:e]
        if np.isfinite(h[-WIN:]).sum() < 30 or np.isfinite(hd).sum() < 365:
            continue
        L = 5 * np.nanpercentile(hd, 99)                              # limite de salto, so com o passado
        lo, hi = np.nanpercentile(h, 0.1) - 300, np.nanpercentile(h, 99.9) + 300
        rows[y] = origin_stats(treat(vals[max(0, e - WIN):e], fl[max(0, e - WIN):e], L, lo, hi))
    return pd.DataFrame(rows).T


def causal_anoms(t, prefix, min_prior=8):
    """Anomalias com media e desvio calculados SO com anos anteriores (base: todos / ultimos 30)."""
    out = {}
    for c in t.columns:
        col = t[c]
        if c.startswith("slp") or c == "dpk":
            out[f"{prefix}__raw__{c}"] = col
            continue
        for base, nb in (("all", 10 ** 6), ("30", 30)):
            a, p, q = {}, {}, {}
            for y, v in col.items():
                prior = col[(col.index < y) & (col.index >= y - nb)].dropna()
                if v != v or len(prior) < min_prior:
                    continue
                a[y] = v - prior.mean()
                p[y] = a[y] / (prior.std() + 1e-9)
                q[y] = float((prior < v).mean())
            out[f"{prefix}__abs{base}__{c}"] = pd.Series(a, dtype=float)
            out[f"{prefix}__pad{base}__{c}"] = pd.Series(p, dtype=float)
            out[f"{prefix}__pct{base}__{c}"] = pd.Series(q, dtype=float)
    return pd.DataFrame(out)


def manaus_table(train):
    """Variaveis da propria serie de Manaus em cada 30/jun."""
    s = train.set_index("data")["cota"]
    rows = {}
    for y in range(1903, s.index.max().year + 1):
        t0 = pd.Timestamp(year=y, month=6, day=30)
        if t0 > s.index.max():
            break
        r = origin_stats(s[t0 - pd.Timedelta(days=WIN - 1):t0].values)
        r["c0"] = float(s[t0])
        prev = s[pd.Timestamp(year=y - 1, month=7, day=1):pd.Timestamp(year=y - 1, month=12, day=31)]
        r["minprev"] = float(prev.min()) if len(prev) > 150 else np.nan
        r["subida"] = r["c0"] - r["minprev"] if r["minprev"] == r["minprev"] else np.nan
        rows[y] = r
    t = pd.DataFrame(rows).T
    f = causal_anoms(t.drop(columns=["c0", "minprev", "subida"]), "man__manaus")
    for c in ("c0", "minprev", "subida", "pk"):
        f[f"man__manaus__raw__{c}"] = t[c]
    return f


# =============================================================================================
# 4. Indices climaticos da NOAA, com o atraso de publicacao emulado
# =============================================================================================
def _monthly(df):
    datas = pd.to_datetime(pd.DataFrame({"year": df["y"].astype(int).values, "month": df["m"].astype(int).values, "day": 1}))
    return pd.Series(df["v"].values, index=pd.PeriodIndex(datas.dt.to_period("M")))


def read_indices(raw_dir):
    """Devolve dict nome -> (serie mensal de anomalias, ultimo mes publicado em 30/jun de cada ano)."""
    nd = Path(raw_dir) / "noaa"
    out = {}
    s = pd.read_csv(nd / "sstoi.indices", sep=r"\s+", header=0,
                    names=["y", "m", "n12", "a12", "n3", "a3", "n4", "a4", "n34", "a34"])
    assert (s["y"].iloc[-1], s["m"].iloc[-1]) == (2024, 5), "sstoi: esperado ultimo mes 2024-05"
    for k in ("12", "3", "4", "34"):                                  # Nino: ultimo = maio
        out[f"nino{k}"] = (_monthly(pd.DataFrame({"y": s["y"], "m": s["m"], "v": s[f"a{k}"]})), 5)
    a = pd.read_csv(nd / "sstoi.atl.indices", sep=r"\s+", header=0,
                    names=["y", "m", "natl", "anatl", "satl", "asatl", "trop", "atrop"])
    for k in ("natl", "satl", "trop"):                                # Atlantico: ultimo = marco
        out[k] = (_monthly(pd.DataFrame({"y": a["y"], "m": a["m"], "v": a[f"a{k}"]})), 3)
    out["grad_atl"] = (out["natl"][0] - out["satl"][0], 3)
    rows, blk = [], 0                                                 # SOI padronizado (2o bloco): ultimo = maio
    for ln in (nd / "soi_cpc.txt").read_text().splitlines():
        if ln.startswith("YEAR"):
            blk += 1
            continue
        if blk == 2 and re.match(r"^\d{4}", ln):
            v = re.findall(r"-?\d+\.\d", ln[4:])
            if len(v) == 12:
                rows += [(int(ln[:4]), m + 1, float(x)) for m, x in enumerate(v)]
    so = pd.DataFrame(rows, columns=["y", "m", "v"])
    so.loc[so["v"].isin(SENT), "v"] = np.nan
    out["soi"] = (_monthly(so).dropna(), 5)
    rows = []                                                         # TNA: ultimo = dezembro do ano anterior
    for ln in (nd / "tna.data").read_text().splitlines()[1:]:
        p = ln.split()
        if len(p) == 13:
            try:
                rows += [(int(p[0]), m + 1, float(x)) for m, x in enumerate(p[1:])]
            except ValueError:
                pass
    tn = pd.DataFrame(rows, columns=["y", "m", "v"])
    tn.loc[tn["v"].isin(SENT), "v"] = np.nan
    out["tna"] = (_monthly(tn).dropna(), 0)
    for k, (ser, _) in out.items():
        assert ser.index.max() <= pd.Period("2024-06"), k
    return out


def index_features(raw_dir, years):
    """Variaveis dos indices para cada ano de origem, usando so meses publicados ate 30/jun daquele ano."""
    ind = read_indices(raw_dir)
    out = {}
    for y in years:
        r = {}
        for name, (ser, mlast) in ind.items():
            last = pd.Period(year=y, month=mlast, freq="M") if mlast else pd.Period(year=y - 1, month=12, freq="M")
            s = ser[ser.index <= last]
            if len(s) < 13 or s.index[-1] != last:
                continue
            p = f"noaa__{name}"
            r[f"{p}__anom__last"] = s.iloc[-1]
            for n in (3, 6):
                r[f"{p}__anom__tr{n}"] = s.iloc[-1] - s.iloc[-1 - n]
            if name in ("natl", "satl", "trop", "grad_atl"):
                r[f"{p}__anom__jfm"] = s.iloc[-3:].mean()
        out[y] = r
    return pd.DataFrame(out).T


def build_features(raw_dir, train, last_year=2024):
    parts = [manaus_table(train)] + [causal_anoms(station_table(raw_dir, s, last_year), f"ana__{s}") for s in STATIONS]
    parts.append(index_features(raw_dir, range(1951, last_year + 1)))
    F = pd.concat(parts, axis=1).sort_index()
    F = F[F.index <= last_year]
    F.index.name = "ano"
    return F.astype(float)


# =============================================================================================
# 5. Modelo: curva-base de 15 anos + PCA (2 componentes) + Ridge, tres membros
# =============================================================================================
def predict_delta(Xtr, Ytr, xte, alpha, nc=NC, center=CENTER):
    """Preve a curva Delta(h) de um ano. Imputacao, padronizacao e PCA so com os anos de treino."""
    Xtr, xte = np.asarray(Xtr, float), np.asarray(xte, float).reshape(1, -1)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med)
    xte = np.where(np.isfinite(xte), xte, med)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xs, xs = (Xtr - mu) / sd, (xte - mu) / sd
    ymu = Ytr.mean(0)
    pca = PCA(min(nc, len(Xs) - 1), random_state=SEED).fit(Ytr - ymu)
    S = pca.transform(Ytr - ymu)
    ssd = S.std(0) + 1e-9
    m = Ridge(alpha=alpha).fit(Xs, S / ssd)

    def f(Z):
        return ymu + (np.asarray(m.predict(Z)).reshape(len(Z), -1) * ssd) @ pca.components_
    k = min(center, len(Ytr))
    # a regressao so DESLOCA a media dos ultimos k anos
    return Ytr[-k:].mean(0) + f(xs)[0] - f(Xs[-k:]).mean(0)


def prever(ano, F, D, c0):
    """Curva de cota prevista (184 dias) para `ano`, treinando so com anos em [START, ano)."""
    tr = [t for t in D.index if START <= t < ano]
    curvas = {nome: c0[ano] + predict_delta(F.loc[tr, feats].values, D.loc[tr].values, F.loc[ano, feats].values, alpha)
              for nome, (feats, alpha) in MEMBROS.items()}
    return np.mean(list(curvas.values()), axis=0), curvas


def nota(pred, obs, min_ref=None):
    """Nota Final: 0,60*RMSE(out-nov) + 0,25*RMSE(jul-dez) + 0,15*|min previsto - min de referencia|."""
    p, o = np.asarray(pred, float), np.asarray(obs, float)
    e2 = (p - o) ** 2
    r_on, r_all = np.sqrt(e2[ON_MASK].mean()), np.sqrt(e2.mean())
    em = abs(p.min() - (o.min() if min_ref is None else min_ref))
    return 0.60 * r_on + 0.25 * r_all + 0.15 * em, r_on, r_all, em


# =============================================================================================
# 6. Execucao
# =============================================================================================
def main():
    base = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Previsao da cota do Rio Negro em Manaus, jul-dez/2024 (Equipe Outliers)")
    ap.add_argument("--treino", default=base / "dados" / "train.csv")
    ap.add_argument("--externos", default=base / "dados" / "externos")
    ap.add_argument("--saida", default=base / "saida")
    ap.add_argument("--observado", default=base / "dados" / "avaliacao" / "cota_observada_jul_dez_2024.csv")
    ap.add_argument("--sem-observado", action="store_true")
    args = ap.parse_args()
    out = Path(args.saida)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def say(txt=""):
        print(txt)
        log.append(txt)

    # --- dados
    train = load_train(args.treino)
    assert train["cota"].diff().abs().max() <= 0.45 + 1e-9
    piv = year_matrix(train)
    D = delta_matrix(piv)
    c0 = piv["06-30"]
    say(f"treino: {len(train)} dias, de {train['data'].min().date()} a {train['data'].max().date()} | "
        f"cota em 30/06/2024 = {c0[2024]:.2f} m")
    F = build_features(args.externos, train)
    say(f"tabela de variaveis: {F.shape[0]} anos x {F.shape[1]} variaveis (14 usadas pelo modelo)")

    # --- backtest: para cada ano Y de 1995 a 2023, treina so com anos < Y
    obs = {y: piv.loc[y, H_DAYS].values for y in ANOS_BT}
    pred_bt = {y: prever(y, F, D, c0)[0] for y in ANOS_BT}
    bt = pd.Series({y: nota(pred_bt[y], obs[y])[0] for y in ANOS_BT})
    m0 = pd.Series({y: nota(c0[y] + D.loc[y - 15:y - 1].mean(0).values, obs[y])[0] for y in ANOS_BT})
    say("\nBACKTEST (Nota Final com o minimo observado de cada ano; menor e melhor)")
    say(f"{'bloco':28s} {'modelo':>7s} {'baseline 15 anos':>17s}")
    for nome, ys in (("2005-2018", range(2005, 2019)), ("2019-2023", range(2019, 2024)), ("media 2005-2023", range(2005, 2024)),
                     ("secas (2005,2010,2015,2023)", [2005, 2010, 2015, 2023]), ("1995-2023", ANOS_BT)):
        say(f"{nome:28s} {bt[list(ys)].mean():7.2f} {m0[list(ys)].mean():17.2f}")

    # --- previsao de 2024
    pred, membros = prever(2024, F, D, c0)
    dias = pd.date_range("2024-07-01", "2024-12-31")
    cota = np.round(pred, 2)
    if (cota == cota.min()).sum() > 1:                                # minimo unico depois do arredondamento
        cota[int(pred.argmin())] -= 0.01
    sub = pd.DataFrame({"data": [d.strftime("%Y-%m-%d") for d in dias], "cota": cota})
    assert len(sub) == 184 and sub["cota"].notna().all()
    assert np.abs(np.diff(np.r_[c0[2024], sub["cota"].values])).max() <= 0.45
    sub.to_csv(out / "previsao_cota_2024.csv", index=False, float_format="%.2f")
    say("\nPREVISAO 2024")
    say(f"minimo previsto: {sub.cota.min():.2f} m em {sub.data[sub.cota.idxmin()]} | media out-nov {cota[ON_MASK].mean():.2f} m | "
        f"31/12: {cota[-1]:.2f} m")
    say("minimo de cada membro: " + ", ".join(f"{k} {v.min():.2f} m" for k, v in membros.items()))

    # --- incerteza e probabilidade (bonus): previsao + curvas inteiras de erro do backtest
    traj = np.array([pred + (obs[y] - pred_bt[y]) for y in ANOS_BT])
    p14 = float((traj[:, OCT_MASK].min(axis=1) < 14).mean())
    q05, q10, q90, q95 = np.quantile(traj, [.05, .10, .90, .95], axis=0)
    say("\nPROBABILIDADE (bonus de alerta)")
    say(f"P(cota < 14 m em outubro/2024) = {p14:.0%} ({int((traj[:, OCT_MASK].min(axis=1) < 14).sum())} de {len(traj)} trajetorias)")
    say(f"P(cota < 17 m em outubro/2024) = {(traj[:, OCT_MASK].min(axis=1) < 17).mean():.0%}")

    # --- observado de jul-dez/2024: so para graficos e Nota Final (nao entra no modelo)
    obs24 = None
    if not args.sem_observado and Path(args.observado).exists():
        obs24 = pd.read_csv(args.observado, parse_dates=["data"]).set_index("data")["cota"].reindex(dias)
        assert obs24.notna().all(), "observado incompleto"
        n, r_on, r_all, em = nota(sub["cota"].values, obs24.values, min_ref=12.11)
        bonus = -0.20 if p14 >= 0.10 else 0.0
        say("\nNOTA FINAL (regras do hackathon, com a cota observada de jul-dez/2024)")
        say(f"0,60 x RMSE out-nov  {r_on:.2f} = {0.60 * r_on:.2f}")
        say(f"0,25 x RMSE jul-dez  {r_all:.2f} = {0.25 * r_all:.2f}")
        say(f"0,15 x |{sub.cota.min():.2f} - 12,11| {em:.2f} = {0.15 * em:.2f}")
        say(f"Nota Final (formula) = {n:.2f} m")
        say(f"Bonus de alerta (P informada {p14:.0%} >= 10%) = {bonus:+.2f} m")
        say(f"Nota Final com bonus = {n + bonus:.2f} m")

    # --- grafico 1: o ano de 2024 inteiro
    jan_jun = train[train["data"] >= "2024-01-01"].set_index("data")["cota"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(jan_jun.index, jan_jun.values, color="k", lw=2, label="Observado (jan-jun/2024)")
    if obs24 is not None:
        ax.plot(obs24.index, obs24.values, color="C3", lw=1.8, label="Observado (jul-dez/2024)")
    ax.fill_between(dias, q05, q95, color="C0", alpha=.15, label="Intervalo de 90%")
    ax.fill_between(dias, q10, q90, color="C0", alpha=.25, label="Intervalo de 80%")
    ax.plot(dias, sub["cota"], color="C0", lw=2, label="Previsto (jul-dez/2024)")
    ax.axvline(CUTOFF, color="gray", ls="--", label="Corte da informacao: 30/06/2024")
    ax.set(xlabel="Data", ylabel="Cota no Porto de Manaus (m)",
           title="Rio Negro em Manaus, 2024: observado e previsto - Equipe Outliers")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "grafico1_2024.png", dpi=150)
    plt.close(fig)

    # --- grafico 2: zoom na seca (set-nov/2024)
    m = (dias >= "2024-09-01") & (dias <= "2024-11-30")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(dias[m], q05[m], q95[m], color="C0", alpha=.15, label="Intervalo de 90%")
    ax.fill_between(dias[m], q10[m], q90[m], color="C0", alpha=.25, label="Intervalo de 80%")
    ax.plot(dias[m], sub["cota"][m], color="C0", lw=2, label="Previsto")
    if obs24 is not None:
        ax.plot(dias[m], obs24.values[m], color="C3", lw=1.8, label="Observado")
    ax.axhline(17, color="orange", ls="--", label="Alerta: 17 m")
    ax.axhline(14, color="red", ls="--", label="Seca severa: 14 m")
    ax.plot(pd.Timestamp("2024-10-09"), 12.11, marker="*", ms=15, color="k", ls="none", label="09/10/2024: 12,11 m (minimo observado)")
    ax.set(xlabel="Data", ylabel="Cota no Porto de Manaus (m)",
           title=f"Vazante de set-nov/2024 | P(cota < 14 m em outubro) = {p14:.0%} - Equipe Outliers")
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "grafico2_vazante.png", dpi=150)
    plt.close(fig)

    say(f"\narquivos gravados em {out.name}/: previsao_cota_2024.csv, grafico1_2024.png, grafico2_vazante.png, resumo_execucao.txt")
    (out / "resumo_execucao.txt").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
