"""Testes: sem vazamento temporal, formato da entrega e reprodutibilidade da previsao."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
SCRIPT = RAIZ / "previsao_rio_negro_2024.py"
CORTE = pd.Timestamp("2024-06-30")


@pytest.fixture(scope="module")
def saida(tmp_path_factory):
    out = tmp_path_factory.mktemp("saida")
    r = subprocess.run([sys.executable, str(SCRIPT), "--saida", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


def test_treino_termina_no_corte():
    train = pd.read_csv(RAIZ / "dados" / "train.csv", parse_dates=["data"])
    assert train["data"].max() == CORTE


@pytest.mark.parametrize("arq", ["cota_13870000_labrea.csv", "cota_15630000_humaita.csv"])
def test_ana_sem_dados_apos_corte(arq):
    df = pd.read_csv(RAIZ / "dados" / "externos" / "ana" / arq, parse_dates=["data"])
    assert df["data"].max() <= CORTE


def test_formato_da_previsao(saida):
    sub = pd.read_csv(saida / "previsao_cota_2024.csv")
    assert list(sub.columns) == ["data", "cota"]
    assert list(sub["data"]) == [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-07-01", "2024-12-31")]
    assert sub["cota"].notna().all()
    assert (sub["cota"] == sub["cota"].min()).sum() == 1          # minimo unico
    c0 = pd.read_csv(RAIZ / "dados" / "train.csv").iloc[-1]["cota"]
    assert np.abs(np.diff(np.r_[c0, sub["cota"].values])).max() <= 0.45 + 1e-9


def test_reproduz_previsao_publicada(saida):
    nova = pd.read_csv(saida / "previsao_cota_2024.csv")
    ref = pd.read_csv(RAIZ / "saida" / "previsao_cota_2024.csv")
    assert (nova["data"] == ref["data"]).all()
    assert np.abs(nova["cota"] - ref["cota"]).max() <= 0.011       # tolerancia de arredondamento entre versoes de bibliotecas


def test_resumo_traz_resultados_principais(saida):
    txt = (saida / "resumo_execucao.txt").read_text(encoding="utf-8")
    assert "minimo previsto:" in txt
    assert "P(cota < 14 m em outubro/2024)" in txt


def test_sem_observado_nao_calcula_nota(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "--saida", str(tmp_path), "--sem-observado"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "NOTA FINAL" not in (tmp_path / "resumo_execucao.txt").read_text(encoding="utf-8")
