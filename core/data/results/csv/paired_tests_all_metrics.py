"""Testes pareados de Wilcoxon nas quatro metricas (nDCG, Recall, MRR, Precision).

Objetivo
--------
O script `statistical_analysis.py` aplica o teste de Wilcoxon pareado apenas ao
nDCG (constante `TARGET_METRIC`). Este script repete **exatamente** o mesmo
procedimento para as 4 metricas (nDCG, Recall, MRR, Precision) x 4 cortes
(k = 10, 25, 50, 100) x 3 pares de pipelines, sem alterar o script original nem
o `paired_tests.csv` ja gerado.

Procedimento (identico ao de `build_paired_tests()`)
----------------------------------------------------
- Wilcoxon signed-rank bilateral (`zero_method="wilcox"`, `correction=False`,
  `method="auto"`).
- Correcao de Holm-Bonferroni com familia = os 3 pares de cada (metrica, corte).
- Tamanho de efeito: correlacao rank-biserial pareada.
- IC de 95% por bootstrap pareado das diferencas
  (`BOOTSTRAP_RESAMPLES = 10000`, `BOOTSTRAP_SEED = 42`), com um gerador novo
  por (metrica, corte) usado em sequencia para os 3 pares, na ordem de `PAIRS`.
  E o que o script original faz por corte, e por isso as linhas de nDCG saem
  identicas as de `paired_tests.csv`.

Entradas (mesma pasta do script)
--------------------------------
- benchmark_beir-trec-covid_k10_20260407T163307.csv
- benchmark_beir-trec-covid_k25_20260407T163427.csv
- benchmark_beir-trec-covid_k50_20260407T163609.csv
- benchmark_beir-trec-covid_k100_20260407T163750.csv

O parsing, a validacao e os testes sao reaproveitados de
`statistical_analysis.py` (`load_all`, `rank_biserial`, `wilcoxon_method_used`,
`paired_bootstrap_ci`, `holm_bonferroni`, `write_csv`).

Saidas (mesma pasta do script)
------------------------------
- paired_tests_all_metrics.csv         (48 linhas: 4 metricas x 4 cortes x 3 pares)
- table_iv_all_metrics.csv             (12 linhas: 3 pares x 4 cortes)
- paired_tests_all_metrics_output.txt  (toda a saida de console deste script)

Execucao
--------
    cd core/data/results/csv
    python paired_tests_all_metrics.py

Nenhum arquivo pre-existente e modificado.
"""

from __future__ import annotations

import io
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

from statistical_analysis import (
    ALPHA,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    K_VALUES,
    METRICS,
    PAIRS,
    BenchmarkFile,
    ValidationError,
    holm_bonferroni,
    load_all,
    paired_bootstrap_ci,
    rank_biserial,
    wilcoxon_method_used,
    write_csv,
)

METRIC_LABELS: dict[str, str] = {
    "ndcg": "nDCG",
    "recall": "Recall",
    "mrr": "MRR",
    "precision": "P",
}

PAIRED_FIELDNAMES = [
    "k", "metric", "pair", "pipeline_a", "pipeline_b",
    "mean_a", "mean_b", "mean_diff",
    "wins", "losses", "ties",
    "wilcoxon_W", "w_plus", "w_minus",
    "p_raw", "p_holm", "significant_holm", "wilcoxon_method",
    "r_rank_biserial",
    "ci_low", "ci_high", "ci_contains_zero",
]

TABLE_IV_FIELDNAMES = [
    "pair", "k",
    "nDCG_p_holm", "nDCG_r",
    "Recall_p_holm", "Recall_r",
    "MRR_p_holm", "MRR_r",
    "P_p_holm", "P_r",
]


class Tee(io.TextIOBase):
    """Duplica a saida de console para um buffer, preservando o stdout."""

    def __init__(self, stream: io.TextIOBase, buffer: io.StringIO) -> None:
        self._stream = stream
        self._buffer = buffer

    def write(self, text: str) -> int:
        self._buffer.write(text)
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()


def metric_label(metric: str, k: int) -> str:
    return f"{METRIC_LABELS[metric]}@{k}"


def build_paired_tests_all_metrics(
    data: dict[int, BenchmarkFile]
) -> list[dict[str, object]]:
    """Mesma logica de `build_paired_tests()`, com laco sobre as 4 metricas."""
    rows: list[dict[str, object]] = []

    for metric, (_, pq_col) in METRICS.items():
        for k in K_VALUES:
            bench = data[k]
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            family: list[dict[str, object]] = []

            for name_a, name_b in PAIRS:
                a = bench.per_query[name_a][pq_col]
                b = bench.per_query[name_b][pq_col]
                diff = a - b

                wins = int(np.sum(diff > 0))
                losses = int(np.sum(diff < 0))
                ties = int(np.sum(diff == 0))

                if wins + losses == 0:
                    raise ValidationError(
                        f"{metric_label(metric, k)} / {name_a}-{name_b}: as "
                        f"{diff.size} diferencas pareadas sao todas zero; o "
                        "teste de Wilcoxon e indefinido nesse caso."
                    )

                statistic, p_value = stats.wilcoxon(
                    a,
                    b,
                    alternative="two-sided",
                    zero_method="wilcox",
                    correction=False,
                    method="auto",
                )

                r_rb, w_plus, w_minus = rank_biserial(diff)
                method = wilcoxon_method_used(wins + losses, diff)
                observed, ci_low, ci_high = paired_bootstrap_ci(a, b, rng)

                family.append(
                    {
                        "k": k,
                        "metric": metric_label(metric, k),
                        "pair": f"{name_a}-{name_b}",
                        "pipeline_a": name_a,
                        "pipeline_b": name_b,
                        "mean_a": float(np.mean(a)),
                        "mean_b": float(np.mean(b)),
                        "mean_diff": observed,
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                        "wilcoxon_W": float(statistic),
                        "w_plus": w_plus,
                        "w_minus": w_minus,
                        "p_raw": float(p_value),
                        "wilcoxon_method": method,
                        "r_rank_biserial": r_rb,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "ci_contains_zero": bool(ci_low <= 0.0 <= ci_high),
                        "_metric_key": metric,
                    }
                )

            adjusted = holm_bonferroni([float(row["p_raw"]) for row in family])
            for row, p_adj in zip(family, adjusted):
                row["p_holm"] = p_adj
                row["significant_holm"] = bool(p_adj < ALPHA)
                rows.append(row)

    return rows


def build_table_iv(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Tabela-resumo par x corte, com p_holm e r de cada metrica (sem recalculo)."""
    indexed = {
        (str(row["_metric_key"]), int(row["k"]), str(row["pair"])): row for row in rows
    }

    table: list[dict[str, object]] = []
    for name_a, name_b in PAIRS:
        pair = f"{name_a}-{name_b}"
        for k in K_VALUES:
            entry: dict[str, object] = {"pair": pair, "k": k}
            for metric, label in METRIC_LABELS.items():
                source = indexed[(metric, k, pair)]
                entry[f"{label}_p_holm"] = source["p_holm"]
                entry[f"{label}_r"] = source["r_rank_biserial"]
            table.append(entry)
    return table


def format_p(p_value: float) -> str:
    return "<0.0001" if p_value < 0.0001 else f"{p_value:.4f}"


def print_header(base_dir: Path, n_queries: int) -> None:
    print("=" * 118)
    print("TESTES PAREADOS NAS QUATRO METRICAS (Wilcoxon signed-rank bilateral)")
    print("=" * 118)
    print(f"python={platform.python_version()}")
    print(f"python_implementation={platform.python_implementation()}")
    print(f"numpy={np.__version__}")
    print(f"scipy={scipy.__version__}")
    print(f"bootstrap_resamples={BOOTSTRAP_RESAMPLES}")
    print(f"bootstrap_seed={BOOTSTRAP_SEED}")
    print(f"alpha={ALPHA}")
    print()
    print(f"Diretorio: {base_dir}")
    print(f"Consultas pareadas por comparacao: {n_queries}")
    print("Metricas: nDCG, Recall, MRR, Precision  |  cortes: k = 10, 25, 50, 100")
    print("Holm-Bonferroni: familia = os 3 pares dentro de cada (metrica, corte).")
    print("IC de 95%: bootstrap pareado das diferencas, gerador novo por (metrica, corte).")
    print("Procedimento identico ao de statistical_analysis.py; as linhas de nDCG")
    print("reproduzem o paired_tests.csv.")
    print()


def print_paired_tests(rows: list[dict[str, object]]) -> None:
    for metric, label in METRIC_LABELS.items():
        subset = [row for row in rows if row["_metric_key"] == metric]
        print("=" * 118)
        print(f"{label} - testes pareados por corte")
        print("=" * 118)
        print(
            f"{'k':>4}  {'pair':<26} {'mean_diff':>10} {'W':>8} "
            f"{'p_raw':>9} {'p_holm':>9} {'sig':>5} {'r_rb':>7} "
            f"{'W/L/T':>10} {'CI95 (bootstrap)':>22} {'CI_has_0':>8}"
        )
        print("-" * 118)
        current_k = None
        for row in subset:
            if current_k is not None and row["k"] != current_k:
                print("-" * 118)
            current_k = row["k"]
            wlt = f"{row['wins']}/{row['losses']}/{row['ties']}"
            ci = f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
            sig = "sim" if row["significant_holm"] else "nao"
            contains = "sim" if row["ci_contains_zero"] else "nao"
            print(
                f"{row['k']:>4}  {row['pair']:<26} {row['mean_diff']:>+10.4f} "
                f"{row['wilcoxon_W']:>8.1f} {format_p(float(row['p_raw'])):>9} "
                f"{format_p(float(row['p_holm'])):>9} {sig:>5} "
                f"{row['r_rank_biserial']:>+7.4f} {wlt:>10} {ci:>22} {contains:>5}"
            )
        print("-" * 118)
        print("Metodo Wilcoxon usado por comparacao:")
        for row in subset:
            print(f"  k={row['k']:>3}  {row['pair']:<26} -> {row['wilcoxon_method']}")
        print()


def print_table_iv(table: list[dict[str, object]]) -> None:
    print("=" * 118)
    print("TABELA-RESUMO (par x corte): p_holm e r rank-biserial por metrica")
    print("=" * 118)
    print(
        f"{'pair':<26} {'k':>4}  "
        f"{'nDCG p':>9} {'nDCG r':>8} {'Recall p':>9} {'Recall r':>9} "
        f"{'MRR p':>9} {'MRR r':>8} {'P p':>9} {'P r':>8}"
    )
    print("-" * 118)
    current_pair = None
    for entry in table:
        if current_pair is not None and entry["pair"] != current_pair:
            print("-" * 118)
        current_pair = entry["pair"]
        print(
            f"{entry['pair']:<26} {entry['k']:>4}  "
            f"{format_p(float(entry['nDCG_p_holm'])):>9} {float(entry['nDCG_r']):>+8.4f} "
            f"{format_p(float(entry['Recall_p_holm'])):>9} {float(entry['Recall_r']):>+9.4f} "
            f"{format_p(float(entry['MRR_p_holm'])):>9} {float(entry['MRR_r']):>+8.4f} "
            f"{format_p(float(entry['P_p_holm'])):>9} {float(entry['P_r']):>+8.4f}"
        )
    print("-" * 118)
    print()


def main() -> int:
    base_dir = Path(__file__).resolve().parent

    console = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, console)
    try:
        try:
            data = load_all(base_dir)
            rows = build_paired_tests_all_metrics(data)
        except ValidationError as exc:
            sys.stdout = original_stdout
            print(f"[ERRO DE VALIDACAO] {exc}", file=sys.stderr)
            return 1

        n_queries = len(data[K_VALUES[0]].query_ids)
        print_header(base_dir, n_queries)

        table = build_table_iv(rows)

        print_paired_tests(rows)
        print_table_iv(table)

        write_csv(base_dir / "paired_tests_all_metrics.csv", rows, PAIRED_FIELDNAMES)
        write_csv(base_dir / "table_iv_all_metrics.csv", table, TABLE_IV_FIELDNAMES)

        print(
            f"Linhas: paired_tests_all_metrics.csv={len(rows)}  |  "
            f"table_iv_all_metrics.csv={len(table)}"
        )
        print(
            "Arquivos gerados: paired_tests_all_metrics.csv, "
            "table_iv_all_metrics.csv, paired_tests_all_metrics_output.txt"
        )
    finally:
        sys.stdout = original_stdout

    (base_dir / "paired_tests_all_metrics_output.txt").write_text(
        console.getvalue(), encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
