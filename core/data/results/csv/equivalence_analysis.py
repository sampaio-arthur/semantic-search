"""Analise de equivalencia (TOST) entre os pipelines statistical e classical.

Objetivo
--------
O teste de Wilcoxon bilateral em `statistical_analysis.py` nao rejeitou a
igualdade entre os pipelines `statistical` e `classical` no nDCG. Isso e
ausencia de evidencia de diferenca, e nao evidencia de equivalencia. Este
script acrescenta um teste de equivalencia formal (TOST, *two one-sided
tests*) para o par `statistical - classical`, em 4 metricas (nDCG, Recall,
MRR, Precision) x 4 cortes (k = 10, 25, 50, 100).

Margem de equivalencia: `delta = 0.05 x media do pipeline classical` para
aquela metrica e corte (margem relativa de 5%), fixada pela regra de Sparck
Jones (1974), reportada por Sanderson (2010, p. 313), segundo a qual
diferencas abaixo de 5% nao sao perceptiveis em avaliacao de recuperacao. A
margem e definida pela regra e nao ajustada aos dados.

Teste primario: TOST pareado com distribuicao t (a margem esta definida
sobre a media). Confirmacao: intervalo de confianca bootstrap de 90%.
Registro suplementar (nao decide nada): TOST com Wilcoxon.

Entradas (mesma pasta do script)
--------------------------------
- benchmark_beir-trec-covid_k10_20260407T163307.csv
- benchmark_beir-trec-covid_k25_20260407T163427.csv
- benchmark_beir-trec-covid_k50_20260407T163609.csv
- benchmark_beir-trec-covid_k100_20260407T163750.csv

O parsing e a validacao sao reaproveitados de `statistical_analysis.py`
(`load_all`), que garante os mesmos 50 `query_id` em todos os pipelines e
cortes e monta os vetores por consulta na mesma ordem.

Saidas (mesma pasta do script)
------------------------------
- equivalence_tests.csv         (16 linhas: 4 metricas x 4 cortes)
- equivalence_sensitivity.csv   (112 linhas: 16 x 7 margens alternativas)
- equivalence_output.txt        (toda a saida de console deste script)

Execucao
--------
    cd core/data/results/csv
    python equivalence_analysis.py

Nenhum arquivo pre-existente e modificado.
"""

from __future__ import annotations

import csv
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
    ValidationError,
    holm_bonferroni,
    load_all,
)

PIPELINE_A = "statistical"
PIPELINE_B = "classical"
PAIR_LABEL = f"{PIPELINE_A}-{PIPELINE_B}"

DELTA_RULE = "relative_5pct"
RELATIVE_MARGIN = 0.05

CI90_LOW_PCT = 5.0
CI90_HIGH_PCT = 95.0
CI90_T_QUANTILE = 0.95

METRIC_LABELS: dict[str, str] = {
    "ndcg": "nDCG",
    "recall": "Recall",
    "mrr": "MRR",
    "precision": "P",
}

ABSOLUTE_MARGINS: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05)
RELATIVE_MARGINS: tuple[float, ...] = (0.025, 0.05, 0.10)

TESTS_FIELDNAMES = [
    "metric", "k", "pair", "n",
    "mean_statistical", "mean_classical", "mean_diff", "sd_diff",
    "delta_rule", "delta",
    "p_lower", "p_upper", "p_tost", "p_tost_holm", "equivalent",
    "ci90_t_low", "ci90_t_high",
    "ci90_boot_low", "ci90_boot_high", "ci90_inside", "agree",
    "p_wilcoxon_tost", "p_wilcoxon_tost_holm",
    "delta_min_t", "delta_min_t_rel", "delta_min_boot", "delta_min_boot_rel",
]

SENSITIVITY_FIELDNAMES = [
    "metric", "k", "delta_type", "delta_param", "delta",
    "p_tost", "p_tost_holm", "equivalent",
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


def tost_t(diff: np.ndarray, delta: float) -> tuple[float, float, float]:
    """TOST pareado baseado na media, com distribuicao t e df = n - 1."""
    n = diff.size
    df = n - 1
    d_bar = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(n))
    p_lower = float(stats.t.sf((d_bar + delta) / se, df))
    p_upper = float(stats.t.cdf((d_bar - delta) / se, df))
    return p_lower, p_upper, max(p_lower, p_upper)


def tost_wilcoxon(diff: np.ndarray, delta: float) -> float:
    """TOST com Wilcoxon (registro suplementar; nao decide equivalencia)."""
    p_lower = float(
        stats.wilcoxon(
            diff + delta,
            alternative="greater",
            zero_method="wilcox",
            correction=False,
            method="auto",
        ).pvalue
    )
    p_upper = float(
        stats.wilcoxon(
            diff - delta,
            alternative="less",
            zero_method="wilcox",
            correction=False,
            method="auto",
        ).pvalue
    )
    return max(p_lower, p_upper)


def bootstrap_ci90(diff: np.ndarray) -> tuple[float, float]:
    n = diff.size
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    boot = diff[idx].mean(axis=1)
    return (
        float(np.percentile(boot, CI90_LOW_PCT)),
        float(np.percentile(boot, CI90_HIGH_PCT)),
    )


def collect_differences(data) -> dict[str, dict[int, dict[str, object]]]:
    """Diferencas pareadas statistical - classical, alinhadas por query_id."""
    result: dict[str, dict[int, dict[str, object]]] = {}
    reference_ids = data[K_VALUES[0]].query_ids

    for metric, (_, pq_col) in METRICS.items():
        per_k: dict[int, dict[str, object]] = {}
        for k in K_VALUES:
            bench = data[k]
            if bench.query_ids != reference_ids:
                raise ValidationError(
                    f"k={k}: ordem/conjunto de query_id difere do corte de referencia "
                    f"k={K_VALUES[0]}; as diferencas pareadas nao estariam alinhadas"
                )
            a = bench.per_query[PIPELINE_A][pq_col]
            b = bench.per_query[PIPELINE_B][pq_col]
            if a.shape != b.shape:
                raise ValidationError(
                    f"k={k}, metrica {metric}: vetores de tamanhos diferentes "
                    f"({a.size} vs {b.size})"
                )
            per_k[k] = {
                "a": a,
                "b": b,
                "diff": a - b,
                "mean_a": float(np.mean(a)),
                "mean_b": float(np.mean(b)),
            }
        result[metric] = per_k
    return result


def build_equivalence_tests(diffs) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for metric in METRICS:
        family: list[dict[str, object]] = []
        for k in K_VALUES:
            entry = diffs[metric][k]
            diff: np.ndarray = entry["diff"]
            n = diff.size
            df = n - 1
            d_bar = float(np.mean(diff))
            sd = float(np.std(diff, ddof=1))
            se = sd / np.sqrt(n)

            mean_b = float(entry["mean_b"])
            delta = RELATIVE_MARGIN * mean_b

            p_lower, p_upper, p_tost = tost_t(diff, delta)

            t_crit = float(stats.t.ppf(CI90_T_QUANTILE, df))
            ci90_t_low = d_bar - t_crit * se
            ci90_t_high = d_bar + t_crit * se

            ci90_boot_low, ci90_boot_high = bootstrap_ci90(diff)
            ci90_inside = bool(ci90_boot_low > -delta and ci90_boot_high < delta)

            p_w_tost = tost_wilcoxon(diff, delta)

            delta_min_t = max(abs(ci90_t_low), abs(ci90_t_high))
            delta_min_boot = max(abs(ci90_boot_low), abs(ci90_boot_high))

            family.append(
                {
                    "metric": metric_label(metric, k),
                    "k": k,
                    "pair": PAIR_LABEL,
                    "n": n,
                    "mean_statistical": float(entry["mean_a"]),
                    "mean_classical": mean_b,
                    "mean_diff": d_bar,
                    "sd_diff": sd,
                    "delta_rule": DELTA_RULE,
                    "delta": delta,
                    "p_lower": p_lower,
                    "p_upper": p_upper,
                    "p_tost": p_tost,
                    "ci90_t_low": ci90_t_low,
                    "ci90_t_high": ci90_t_high,
                    "ci90_boot_low": ci90_boot_low,
                    "ci90_boot_high": ci90_boot_high,
                    "ci90_inside": ci90_inside,
                    "p_wilcoxon_tost": p_w_tost,
                    "delta_min_t": delta_min_t,
                    "delta_min_t_rel": delta_min_t / mean_b,
                    "delta_min_boot": delta_min_boot,
                    "delta_min_boot_rel": delta_min_boot / mean_b,
                    "_metric_key": metric,
                }
            )

        adjusted = holm_bonferroni([float(row["p_tost"]) for row in family])
        adjusted_w = holm_bonferroni([float(row["p_wilcoxon_tost"]) for row in family])
        for row, p_adj, p_adj_w in zip(family, adjusted, adjusted_w):
            row["p_tost_holm"] = p_adj
            row["equivalent"] = bool(p_adj < ALPHA)
            row["p_wilcoxon_tost_holm"] = p_adj_w
            row["agree"] = bool(row["equivalent"] == row["ci90_inside"])
            rows.append(row)

    return rows


def build_sensitivity(diffs) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    specs: list[tuple[str, float]] = [("absolute", m) for m in ABSOLUTE_MARGINS]
    specs += [("relative", m) for m in RELATIVE_MARGINS]

    for metric in METRICS:
        for delta_type, delta_param in specs:
            family: list[dict[str, object]] = []
            for k in K_VALUES:
                entry = diffs[metric][k]
                diff: np.ndarray = entry["diff"]
                if delta_type == "absolute":
                    delta = float(delta_param)
                else:
                    delta = float(delta_param) * float(entry["mean_b"])

                _, _, p_tost = tost_t(diff, delta)
                family.append(
                    {
                        "metric": metric_label(metric, k),
                        "k": k,
                        "delta_type": delta_type,
                        "delta_param": float(delta_param),
                        "delta": delta,
                        "p_tost": p_tost,
                    }
                )

            adjusted = holm_bonferroni([float(row["p_tost"]) for row in family])
            for row, p_adj in zip(family, adjusted):
                row["p_tost_holm"] = p_adj
                row["equivalent"] = bool(p_adj < ALPHA)
                rows.append(row)

    return rows


def format_p(p_value: float) -> str:
    return "<0.0001" if p_value < 0.0001 else f"{p_value:.4f}"


def print_header(base_dir: Path, n_queries: int) -> None:
    print("=" * 110)
    print("ANALISE DE EQUIVALENCIA (TOST) - statistical vs. classical")
    print("=" * 110)
    print(f"python={platform.python_version()}")
    print(f"python_implementation={platform.python_implementation()}")
    print(f"numpy={np.__version__}")
    print(f"scipy={scipy.__version__}")
    print(f"bootstrap_resamples={BOOTSTRAP_RESAMPLES}")
    print(f"bootstrap_seed={BOOTSTRAP_SEED}")
    print(f"alpha={ALPHA}")
    print()
    print(f"Diretorio: {base_dir}")
    print(f"Par: {PAIR_LABEL}  |  consultas pareadas por comparacao: {n_queries}")
    print(f"Margem: delta = {RELATIVE_MARGIN:g} x media(classical)  ({DELTA_RULE})")
    print("Regra de 5%: Sparck Jones (1974), via Sanderson (2010, p. 313).")
    print("Teste primario: TOST pareado t (a margem e definida sobre a media).")
    print("Confirmacao: IC bootstrap de 90% inteiramente dentro de (-delta, +delta).")
    print("Holm-Bonferroni: familia = os 4 cortes dentro de cada metrica.")
    print()


def print_tests(rows: list[dict[str, object]]) -> None:
    print("=" * 130)
    print("SAIDA 1 - TOST t PAREADO (primario) + CONFIRMACAO BOOTSTRAP 90%")
    print("=" * 130)
    for metric in METRICS:
        subset = [row for row in rows if row["_metric_key"] == metric]
        print(f"--- {METRIC_LABELS[metric]} " + "-" * (126 - len(METRIC_LABELS[metric])))
        print(
            f"{'metric':<12} {'n':>3} {'mean_stat':>10} {'mean_clas':>10} "
            f"{'mean_diff':>11} {'sd_diff':>9} {'delta':>9} "
            f"{'p_tost':>9} {'p_holm':>9} {'equiv':>6} "
            f"{'IC90 t':>22} {'IC90 bootstrap':>22} {'dentro':>7} {'agree':>6}"
        )
        for row in subset:
            ci_t = f"[{row['ci90_t_low']:+.6f}, {row['ci90_t_high']:+.6f}]"
            ci_b = f"[{row['ci90_boot_low']:+.6f}, {row['ci90_boot_high']:+.6f}]"
            print(
                f"{row['metric']:<12} {row['n']:>3} "
                f"{row['mean_statistical']:>10.6f} {row['mean_classical']:>10.6f} "
                f"{row['mean_diff']:>+11.6f} {row['sd_diff']:>9.6f} "
                f"{row['delta']:>9.6f} "
                f"{format_p(float(row['p_tost'])):>9} "
                f"{format_p(float(row['p_tost_holm'])):>9} "
                f"{('sim' if row['equivalent'] else 'nao'):>6} "
                f"{ci_t:>22} {ci_b:>22} "
                f"{('sim' if row['ci90_inside'] else 'nao'):>7} "
                f"{('sim' if row['agree'] else 'NAO'):>6}"
            )
        print()

    print("=" * 130)
    print("SAIDA 2 - MARGEM MINIMA DE EQUIVALENCIA (menor margem sustentada pelos dados)")
    print("=" * 130)
    print(
        f"{'metric':<12} {'delta_min_t':>13} {'rel_t':>9} "
        f"{'delta_min_boot':>15} {'rel_boot':>9}"
    )
    for row in rows:
        print(
            f"{row['metric']:<12} {row['delta_min_t']:>13.6f} "
            f"{row['delta_min_t_rel']:>9.4f} "
            f"{row['delta_min_boot']:>15.6f} {row['delta_min_boot_rel']:>9.4f}"
        )
    print()

    print("=" * 130)
    print("SAIDA 3 - TOST COM WILCOXON (registro suplementar, nao decide equivalencia)")
    print("=" * 130)
    print("O Wilcoxon testa a pseudo-mediana; a margem aqui esta definida sobre a media.")
    print("Em metricas com muitas diferencas iguais a zero (MRR) ele aceita equivalencia")
    print("mesmo quando a media esta fora da margem. Reportado apenas para transparencia.")
    print(f"{'metric':<12} {'p_wilcoxon_tost':>17} {'p_wilcoxon_tost_holm':>22}")
    for row in rows:
        print(
            f"{row['metric']:<12} "
            f"{format_p(float(row['p_wilcoxon_tost'])):>17} "
            f"{format_p(float(row['p_wilcoxon_tost_holm'])):>22}"
        )
    print()


def print_sensitivity(rows: list[dict[str, object]]) -> None:
    print("=" * 110)
    print("SAIDA 4 - SENSIBILIDADE A MARGEM (TOST t, Holm sobre os 4 cortes)")
    print("=" * 110)
    print(
        f"{'metric':<12} {'delta_type':<10} {'delta_param':>12} {'delta':>11} "
        f"{'p_tost':>9} {'p_holm':>9} {'equiv':>6}"
    )
    current = None
    for row in rows:
        key = (row["metric"].split("@")[0], row["delta_type"], row["delta_param"])
        if current is not None and key != current:
            print("-" * 110)
        current = key
        print(
            f"{row['metric']:<12} {row['delta_type']:<10} {row['delta_param']:>12g} "
            f"{row['delta']:>11.6f} "
            f"{format_p(float(row['p_tost'])):>9} "
            f"{format_p(float(row['p_tost_holm'])):>9} "
            f"{('sim' if row['equivalent'] else 'nao'):>6}"
        )
    print("-" * 110)
    print()


def _serialize(value: object) -> object:
    """Mesmo formato do `paired_tests.csv` (10 casas, booleanos true/false).

    Excecao: p-valores do TOST podem ser muito menores que 1e-10 e seriam
    achatados para "0.0000000000" pelo formato fixo, o que apagaria
    informacao. Nesses casos usa-se `repr`, que preserva a precisao completa
    do float (notacao cientifica com ponto decimal).
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.10f}"
        if value != 0.0 and float(text) == 0.0:
            return repr(value)
        return text
    return value


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialize(row[key]) for key in fieldnames})


def main() -> int:
    base_dir = Path(__file__).resolve().parent

    console = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, console)
    try:
        try:
            data = load_all(base_dir)
            diffs = collect_differences(data)
        except ValidationError as exc:
            sys.stdout = original_stdout
            print(f"[ERRO DE VALIDACAO] {exc}", file=sys.stderr)
            return 1

        n_queries = len(data[K_VALUES[0]].query_ids)
        print_header(base_dir, n_queries)

        tests = build_equivalence_tests(diffs)
        sensitivity = build_sensitivity(diffs)

        print_tests(tests)
        print_sensitivity(sensitivity)

        write_csv(base_dir / "equivalence_tests.csv", tests, TESTS_FIELDNAMES)
        write_csv(
            base_dir / "equivalence_sensitivity.csv", sensitivity, SENSITIVITY_FIELDNAMES
        )

        print(
            f"Linhas: equivalence_tests.csv={len(tests)}  |  "
            f"equivalence_sensitivity.csv={len(sensitivity)}"
        )
        print(
            "Arquivos gerados: equivalence_tests.csv, equivalence_sensitivity.csv, "
            "equivalence_output.txt"
        )
    finally:
        sys.stdout = original_stdout

    (base_dir / "equivalence_output.txt").write_text(
        console.getvalue(), encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
