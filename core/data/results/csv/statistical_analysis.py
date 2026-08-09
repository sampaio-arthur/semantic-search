from __future__ import annotations

import csv
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

CSV_FILES: dict[int, str] = {
    10: "benchmark_beir-trec-covid_k10_20260407T163307.csv",
    25: "benchmark_beir-trec-covid_k25_20260407T163427.csv",
    50: "benchmark_beir-trec-covid_k50_20260407T163609.csv",
    100: "benchmark_beir-trec-covid_k100_20260407T163750.csv",
}

K_VALUES: tuple[int, ...] = (10, 25, 50, 100)
PIPELINES: tuple[str, ...] = ("classical", "quantum", "statistical")

METRICS: dict[str, tuple[str, str]] = {
    "ndcg": ("mean_ndcg_at_k", "ndcg_at_k"),
    "recall": ("mean_recall_at_k", "recall_at_k"),
    "mrr": ("mean_mrr_at_k", "mrr_at_k"),
    "precision": ("mean_precision_at_k", "precision_at_k"),
}

TIMING_COLUMNS: tuple[str, ...] = ("encode_time_ms", "search_time_ms", "total_time_ms")

EXPECTED_QUERIES = 50
ALPHA = 0.05
RECONCILIATION_TOLERANCE = 1e-4

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
CI_LOW_PCT = 2.5
CI_HIGH_PCT = 97.5

PAIRS: tuple[tuple[str, str], ...] = (
    ("statistical", "classical"),
    ("classical", "quantum"),
    ("statistical", "quantum"),
)

TARGET_METRIC = "ndcg"


class ValidationError(Exception):
    pass


@dataclass(slots=True)
class BenchmarkFile:

    k: int
    filename: str
    aggregate: dict[str, dict[str, float]]
    per_query: dict[str, dict[str, np.ndarray]]
    query_ids: list[str]


def _split_blocks(lines: list[str]) -> tuple[list[str], list[str]]:
    blank_indices = [i for i, line in enumerate(lines) if not line.strip()]
    if not blank_indices:
        raise ValidationError("nenhuma linha em branco separando os dois blocos")

    for idx in blank_indices:
        remainder = [ln for ln in lines[idx + 1 :] if ln.strip()]
        if remainder and remainder[0].startswith("pipeline,query_id"):
            first = [ln for ln in lines[:idx] if ln.strip()]
            return first, remainder

    raise ValidationError(
        "não foi possível localizar o cabeçalho do bloco por consulta "
        "('pipeline,query_id,...') após uma linha em branco"
    )


def _parse_rows(block: list[str], expected_header_prefix: str, filename: str) -> list[dict[str, str]]:
    header_line = block[0]
    if not header_line.startswith(expected_header_prefix):
        raise ValidationError(
            f"{filename}: cabeçalho inesperado.\n"
            f"  esperado começar com: {expected_header_prefix}\n"
            f"  encontrado:           {header_line}"
        )
    return list(csv.DictReader(block))


def load_benchmark(path: Path, k: int) -> BenchmarkFile:
    if not path.exists():
        raise ValidationError(f"arquivo não encontrado: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    agg_block, pq_block = _split_blocks(lines)

    agg_rows = _parse_rows(agg_block, "pipeline,k,query_count,error_count", path.name)
    pq_rows = _parse_rows(pq_block, "pipeline,query_id,query_text", path.name)

    agg_pipelines = [row["pipeline"].strip() for row in agg_rows]
    if sorted(agg_pipelines) != sorted(PIPELINES):
        raise ValidationError(
            f"{path.name}: bloco agregado deve conter exatamente os pipelines "
            f"{sorted(PIPELINES)}, encontrado {sorted(agg_pipelines)}"
        )

    aggregate: dict[str, dict[str, float]] = {}
    for row in agg_rows:
        pipeline = row["pipeline"].strip()
        error_count = int(row["error_count"])
        if error_count != 0:
            raise ValidationError(
                f"{path.name}: error_count={error_count} para o pipeline "
                f"'{pipeline}' (esperado 0)"
            )
        row_k = int(row["k"])
        if row_k != k:
            raise ValidationError(
                f"{path.name}: coluna k={row_k} para o pipeline '{pipeline}', "
                f"mas o arquivo declara k={k} no nome"
            )
        numeric_cols = [agg for agg, _ in METRICS.values()] + [
            f"mean_{col}" for col in TIMING_COLUMNS
        ]
        aggregate[pipeline] = {col: float(row[col]) for col in numeric_cols}

    grouped: dict[str, dict[str, dict[str, float]]] = {p: {} for p in PIPELINES}
    for row in pq_rows:
        pipeline = row["pipeline"].strip()
        if pipeline not in grouped:
            raise ValidationError(
                f"{path.name}: pipeline desconhecido '{pipeline}' no bloco por consulta"
            )
        query_id = row["query_id"].strip()
        if query_id in grouped[pipeline]:
            raise ValidationError(
                f"{path.name}: query_id duplicado '{query_id}' no pipeline '{pipeline}'"
            )
        value_cols = [pq for _, pq in METRICS.values()] + list(TIMING_COLUMNS)
        grouped[pipeline][query_id] = {col: float(row[col]) for col in value_cols}

    for pipeline in PIPELINES:
        count = len(grouped[pipeline])
        if count != EXPECTED_QUERIES:
            raise ValidationError(
                f"{path.name}: pipeline '{pipeline}' tem {count} consultas "
                f"(esperado {EXPECTED_QUERIES})"
            )

    reference_ids = set(grouped[PIPELINES[0]])
    for pipeline in PIPELINES[1:]:
        if set(grouped[pipeline]) != reference_ids:
            missing = reference_ids - set(grouped[pipeline])
            extra = set(grouped[pipeline]) - reference_ids
            raise ValidationError(
                f"{path.name}: conjunto de query_id difere entre "
                f"'{PIPELINES[0]}' e '{pipeline}'. Ausentes: {sorted(missing)}; "
                f"Excedentes: {sorted(extra)}"
            )

    query_ids = sorted(reference_ids, key=lambda qid: (int(qid) if qid.isdigit() else 0, qid))

    per_query: dict[str, dict[str, np.ndarray]] = {}
    for pipeline in PIPELINES:
        columns = [pq for _, pq in METRICS.values()] + list(TIMING_COLUMNS)
        per_query[pipeline] = {
            col: np.array([grouped[pipeline][qid][col] for qid in query_ids], dtype=float)
            for col in columns
        }

    return BenchmarkFile(
        k=k,
        filename=path.name,
        aggregate=aggregate,
        per_query=per_query,
        query_ids=query_ids,
    )


def load_all(base_dir: Path) -> dict[int, BenchmarkFile]:
    data = {k: load_benchmark(base_dir / filename, k) for k, filename in CSV_FILES.items()}

    reference_k = K_VALUES[0]
    reference_ids = set(data[reference_k].query_ids)
    for k in K_VALUES[1:]:
        current = set(data[k].query_ids)
        if current != reference_ids:
            raise ValidationError(
                f"conjunto de query_id difere entre k={reference_k} e k={k}. "
                f"Ausentes: {sorted(reference_ids - current)}; "
                f"Excedentes: {sorted(current - reference_ids)}"
            )

    return data


def build_reconciliation(data: dict[int, BenchmarkFile]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k in K_VALUES:
        bench = data[k]
        for pipeline in PIPELINES:
            for metric, (agg_col, pq_col) in METRICS.items():
                aggregated = bench.aggregate[pipeline][agg_col]
                recomputed = float(np.mean(bench.per_query[pipeline][pq_col]))
                diff = abs(aggregated - recomputed)
                rows.append(
                    {
                        "k": k,
                        "file": bench.filename,
                        "pipeline": pipeline,
                        "metric": metric,
                        "aggregated": aggregated,
                        "recomputed": recomputed,
                        "abs_diff": diff,
                        "flagged": diff > RECONCILIATION_TOLERANCE,
                    }
                )
    return rows


def build_descriptives(data: dict[int, BenchmarkFile]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    columns = [(name, pq_col) for name, (_, pq_col) in METRICS.items()]
    columns += [(col, col) for col in TIMING_COLUMNS]

    for k in K_VALUES:
        bench = data[k]
        for pipeline in PIPELINES:
            for label, column in columns:
                values = bench.per_query[pipeline][column]
                rows.append(
                    {
                        "k": k,
                        "pipeline": pipeline,
                        "metric": label,
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)),
                        "n": int(values.size),
                    }
                )
    return rows

def rank_biserial(diff: np.ndarray) -> tuple[float, float, float]:
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return 0.0, 0.0, 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    w_plus = float(np.sum(ranks[nonzero > 0]))
    w_minus = float(np.sum(ranks[nonzero < 0]))
    total = w_plus + w_minus
    if total == 0.0:
        return 0.0, w_plus, w_minus
    return (w_plus - w_minus) / total, w_plus, w_minus


def wilcoxon_method_used(n_nonzero: int, diff: np.ndarray) -> str:
    if n_nonzero == 0:
        return "undefined"
    has_zeros = bool(np.any(diff == 0))
    magnitudes = np.abs(diff[diff != 0])
    has_ties = magnitudes.size != np.unique(magnitudes).size
    if n_nonzero <= 50 and not has_zeros and not has_ties:
        return "exact"
    return "normal approximation"


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    n = a.size
    indices = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    diff = a - b
    boot_stats = diff[indices].mean(axis=1)
    low = float(np.percentile(boot_stats, CI_LOW_PCT))
    high = float(np.percentile(boot_stats, CI_HIGH_PCT))
    observed = float(np.mean(diff))
    return observed, low, high


def holm_bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p_values[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def build_paired_tests(data: dict[int, BenchmarkFile]) -> list[dict[str, object]]:
    _, pq_col = METRICS[TARGET_METRIC]
    rows: list[dict[str, object]] = []

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
                    "metric": f"nDCG@{k}",
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
                }
            )

        adjusted = holm_bonferroni([float(row["p_raw"]) for row in family])
        for row, p_adj in zip(family, adjusted):
            row["p_holm"] = p_adj
            row["significant_holm"] = bool(p_adj < ALPHA)
            rows.append(row)

    return rows

def format_p(p_value: float) -> str:
    return "<0.0001" if p_value < 0.0001 else f"{p_value:.4f}"


def print_reconciliation(rows: list[dict[str, object]]) -> None:
    print("=" * 88)
    print("SAIDA 1 - RECONCILIACAO DE FONTES (bloco agregado vs. recalculo por consulta)")
    print("=" * 88)
    header = f"{'k':>4}  {'pipeline':<12} {'metric':<10} {'aggregated':>12} {'recomputed':>12} {'abs_diff':>12}  flag"
    print(header)
    print("-" * 88)
    for row in rows:
        flag = " <<< DIVERGENCIA" if row["flagged"] else ""
        print(
            f"{row['k']:>4}  {row['pipeline']:<12} {row['metric']:<10} "
            f"{row['aggregated']:>12.6f} {row['recomputed']:>12.6f} "
            f"{row['abs_diff']:>12.6f}{flag}"
        )
    print("-" * 88)
    flagged = sum(1 for row in rows if row["flagged"])
    print(f"Total de comparacoes: {len(rows)}  |  Divergencias > {RECONCILIATION_TOLERANCE:g}: {flagged}")
    print()


def print_descriptives(rows: list[dict[str, object]]) -> None:
    print("=" * 88)
    print("SAIDA 2 - ESTATISTICAS DESCRITIVAS (Tabela I, desvio amostral ddof=1)")
    print("=" * 88)
    print(f"{'k':>4}  {'pipeline':<12} {'metric':<16} {'mean':>12} {'std':>12} {'n':>4}")
    print("-" * 88)
    current_k = None
    for row in rows:
        if current_k is not None and row["k"] != current_k:
            print("-" * 88)
        current_k = row["k"]
        print(
            f"{row['k']:>4}  {row['pipeline']:<12} {row['metric']:<16} "
            f"{row['mean']:>12.4f} {row['std']:>12.4f} {row['n']:>4}"
        )
    print("-" * 88)
    print()


def print_paired_tests(rows: list[dict[str, object]]) -> None:
    print("=" * 118)
    print("SAIDA 3 - TESTES PAREADOS (nDCG@k, Wilcoxon signed-rank bilateral)")
    print("=" * 118)
    print(
        f"{'k':>4}  {'pair':<26} {'mean_diff':>10} {'W':>8} "
        f"{'p_raw':>9} {'p_holm':>9} {'sig':>5} {'r_rb':>7} "
        f"{'W/L/T':>10} {'CI95 (bootstrap)':>22} {'CI_has_0':>8}"
    )
    print("-" * 118)
    current_k = None
    for row in rows:
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
    print(f"Familia Holm-Bonferroni: os 3 pares dentro de cada k  |  alfa = {ALPHA}")
    print("Metodo Wilcoxon usado por comparacao:")
    for row in rows:
        print(f"  k={row['k']:>3}  {row['pair']:<26} -> {row['wilcoxon_method']}")
    print()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialize(row[key]) for key in fieldnames})


def _serialize(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.10f}"
    return value


def write_environment(path: Path) -> None:
    lines = [
        f"python={platform.python_version()}",
        f"python_implementation={platform.python_implementation()}",
        f"numpy={np.__version__}",
        f"scipy={scipy.__version__}",
        f"bootstrap_resamples={BOOTSTRAP_RESAMPLES}",
        f"bootstrap_seed={BOOTSTRAP_SEED}",
        f"alpha={ALPHA}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")

def main() -> int:
    base_dir = Path(__file__).resolve().parent

    try:
        data = load_all(base_dir)
    except ValidationError as exc:
        print(f"[ERRO DE VALIDACAO] {exc}", file=sys.stderr)
        return 1

    n_queries = len(data[K_VALUES[0]].query_ids)
    print(f"Diretorio: {base_dir}")
    print(f"Arquivos validados: {len(data)}  |  consultas por pipeline: {n_queries}")
    print()

    reconciliation = build_reconciliation(data)
    descriptives = build_descriptives(data)
    paired = build_paired_tests(data)

    print_reconciliation(reconciliation)
    print_descriptives(descriptives)
    print_paired_tests(paired)

    write_csv(
        base_dir / "reconciliation.csv",
        reconciliation,
        ["k", "file", "pipeline", "metric", "aggregated", "recomputed", "abs_diff", "flagged"],
    )
    write_csv(
        base_dir / "descriptives.csv",
        descriptives,
        ["k", "pipeline", "metric", "mean", "std", "n"],
    )
    write_csv(
        base_dir / "paired_tests.csv",
        paired,
        [
            "k", "metric", "pair", "pipeline_a", "pipeline_b",
            "mean_a", "mean_b", "mean_diff",
            "wins", "losses", "ties",
            "wilcoxon_W", "w_plus", "w_minus",
            "p_raw", "p_holm", "significant_holm", "wilcoxon_method",
            "r_rank_biserial",
            "ci_low", "ci_high", "ci_contains_zero",
        ],
    )
    write_environment(base_dir / "environment.txt")

    print("Arquivos gerados: reconciliation.csv, descriptives.csv, paired_tests.csv, environment.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
