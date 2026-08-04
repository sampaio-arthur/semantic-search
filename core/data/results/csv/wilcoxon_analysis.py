"""
Wilcoxon Signed-Rank Test — Análise de Significância Estatística
TCC: A Controlled Comparative Study of Vector Transformations in Semantic Search
"""

import csv
import sys
import numpy as np
from pathlib import Path
from scipy import stats


CSV_FILES = {
    10:  "benchmark_beir-trec-covid_k10_20260407T163307.csv",
    25:  "benchmark_beir-trec-covid_k25_20260407T163427.csv",
    50:  "benchmark_beir-trec-covid_k50_20260407T163609.csv",
    100: "benchmark_beir-trec-covid_k100_20260407T163750.csv",
}

ALPHA = 0.05


def load_per_query_ndcg(filepath: Path):
    content = filepath.read_text()
    lines = content.strip().split("\n")
    header_idx = None
    for i, line in enumerate(lines):
        if "query_id" in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Seção por query não encontrada em {filepath.name}")
    classical, quantum, statistical = [], [], []
    for line in lines[header_idx + 1:]:
        if not line.strip():
            continue
        parts = list(csv.reader([line]))[0]
        if len(parts) < 4:
            continue
        pipeline = parts[0].strip()
        try:
            ndcg = float(parts[3])
        except (ValueError, IndexError):
            continue
        if pipeline == "classical":
            classical.append(ndcg)
        elif pipeline == "quantum":
            quantum.append(ndcg)
        elif pipeline == "statistical":
            statistical.append(ndcg)
    return np.array(classical), np.array(quantum), np.array(statistical)


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    nz = diff[diff != 0]
    if len(nz) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    W_plus  = float(np.sum(ranks[nz > 0]))
    W_minus = float(np.sum(ranks[nz < 0]))
    T = len(nz) * (len(nz) + 1) / 2
    return (W_plus - W_minus) / T


def wilcoxon_pair(a: np.ndarray, b: np.ndarray):
    diff = a - b
    wins   = int(np.sum(diff > 0))
    losses = int(np.sum(diff < 0))
    ties   = int(np.sum(diff == 0))
    delta  = float(np.mean(diff))

    w_one, p_one = stats.wilcoxon(a, b, alternative="greater")
    w_two, p_two = stats.wilcoxon(a, b, alternative="two-sided")
    r = rank_biserial(a, b)

    return {
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)), "delta": delta,
        "wins": wins, "losses": losses, "ties": ties,
        "W_one": w_one, "p_one": p_one,
        "W_two": w_two, "p_two": p_two,
        "r": r,
        "significant": p_two < ALPHA,
    }


def print_pair(name_a, name_b, res):
    sig = "SIGNIFICATIVO" if res["significant"] else "nao significativo"
    print(f"  {name_a} vs {name_b}")
    print(f"    mean_{name_a[:4]}={res['mean_a']:.4f}  mean_{name_b[:4]}={res['mean_b']:.4f}  delta={res['delta']:+.4f}")
    print(f"    wins={res['wins']}  losses={res['losses']}  ties={res['ties']}")
    print(f"    W(two-sided)={res['W_two']:.2f}  p(one-tailed)={res['p_one']:.4f}  p(two-tailed)={res['p_two']:.4f}  r_rb={res['r']:.3f}")
    print(f"    -> {sig} (alpha={ALPHA})")


def main():
    script_dir = Path(__file__).parent
    all_results = {}

    for k, filename in CSV_FILES.items():
        filepath = script_dir / filename
        if not filepath.exists():
            print(f"[ERRO] Arquivo nao encontrado: {filepath}")
            sys.exit(1)

        c, q, s = load_per_query_ndcg(filepath)

        if len(c) == 0:
            print(f"[ERRO] Nenhum dado por query encontrado em {filename}")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"k={k}  n={len(c)}  Classical={np.mean(c):.4f}  Statistical={np.mean(s):.4f}  Quantum={np.mean(q):.4f}")
        print(f"{'='*60}")

        res_sc = wilcoxon_pair(s, c)
        res_cq = wilcoxon_pair(c, q)
        res_sq = wilcoxon_pair(s, q)

        print_pair("Statistical", "Classical", res_sc)
        print_pair("Classical",   "Quantum",   res_cq)
        print_pair("Statistical", "Quantum",   res_sq)

        all_results[k] = {"sc": res_sc, "cq": res_cq, "sq": res_sq}

    print(f"\n{'='*65}")
    print("TABELA p-values (two-tailed Wilcoxon, n=50)")
    print(f"{'='*65}")
    print(f"{'Comparacao':<30} {'k=10':>8} {'k=25':>8} {'k=50':>8} {'k=100':>8}")
    print("-"*65)
    pairs = [("Statistical vs Classical","sc"), ("Classical vs Quantum","cq"), ("Statistical vs Quantum","sq")]
    for label, key in pairs:
        row = f"{label:<30}"
        for k in [10, 25, 50, 100]:
            p = all_results[k][key]["p_two"]
            marker = "*" if p < ALPHA else " "
            row += f"  {p:.4f}{marker}"
        print(row)
    print("-"*65)

    print(f"\n{'='*65}")
    print("TABELA effect size r_rb = (W+ - W-) / T")
    print(f"{'='*65}")
    print(f"{'Comparacao':<30} {'k=10':>8} {'k=25':>8} {'k=50':>8} {'k=100':>8}")
    print("-"*65)
    for label, key in pairs:
        row = f"{label:<30}"
        for k in [10, 25, 50, 100]:
            row += f"  {all_results[k][key]['r']:>8.3f}"
        print(row)
    print("-"*65)
    print(f"* p < {ALPHA}  |  r_rb: <0.1 negligivel, 0.1-0.3 pequeno, 0.3-0.5 medio, >0.5 grande")


if __name__ == "__main__":
    main()