import pandas as pd
import numpy as np
from typing import Dict


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute Precision, Recall, F1, F2, F3 from a detailed evaluation results CSV.

    Expected columns: sample_id, total_matches, total_gold_facts,
                      total_relevant_unmatched, total_irrelevant_unmatched
    """
    sample_level = df.groupby('sample_id').first().reset_index()

    sample_level['total_predicted_facts'] = (
        sample_level['total_matches'] +
        sample_level['total_relevant_unmatched'] +
        sample_level['total_irrelevant_unmatched']
    )

    # Core metrics
    # Fact Recall (IOGT) = Matches / Gold Facts
    sample_level['fact_recall'] = (
        sample_level['total_matches'] / sample_level['total_gold_facts']
    )

    # Fact Precision = Matches / Predicted Facts
    sample_level['fact_precision'] = (
        sample_level['total_matches'] / sample_level['total_predicted_facts']
    )

    # Relevance = (Matches + Relevant Unmatched) / Predicted Facts
    sample_level['relevance'] = (
        (sample_level['total_matches'] + sample_level['total_relevant_unmatched']) /
        sample_level['total_predicted_facts']
    )

    # F-score family: F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
    for beta in [1, 2, 3]:
        p = sample_level['fact_precision']
        r = sample_level['fact_recall']
        denom = beta**2 * p + r
        sample_level[f'f{beta}'] = np.where(
            denom > 0,
            (1 + beta**2) * p * r / denom,
            0.0
        )

    summary = {
        'total_samples': len(sample_level),
        'avg_fact_recall': sample_level['fact_recall'].mean(),
        'avg_fact_precision': sample_level['fact_precision'].mean(),
        'avg_relevance': sample_level['relevance'].mean(),
        'avg_f1': sample_level['f1'].mean(),
        'avg_f2': sample_level['f2'].mean(),
        'avg_f3': sample_level['f3'].mean(),
        'avg_matches': sample_level['total_matches'].mean(),
        'avg_gold_facts': sample_level['total_gold_facts'].mean(),
        'avg_predicted_facts': sample_level['total_predicted_facts'].mean(),
        'avg_contradictions': sample_level['total_contradictions'].mean(),
    }

    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    print(f"Total Samples:              {summary['total_samples']}")
    print(f"\nCORE METRICS:")
    print(f"  Fact Recall (IOGT):       {summary['avg_fact_recall']:.4f}  ({summary['avg_fact_recall']*100:.2f}%)")
    print(f"  Fact Precision:           {summary['avg_fact_precision']:.4f}  ({summary['avg_fact_precision']*100:.2f}%)")
    print(f"  Relevance:                {summary['avg_relevance']:.4f}  ({summary['avg_relevance']*100:.2f}%)")
    print(f"\nF-SCORES:")
    print(f"  F1 (balanced):            {summary['avg_f1']:.4f}")
    print(f"  F2 (recall-weighted 2x):  {summary['avg_f2']:.4f}")
    print(f"  F3 (recall-weighted 3x):  {summary['avg_f3']:.4f}")
    print(f"\nSUPPORTING:")
    print(f"  Avg Matches:              {summary['avg_matches']:.2f}")
    print(f"  Avg Gold Facts:           {summary['avg_gold_facts']:.2f}")
    print(f"  Avg Predicted Facts:      {summary['avg_predicted_facts']:.2f}")
    print(f"  Avg Contradictions:       {summary['avg_contradictions']:.2f}")
    print("=" * 80)
    print("\nFORMULAS:")
    print("  Fact Recall    = Matches / Gold Facts")
    print("  Fact Precision = Matches / Predicted Facts")
    print("  Relevance      = (Matches + Relevant Unmatched) / Predicted Facts")
    print("  F_beta         = (1 + b²) × P × R / (b² × P + R)")
    print("=" * 80)

    return summary


def compute_metrics_from_file(csv_path: str) -> Dict[str, float]:
    df = pd.read_csv(csv_path, sep=',', on_bad_lines='skip', engine='python')
    return compute_metrics(df)
