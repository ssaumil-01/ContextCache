import csv
import json
import math
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from rag_cache.integrations.embeddings.sentence_transformer import SentenceTransformerEmbedder


def compute_cosine_similarity(vec_a, vec_b):
    """Computes cosine similarity (dot product since vectors are normalized)."""
    return sum(a * b for a, b in zip(vec_a, vec_b))


def calculate_percentile(data, p):
    """Calculates percentile in pure Python (linear interpolation)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return sorted_data[int(idx)]
    return sorted_data[low] * (high - idx) + sorted_data[high] * (idx - low)


def calculate_statistics(data):
    """Computes standard statistics for a list of floats."""
    if not data:
        return {}
    n = len(data)
    data_sorted = sorted(data)
    mean = sum(data) / n
    median = calculate_percentile(data_sorted, 0.5)
    p10 = calculate_percentile(data_sorted, 0.10)
    p25 = calculate_percentile(data_sorted, 0.25)
    p75 = calculate_percentile(data_sorted, 0.75)
    p90 = calculate_percentile(data_sorted, 0.90)
    return {
        "min": min(data),
        "max": max(data),
        "mean": mean,
        "median": median,
        "p10": p10,
        "p25": p25,
        "p75": p75,
        "p90": p90,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset.json")

    print("=== Step 1: Loading Calibration Dataset ===")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r") as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} query pairs successfully.")

    print("\n=== Step 2: Generating Embeddings & Similarities ===")
    # Initialize SentenceTransformer from integrations
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

    pos_scores = []
    neg_scores = []
    processed_pairs = []

    for idx, pair in enumerate(pairs):
        q_a = pair["query_a"]
        q_b = pair["query_b"]
        label = pair["expected_label"]
        desc = pair.get("description", "")

        # Embed using ContextCache SentenceTransformer
        vec_a = embedder.embed_query(q_a)
        vec_b = embedder.embed_query(q_b)
        sim = compute_cosine_similarity(vec_a, vec_b)

        if label == 1:
            pos_scores.append(sim)
        else:
            neg_scores.append(sim)

        processed_pairs.append(
            {
                "pair_index": idx,
                "query_a": q_a,
                "query_b": q_b,
                "expected_label": label,
                "similarity": sim,
                "description": desc,
            }
        )

    print(f"Processed {len(pos_scores)} positive pairs and {len(neg_scores)} negative pairs.")

    print("\n=== Step 3: Distribution Analysis ===")
    pos_stats = calculate_statistics(pos_scores)
    neg_stats = calculate_statistics(neg_scores)

    print("\nPositive Matches Similarity Distribution:")
    for k, v in pos_stats.items():
        print(f"  - {k:<6}: {v:.4f}")

    print("\nNegative Matches Similarity Distribution:")
    for k, v in neg_stats.items():
        print(f"  - {k:<6}: {v:.4f}")

    print("\n=== Step 4: Threshold Sweep ===")
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    sweep_results = []

    print(
        f"{'Threshold':<10} | {'TP':<4} | {'FP':<4} | {'TN':<4} | {'FN':<4} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<8}"
    )
    print("-" * 72)

    for th in thresholds:
        tp = fp = tn = fn = 0
        for p in processed_pairs:
            pred = 1 if p["similarity"] >= th else 0
            actual = p["expected_label"]
            if pred == 1 and actual == 1:
                tp += 1
            elif pred == 1 and actual == 0:
                fp += 1
            elif pred == 0 and actual == 0:
                tn += 1
            elif pred == 0 and actual == 1:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        sweep_results.append(
            {
                "threshold": th,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )
        print(
            f"{th:<10.2f} | {tp:<4} | {fp:<4} | {tn:<4} | {fn:<4} | {precision:<9.4f} | {recall:<9.4f} | {f1:<8.4f}"
        )

    print("\n=== Step 5: Recommendation Engine ===")

    # 1. Max F1
    max_f1_item = max(sweep_results, key=lambda x: x["f1_score"])

    # 2. Safety-First (Highest Precision where Recall > 0.40)
    safety_candidates = [x for x in sweep_results if x["precision"] >= 0.95 and x["recall"] > 0.40]
    if safety_candidates:
        safety_item = max(safety_candidates, key=lambda x: x["threshold"])
    else:
        # Fallback to highest precision
        safety_item = max(sweep_results, key=lambda x: (x["precision"], x["threshold"]))

    # 3. Balanced (Maximizes F1 while ensuring Precision is at least 0.80)
    balanced_candidates = [x for x in sweep_results if x["precision"] >= 0.80]
    if balanced_candidates:
        balanced_item = max(balanced_candidates, key=lambda x: x["f1_score"])
    else:
        balanced_item = max_f1_item

    print(
        f"1. Max F1 Score Threshold: {max_f1_item['threshold']:.2f} (F1={max_f1_item['f1_score']:.4f}, Precision={max_f1_item['precision']:.4f}, Recall={max_f1_item['recall']:.4f})"
    )
    print(
        f"2. Safety-First Threshold : {safety_item['threshold']:.2f} (Precision={safety_item['precision']:.4f}, Recall={safety_item['recall']:.4f})"
    )
    print(
        f"3. Balanced Threshold     : {balanced_item['threshold']:.2f} (F1={balanced_item['f1_score']:.4f}, Precision={balanced_item['precision']:.4f}, Recall={balanced_item['recall']:.4f})"
    )

    # Final Report Structures
    report_json_path = os.path.join(script_dir, "calibration_report.json")
    report_csv_path = os.path.join(script_dir, "calibration_report.csv")

    report_data = {
        "dataset_size": len(pairs),
        "positive_pairs_count": len(pos_scores),
        "negative_pairs_count": len(neg_scores),
        "distributions": {"positive_matches": pos_stats, "negative_matches": neg_stats},
        "sweep_details": sweep_results,
        "recommendations": {
            "max_f1": {
                "threshold": max_f1_item["threshold"],
                "precision": max_f1_item["precision"],
                "recall": max_f1_item["recall"],
                "f1_score": max_f1_item["f1_score"],
                "confusion_matrix": {
                    "tp": max_f1_item["tp"],
                    "fp": max_f1_item["fp"],
                    "tn": max_f1_item["tn"],
                    "fn": max_f1_item["fn"],
                },
            },
            "safety_first": {
                "threshold": safety_item["threshold"],
                "precision": safety_item["precision"],
                "recall": safety_item["recall"],
                "f1_score": safety_item["f1_score"],
                "confusion_matrix": {
                    "tp": safety_item["tp"],
                    "fp": safety_item["fp"],
                    "tn": safety_item["tn"],
                    "fn": safety_item["fn"],
                },
            },
            "balanced": {
                "threshold": balanced_item["threshold"],
                "precision": balanced_item["precision"],
                "recall": balanced_item["recall"],
                "f1_score": balanced_item["f1_score"],
                "confusion_matrix": {
                    "tp": balanced_item["tp"],
                    "fp": balanced_item["fp"],
                    "tn": balanced_item["tn"],
                    "fn": balanced_item["fn"],
                },
            },
        },
    }

    print("\n=== Step 6: Exporting Reports ===")

    # Export JSON
    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"Exported JSON report to {report_json_path}")

    # Export CSV
    with open(report_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Threshold", "TP", "FP", "TN", "FN", "Precision", "Recall", "F1_Score"])
        for r in sweep_results:
            writer.writerow(
                [
                    r["threshold"],
                    r["tp"],
                    r["fp"],
                    r["tn"],
                    r["fn"],
                    f"{r['precision']:.4f}",
                    f"{r['recall']:.4f}",
                    f"{r['f1_score']:.4f}",
                ]
            )
    print(f"Exported CSV report to {report_csv_path}")


if __name__ == "__main__":
    main()
