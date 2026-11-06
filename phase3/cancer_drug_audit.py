import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


CANCER = "TEST"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Create cancer-drug response audits for label generation.")
    parser.add_argument("--cancer", default=CANCER, help="Use the Phase 1 output for one cancer")
    parser.add_argument("--responses", type=Path, default=None)
    parser.add_argument("--drugs", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    phase1_dir = PROJECT_ROOT / "phase1" / "output" / args.cancer
    args.responses = args.responses or phase1_dir / "response_pairs.csv"
    args.drugs = args.drugs or phase1_dir / "drug_features.csv"
    args.output_dir = args.output_dir or PROJECT_ROOT / "phase3" / "audit" / args.cancer
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.drugs.open(encoding="utf-8-sig", newline="") as handle:
        drug_names = {row["drug_id"]: row["drug_name"] for row in csv.DictReader(handle)}
    pair_values = defaultdict(list)
    with args.responses.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["ln_ic50"] != "":
                pair_values[(row["cancer_type"], row["drug_id"], row["cell_id"])].append(float(row["ln_ic50"]))

    cancer_drug_pairs = defaultdict(list)
    for (cancer_type, drug_id, cell_id), values in pair_values.items():
        cancer_drug_pairs[(cancer_type, drug_id)].append((cell_id, values))

    group_rows, pair_rows = [], []
    for (cancer_type, drug_id), pairs in sorted(cancer_drug_pairs.items()):
        threshold = statistics.median([statistics.median(values) for _, values in pairs])
        sensitive = resistant = ties = 0
        for cell_id, values in sorted(pairs):
            pair_median = statistics.median(values)
            label = 1 if pair_median < threshold else 0 if pair_median > threshold else -1
            sensitive += label == 1
            resistant += label == 0
            ties += label == -1
            pair_rows.append({
                "cancer_type": cancer_type,
                "drug_id": drug_id,
                "drug_name": drug_names[drug_id],
                "cell_id": cell_id,
                "replicate_count": len(values),
                "replicate_min": min(values),
                "replicate_median": pair_median,
                "replicate_max": max(values),
                "replicate_delta": max(values) - min(values),
                "cancer_drug_median_threshold": threshold,
                "preliminary_pair_label": label,
                "threshold_tie": label == -1,
                "replicate_crosses_threshold": any(value < threshold for value in values) and any(value > threshold for value in values),
            })
        group_rows.append({
            "cancer_type": cancer_type,
            "drug_id": drug_id,
            "drug_name": drug_names[drug_id],
            "cell_count": len(pairs),
            "cancer_drug_median_threshold": threshold,
            "preliminary_sensitive_pair_count": sensitive,
            "preliminary_resistant_pair_count": resistant,
            "preliminary_threshold_tie_pair_count": ties,
        })

    write_csv(args.output_dir / "cancer_drug_response_audit.csv", list(group_rows[0]), group_rows)
    write_csv(args.output_dir / "cancer_drug_pair_audit.csv", list(pair_rows[0]), pair_rows)
    print(f"Wrote {len(group_rows)} cancer-drug groups and {len(pair_rows)} unique pairs.")


if __name__ == "__main__":
    main()
