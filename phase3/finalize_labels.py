import argparse
import csv
from pathlib import Path


CANCER = "TEST"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Create final cancer-drug response labels.")
    parser.add_argument("--cancer", default=CANCER, help="Finalize labels for one cancer")
    parser.add_argument("--audit-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    args.audit_dir = args.audit_dir or PROJECT_ROOT / "phase3" / "audit" / args.cancer
    args.output_dir = args.output_dir or PROJECT_ROOT / "phase3" / "output" / args.cancer
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = read_csv(args.audit_dir / "cancer_drug_pair_audit.csv")
    group_rows = read_csv(args.audit_dir / "cancer_drug_response_audit.csv")
    group_lookup = {(row["cancer_type"], row["drug_id"]): row for row in group_rows}
    final_pairs = []
    for row in pair_rows:
        group = group_lookup[(row["cancer_type"], row["drug_id"])]
        is_tie = row["threshold_tie"] == "True"
        final_pairs.append({
            "cancer_type": row["cancer_type"],
            "cell_id": row["cell_id"],
            "drug_id": row["drug_id"],
            "drug_name": row["drug_name"],
            "pair_median_ln_ic50": row["replicate_median"],
            "threshold": row["cancer_drug_median_threshold"],
            "response_label": "" if is_tie else row["preliminary_pair_label"],
            "is_tie": int(is_tie),
            "replicate_count": row["replicate_count"],
            "replicate_min": row["replicate_min"],
            "replicate_max": row["replicate_max"],
            "replicate_delta": row["replicate_delta"],
            "replicate_crossing": int(row["replicate_crosses_threshold"] == "True"),
            "group_cell_count": group["cell_count"],
            "small_group": int(int(group["cell_count"]) < 4),
        })
    final_pairs.sort(key=lambda row: (row["cancer_type"], row["drug_name"], row["cell_id"]))
    write_csv(args.output_dir / "final_pair_labels.csv", list(final_pairs[0]), final_pairs)

    thresholds = [{
        "cancer_type": row["cancer_type"],
        "drug_id": row["drug_id"],
        "drug_name": row["drug_name"],
        "cell_count": row["cell_count"],
        "threshold": row["cancer_drug_median_threshold"],
        "n_sensitive": row["preliminary_sensitive_pair_count"],
        "n_resistant": row["preliminary_resistant_pair_count"],
        "n_tie": row["preliminary_threshold_tie_pair_count"],
    } for row in sorted(group_rows, key=lambda row: (row["cancer_type"], row["drug_name"]))]
    write_csv(args.output_dir / "cancer_drug_thresholds.csv", list(thresholds[0]), thresholds)
    sensitive = sum(row["response_label"] == "1" for row in final_pairs)
    resistant = sum(row["response_label"] == "0" for row in final_pairs)
    print(f"pairs={len(final_pairs)} sensitive={sensitive} resistant={resistant} ties={len(final_pairs) - sensitive - resistant}")


if __name__ == "__main__":
    main()
