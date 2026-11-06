import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

CANCER = "TEST"
N_TEST_DRUGS = 5
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_features(path, id_column, metadata_columns):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header[0] != id_column:
            raise ValueError(f"Expected {id_column} as the first column in {path}.")
        values = {row[0]: np.asarray(row[metadata_columns:], dtype=np.float32) for row in reader}
    return values


def load_samples(labels_path, cells_path, drugs_path):
    cells = read_features(cells_path, "cell_id", 2)
    drugs = read_features(drugs_path, "drug_id", 2)
    with labels_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            (row["cancer_type"], row["cell_id"], row["drug_id"]): {
                "cancer_type": row["cancer_type"], "cell_id": row["cell_id"],
                "drug_id": row["drug_id"], "drug_name": row["drug_name"],
                "label": int(row["response_label"]), "cell": cells[row["cell_id"]],
                "drug": drugs[row["drug_id"]],
            }
            for row in rows if row["is_tie"] != "1"
        }


def label_counts(samples):
    counts = Counter(int(sample["label"]) for sample in samples)
    return {"label_0": counts[0], "label_1": counts[1]}


def require_two_classes(samples, subset_name):
    counts = label_counts(samples)
    if counts["label_0"] == 0 or counts["label_1"] == 0:
        raise ValueError(f"{subset_name} must contain both labels: {counts}")


def main():
    parser = argparse.ArgumentParser(description="Create a drug-held-out split.")
    parser.add_argument("--cancer", default=CANCER, help="Cancer code, e.g. PRAD.")
    parser.add_argument(
        "--n-test-drugs",
        type=int,
        default=N_TEST_DRUGS,
        help="Number of drugs assigned to independent_test.",
    )
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to phase5/results/<cancer>.",
    )
    args = parser.parse_args()

    cancer = args.cancer.upper()
    if args.n_test_drugs < 1:
        raise ValueError("--n-test-drugs must be at least 1.")

    output_dir = args.output_dir or (PROJECT_ROOT / "phase5" / "results" / cancer)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = PROJECT_ROOT / "phase3" / "output" / cancer / "final_pair_labels.csv"
    cells_path = PROJECT_ROOT / "phase2" / "output" / cancer / "cell_gsva.csv"
    drugs_path = PROJECT_ROOT / "phase1" / "output" / cancer / "drug_features.csv"

    all_samples = list(load_samples(labels_path, cells_path, drugs_path).values())
    samples = [sample for sample in all_samples if sample["cancer_type"] == cancer]
    if not samples:
        raise ValueError(f"No valid pairs found for {cancer}.")

    samples_by_drug = defaultdict(list)
    for sample in samples:
        samples_by_drug[sample["drug_id"]].append(sample)

    if args.n_test_drugs >= len(samples_by_drug):
        raise ValueError(
            f"{cancer} has only {len(samples_by_drug)} valid drugs; "
            f"cannot hold out {args.n_test_drugs}."
        )

    ranked_drug_ids = sorted(
        samples_by_drug,
        key=lambda drug_id: (-len(samples_by_drug[drug_id]), drug_id),
    )
    test_drug_ids = set(ranked_drug_ids[: args.n_test_drugs])
    independent_test = [sample for sample in samples if sample["drug_id"] in test_drug_ids]
    development = [sample for sample in samples if sample["drug_id"] not in test_drug_ids]

    require_two_classes(independent_test, "independent_test")
    require_two_classes(development, "development")

    development_drug_ids = {sample["drug_id"] for sample in development}
    if len(development_drug_ids) < 5:
        raise ValueError("development must contain at least five drugs.")

    y = np.asarray([sample["label"] for sample in development])
    groups = np.asarray([sample["drug_id"] for sample in development])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed + 1)
    fold_of = {}
    for fold, (train_pos, valid_pos) in enumerate(splitter.split(np.zeros(len(development)), y, groups), 1):
        train_samples = [development[int(i)] for i in train_pos]
        valid_samples = [development[int(i)] for i in valid_pos]
        require_two_classes(train_samples, f"fold {fold} training set")
        require_two_classes(valid_samples, f"fold {fold} validation set")
        for pos in valid_pos:
            fold_of[int(pos)] = fold

    def assignment_rows(subset, subset_name):
        return [
            {
                "cancer_type": cancer,
                "set": subset_name,
                "cv_fold": fold_of[pos] if subset_name == "development" else "",
                "cell_id": sample["cell_id"],
                "drug_id": sample["drug_id"],
                "drug_name": sample["drug_name"],
                "true_label": sample["label"],
                "drug_effective_pair_count": len(samples_by_drug[sample["drug_id"]]),
            }
            for pos, sample in enumerate(subset)
        ]

    rows = assignment_rows(development, "development") + assignment_rows(
        independent_test, "independent_test"
    )

    assignment_path = output_dir / "drug_holdout_split_assignments.csv"
    with assignment_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"DRUG_HOLDOUT_SPLIT_COMPLETE cancer={cancer}")
    print(f"development: {len(development)} pairs, {len(development_drug_ids)} drugs")
    print(f"independent_test: {len(independent_test)} pairs, {len(test_drug_ids)} drugs")
    print(f"assignment: {assignment_path}")


if __name__ == "__main__":
    main()
