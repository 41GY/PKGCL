import argparse
import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path


CANCER = "TEST"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

csv.field_size_limit(2**31 - 1)
MISSING = {"", "na", "nan", "null", "none", "n/a"}
META_ROWS = {"COSMIC_ID", "DRUG_NAME", "LN_IC50", "ModelID", "CCLEName"}


def is_missing(value):
    return value is None or value.strip().lower() in MISSING


def as_float(value):
    if is_missing(value):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def stable_drug_id(drug_name):
    digest = hashlib.sha256(drug_name.strip().casefold().encode("utf-8")).hexdigest()[:12]
    return f"DRUG_{digest.upper()}"


def read_metadata(reader, path, expected_columns):
    metadata = {}
    while len(metadata) < len(META_ROWS):
        row = next(reader, None)
        if row is None:
            raise ValueError(f"{path}: metadata rows are incomplete")
        if len(row) != expected_columns:
            raise ValueError(f"{path}: metadata row has {len(row)} columns, expected {expected_columns}")
        if row[0] not in META_ROWS:
            raise ValueError(f"{path}: unexpected row {row[0]!r} before metadata was complete")
        metadata[row[0]] = row[1:]
    return metadata


def groups(values):
    result = defaultdict(list)
    for index, value in enumerate(values):
        if not is_missing(value):
            result[value.strip()].append(index)
    return result


def write_rows(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Split PKG-UCL wide matrices into modeling inputs.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--cancer", default=CANCER, help="Cancer CSV filename stem, such as PAAD")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir = args.output_dir or (PROJECT_ROOT / "phase1" / "output" / args.cancer if args.cancer else PROJECT_ROOT / "phase1" / "output")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.cancer and Path(args.cancer).name != args.cancer:
        raise ValueError("--cancer must be a filename stem, not a path")
    files = [args.data_dir / f"{args.cancer}.csv"] if args.cancer else sorted(args.data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {args.data_dir}")
    if args.cancer and not files[0].is_file():
        raise FileNotFoundError(files[0])

    expression_labels = fingerprint_labels = None
    cell_expression, drug_features, responses = {}, {}, []

    for path in files:
        cancer_type = path.stem
        print(f"Processing {path.name}...", flush=True)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            metadata = read_metadata(reader, path, len(header))
            model_ids = [value.strip() for value in metadata["ModelID"]]
            cosmic_ids = [value.strip() for value in metadata["COSMIC_ID"]]
            cell_ids = [model or cosmic for model, cosmic in zip(model_ids, cosmic_ids)]
            drug_names = [value.strip() for value in metadata["DRUG_NAME"]]
            cell_columns, drug_columns = groups(cell_ids), groups(drug_names)
            local_cell_expression = {cell_id: [] for cell_id in cell_columns}
            local_drug_features = {drug_name: [] for drug_name in drug_columns}
            local_expression_labels, local_fingerprint_labels = [], []

            for line_number, row in enumerate(reader, start=len(META_ROWS) + 2):
                if len(row) != len(header):
                    raise ValueError(f"{path}: row {line_number} has {len(row)} columns, expected {len(header)}")
                label, values = row[0], row[1:]
                if label.startswith("feature_") and label.endswith("_drug"):
                    local_fingerprint_labels.append(label)
                    for drug_name, indices in drug_columns.items():
                        canonical = values[indices[0]]
                        if any(values[index] != canonical for index in indices[1:]):
                            raise ValueError(f"{path}: drug fingerprint differs across cells for {drug_name!r}")
                        local_drug_features[drug_name].append(canonical)
                elif label.endswith("_exp"):
                    local_expression_labels.append(label[:-4])
                    for cell_id, indices in cell_columns.items():
                        canonical = values[indices[0]]
                        if any(values[index] != canonical for index in indices[1:]):
                            raise ValueError(f"{path}: expression differs across drugs for cell {cell_id!r}")
                        local_cell_expression[cell_id].append(canonical)

        if expression_labels is None:
            expression_labels = local_expression_labels
        elif expression_labels != local_expression_labels:
            raise ValueError(f"{path}: expression gene names or order differ from previous files")
        if fingerprint_labels is None:
            fingerprint_labels = local_fingerprint_labels
        elif fingerprint_labels != local_fingerprint_labels:
            raise ValueError(f"{path}: fingerprint feature names or order differ from previous files")

        for cell_id, vector in local_cell_expression.items():
            key = (cell_id, cancer_type)
            if key in cell_expression and cell_expression[key] != vector:
                raise ValueError(f"{path}: conflicting expression vector for {key!r}")
            cell_expression[key] = vector
        for drug_name, vector in local_drug_features.items():
            if drug_name in drug_features and drug_features[drug_name] != vector:
                raise ValueError(f"{path}: conflicting fingerprint vector for {drug_name!r}")
            drug_features.setdefault(drug_name, vector)

        replicate_numbers = defaultdict(int)
        for cell_id, drug_name, value in zip(cell_ids, drug_names, map(as_float, metadata["LN_IC50"])):
            if cell_id and drug_name:
                replicate_numbers[(cell_id, drug_name)] += 1
                responses.append({
                    "cell_id": cell_id,
                    "drug_name": drug_name,
                    "cancer_type": cancer_type,
                    "replicate_id": replicate_numbers[(cell_id, drug_name)],
                    "ln_ic50": value,
                })

    drug_ids = {name: stable_drug_id(name) for name in sorted(drug_features)}
    if len(set(drug_ids.values())) != len(drug_ids):
        raise RuntimeError("Synthetic drug ID collision detected")

    write_rows(
        args.output_dir / "cell_expression.csv",
        ["cell_id", "cancer_type", *expression_labels],
        ([cell_id, cancer_type, *vector] for (cell_id, cancer_type), vector in sorted(cell_expression.items(), key=lambda item: (item[0][1], item[0][0]))),
    )
    write_rows(
        args.output_dir / "drug_features.csv",
        ["drug_id", "drug_name", *fingerprint_labels],
        ([drug_ids[drug_name], drug_name, *drug_features[drug_name]] for drug_name in sorted(drug_features)),
    )
    write_rows(
        args.output_dir / "response_pairs.csv",
        ["cell_id", "drug_id", "cancer_type", "replicate_id", "ln_ic50"],
        ([row["cell_id"], drug_ids[row["drug_name"]], row["cancer_type"], row["replicate_id"], row["ln_ic50"]] for row in responses),
    )
    print(f"Wrote {len(cell_expression)} cells, {len(drug_features)} drugs, and {len(responses)} response observations.")


if __name__ == "__main__":
    main()
