import argparse
import csv
import math
import subprocess
from pathlib import Path


CANCER = "TEST"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def validate_expression(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header[:2] != ["cell_id", "cancer_type"]:
            raise ValueError("Expression file must start with cell_id,cancer_type")
        genes = header[2:]
        if len(genes) != len(set(genes)):
            raise ValueError("Expression matrix contains duplicate gene symbols")
        identifiers = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"{path}: row {line_number} has {len(row)} columns, expected {len(header)}")
            try:
                values = [float(value) for value in row[2:]]
            except ValueError as error:
                raise ValueError(f"{path}: row {line_number} contains a non-numeric expression value") from error
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}: row {line_number} contains a non-finite expression value")
            identifiers.append(row[:2])
    if len(identifiers) < 2:
        raise ValueError("At least two cells are required for GSVA")
    return identifiers


def validate_gsva(path, expected_identifiers):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header[:2] != ["cell_id", "cancer_type"] or len(header) != 52:
            raise ValueError("GSVA output must contain cell_id, cancer_type, and 50 pathways")
        identifiers = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"{path}: row {line_number} has {len(row)} columns, expected {len(header)}")
            try:
                scores = [float(value) for value in row[2:]]
            except ValueError as error:
                raise ValueError(f"{path}: row {line_number} contains a non-numeric GSVA score") from error
            if not all(math.isfinite(score) for score in scores):
                raise ValueError(f"{path}: row {line_number} contains a non-finite GSVA score")
            identifiers.append(row[:2])
    if identifiers != expected_identifiers:
        raise ValueError("GSVA output cells differ from the expression input")


def main():
    parser = argparse.ArgumentParser(description="Generate Hallmark GSVA features.")
    parser.add_argument("--cancer", default=CANCER, help="Use the Phase 1 output for one cancer")
    parser.add_argument("--expression", type=Path, default=None)
    parser.add_argument("--gmt", type=Path, default=PROJECT_ROOT / "data" / "hallmark.gmt")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    args.expression = args.expression or (PROJECT_ROOT / "phase1" / "output" / args.cancer / "cell_expression.csv")
    args.output_dir = args.output_dir or (PROJECT_ROOT / "phase2" / "output" / args.cancer)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    identifiers = validate_expression(args.expression)
    output_path = args.output_dir / "cell_gsva.csv"
    subprocess.run(
        [
            args.rscript,
            str(Path(__file__).with_name("run_gsva.R")),
            str(args.expression.relative_to(PROJECT_ROOT)),
            str(args.gmt.relative_to(PROJECT_ROOT)),
            str(output_path.relative_to(PROJECT_ROOT)),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    validate_gsva(output_path, identifiers)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
