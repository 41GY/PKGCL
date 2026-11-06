import csv
import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

CANCER = "TEST"

LABEL_PATH = PROJECT_ROOT / "phase3" / "output" / CANCER / "final_pair_labels.csv"

CELL_PATH = PROJECT_ROOT / "phase2" / "output" / CANCER / "cell_gsva.csv"

DRUG_PATH = PROJECT_ROOT / "phase1" / "output" / CANCER / "drug_features.csv"

GRAPH_PATH = PROJECT_ROOT / "phase4" / "output" / CANCER / "pathway_edges.csv"

OUT = HERE / "IG_analysis" / CANCER

MAIN_SCRIPT_PATH = HERE / "PKGCL.py"


FIXED_HYPERPARAMETERS = {
    "seed": 0,
    "learning_rate": 0.01,
    "weight_decay": 0.001,
    "dropout": 0.5,
    "lambda_cl": 0.2,
    "batch_size": 512,
    "mc_draws": 5,
    "scale": 5.0,
}

FIXED_EPOCHS = 122

IG_BASELINE = "zero"

IG_STEPS = 128

IG_BATCH_SIZE = 128


def load_main_module():
    if not MAIN_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Main PKGUCL script was not found: {MAIN_SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location(
        "pkgucl_main_model",
        MAIN_SCRIPT_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import main PKGUCL script: {MAIN_SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_pathway_names(path):
    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:
        header = next(csv.reader(f))

    if header[:2] != ["cell_id", "cancer_type"]:
        raise ValueError(
            "Unexpected cell_gsva.csv columns. "
            "Expected cell_id and cancer_type as the first two columns."
        )

    pathway_names = header[2:]

    if len(pathway_names) != 50:
        raise ValueError(f"Expected 50 GSVA pathways, found {len(pathway_names)}.")

    return pathway_names


def train_full_cohort_model(
    core,
    train_data,
    hp,
    epochs,
    adjacency,
    device,
):
    """Train PKGUCL once on every available sample for IG analysis."""
    core.set_seed(hp["seed"])

    adjacency = adjacency.to(device)

    model = core.PKGUCL(dropout=hp["dropout"]).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
    )

    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        metrics = core.train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            threshold=None,
            hp=hp,
            adj=adjacency,
            device=device,
        )

        history.append(
            {
                "epoch": epoch,
                **metrics,
            }
        )

        print(
            f"Epoch {epoch}/{epochs}: "
            f"loss={metrics['train_loss']:.6f}, "
            f"bce={metrics['train_bce']:.6f}, "
            f"contrastive={metrics['train_contrastive']:.6f}",
            flush=True,
        )

    return model, history


def make_ig_baseline(cell):
    if IG_BASELINE == "zero":
        return torch.zeros_like(cell)

    if IG_BASELINE == "cohort_mean":
        return cell.mean(
            dim=0,
            keepdim=True,
        ).expand_as(cell)

    raise ValueError(f"Unknown IG_BASELINE: {IG_BASELINE}")


def integrated_gradients(
    model,
    cell,
    drug,
    adjacency,
    device,
):
    """
    Attribute the positive-class logit to the 50 cell GSVA inputs.

    The drug descriptor and pathway graph are held fixed. Positive values
    increase the positive-class logit relative to the selected baseline;
    negative values decrease it.
    """
    model.eval()

    cell = cell.to(device)

    drug = drug.to(device)

    adjacency = adjacency.to(device)

    baseline = make_ig_baseline(cell)

    input_difference = cell - baseline

    total_gradient = torch.zeros_like(cell)

    for step in range(
        1,
        IG_STEPS + 1,
    ):
        alpha = step / IG_STEPS

        interpolated_cell = (
            (baseline + alpha * input_difference).detach().requires_grad_(True)
        )

        logits = model(
            interpolated_cell,
            drug,
            adjacency,
        )

        gradients = torch.autograd.grad(
            outputs=logits.sum(),
            inputs=interpolated_cell,
            create_graph=False,
            retain_graph=False,
        )[0]

        total_gradient += gradients

    attribution = input_difference * total_gradient / IG_STEPS

    with torch.no_grad():
        input_logit = model(
            cell,
            drug,
            adjacency,
        )

        baseline_logit = model(
            baseline,
            drug,
            adjacency,
        )

    return (
        attribution.detach().cpu(),
        input_logit.detach().cpu(),
        baseline_logit.detach().cpu(),
    )


def compute_ig_for_cohort(
    model,
    data,
    adjacency,
    device,
):
    attribution_parts = []
    logit_parts = []
    baseline_logit_parts = []

    total_samples = len(data[2])

    for start in range(
        0,
        total_samples,
        IG_BATCH_SIZE,
    ):
        stop = min(
            start + IG_BATCH_SIZE,
            total_samples,
        )

        attribution, logits, baseline_logits = integrated_gradients(
            model=model,
            cell=data[0][start:stop],
            drug=data[1][start:stop],
            adjacency=adjacency,
            device=device,
        )

        attribution_parts.append(attribution)

        logit_parts.append(logits)

        baseline_logit_parts.append(baseline_logits)

        print(
            f"IG: {stop}/{total_samples} samples completed.",
            flush=True,
        )

    return (
        torch.cat(
            attribution_parts,
            dim=0,
        ).numpy(),
        torch.cat(
            logit_parts,
            dim=0,
        ).numpy(),
        torch.cat(
            baseline_logit_parts,
            dim=0,
        ).numpy(),
    )


def build_sample_attribution_rows(
    keys,
    samples,
    pathway_names,
    attributions,
    logits,
    baseline_logits,
):
    rows = []

    probabilities = 1.0 / (1.0 + np.exp(-logits))

    for index, key in enumerate(keys):
        sample = samples[key]
        attribution = attributions[index]
        logit_delta = logits[index] - baseline_logits[index]
        attribution_sum = float(attribution.sum())

        row = {
            "cancer_type": sample["cancer_type"],
            "cell_id": sample["cell_id"],
            "drug_id": sample["drug_id"],
            "drug_name": sample["drug_name"],
            "true_label": sample["label"],
            "prediction_logit": float(logits[index]),
            "prediction_probability": float(probabilities[index]),
            "baseline_logit": float(baseline_logits[index]),
            "logit_delta": float(logit_delta),
            "ig_attribution_sum": attribution_sum,
            "ig_completeness_error": float(attribution_sum - logit_delta),
        }

        for pathway_name, value in zip(
            pathway_names,
            attribution,
        ):
            row[pathway_name] = float(value)

        rows.append(row)

    return rows


def build_global_pathway_rows(
    pathway_names,
    attributions,
    labels,
):
    rows = []

    labels = np.asarray(
        labels,
        dtype=int,
    )

    for index, pathway_name in enumerate(pathway_names):
        values = attributions[:, index]
        label_zero = values[labels == 0]
        label_one = values[labels == 1]

        rows.append(
            {
                "pathway": pathway_name,
                "mean_ig": float(values.mean()),
                "mean_absolute_ig": float(np.abs(values).mean()),
                "mean_ig_label_0": (
                    float(label_zero.mean()) if len(label_zero) else None
                ),
                "mean_ig_label_1": (
                    float(label_one.mean()) if len(label_one) else None
                ),
                "mean_absolute_ig_label_0": (
                    float(np.abs(label_zero).mean()) if len(label_zero) else None
                ),
                "mean_absolute_ig_label_1": (
                    float(np.abs(label_one).mean()) if len(label_one) else None
                ),
            }
        )

    return sorted(
        rows,
        key=lambda row: row["mean_absolute_ig"],
        reverse=True,
    )


def main():
    torch.set_num_threads(4)

    torch.set_num_interop_threads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    print(
        "PKGUCL full-training Integrated Gradients analysis",
        flush=True,
    )

    print(
        f"Cancer type: {CANCER}",
        flush=True,
    )

    print(
        f"Device: {device}",
        flush=True,
    )

    print(
        f"Fixed epochs: {FIXED_EPOCHS}",
        flush=True,
    )

    print(
        f"IG baseline: {IG_BASELINE}",
        flush=True,
    )

    print(
        f"IG steps: {IG_STEPS}",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    core = load_main_module()

    samples = core.load_samples(
        LABEL_PATH,
        CELL_PATH,
        DRUG_PATH,
    )

    all_keys = [key for key in samples if key[0] == CANCER]

    if not all_keys:
        raise ValueError(f"No usable non-tie samples were found for {CANCER}.")

    pathway_names = load_pathway_names(CELL_PATH)

    train_data = core.tensorize(
        all_keys,
        samples,
    )

    adjacency = core.load_graph(GRAPH_PATH)

    print(
        f"Full-training sample count: {len(all_keys)}",
        flush=True,
    )

    print("Training full-cohort PKGUCL model...", flush=True)
    model, _ = train_full_cohort_model(
        core=core,
        train_data=train_data,
        hp=FIXED_HYPERPARAMETERS,
        epochs=FIXED_EPOCHS,
        adjacency=adjacency,
        device=device,
    )
    print("Computing Integrated Gradients for all samples...", flush=True)
    attributions, logits, baseline_logits = compute_ig_for_cohort(
        model=model,
        data=train_data,
        adjacency=adjacency,
        device=device,
    )

    sample_rows = build_sample_attribution_rows(
        keys=all_keys,
        samples=samples,
        pathway_names=pathway_names,
        attributions=attributions,
        logits=logits,
        baseline_logits=baseline_logits,
    )

    global_rows = build_global_pathway_rows(
        pathway_names=pathway_names,
        attributions=attributions,
        labels=train_data[2].numpy(),
    )

    core.write(
        OUT / "ig_sample_attributions.csv",
        sample_rows,
    )

    core.write(
        OUT / "ig_global_pathway_importance.csv",
        global_rows,
    )

    print(
        "=" * 70,
        flush=True,
    )

    print(
        "IG analysis completed.",
        flush=True,
    )

    print(
        "Outputs:",
        flush=True,
    )

    print("1. ig_sample_attributions.csv", flush=True)
    print("2. ig_global_pathway_importance.csv", flush=True)

    print(
        "=" * 70,
        flush=True,
    )


if __name__ == "__main__":
    main()
