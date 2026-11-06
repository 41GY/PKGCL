import csv
import hashlib
import random
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch import nn


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
CANCER = "TEST"
OUT = HERE / "results_drug_holdout" / CANCER

LABEL_PATH = PROJECT_ROOT / "phase3" / "output" / CANCER / "final_pair_labels.csv"

CELL_PATH = PROJECT_ROOT / "phase2" / "output" / CANCER / "cell_gsva.csv"

DRUG_PATH = PROJECT_ROOT / "phase1" / "output" / CANCER / "drug_features.csv"

GRAPH_PATH = PROJECT_ROOT / "phase4" / "output" / CANCER / "pathway_edges.csv"

SPLIT_PATH = HERE / "results" / CANCER / "drug_holdout_split_assignments.csv"

N_FOLDS = 5

EVAL_MC_DRAWS = 10


FULL_PARAM_DICT = {
    "seed": [0],
    "learning_rate": [1e-2],
    "weight_decay": [0.001],
    "dropout": [0.1, 0.2, 0.5],
    "lambda_cl": [0.1, 0.2, 0.3, 0.4, 0.5],
    "batch_size": [512],
    "mc_draws": [5],
    "scale": [5.0],
    "max_epochs": [1000],
}


def make_param_grid(param_dict):
    keys = list(param_dict.keys())

    return [
        dict(zip(keys, values))
        for values in product(*(param_dict[key] for key in keys))
    ]


FULL_PARAM_GRID = make_param_grid(FULL_PARAM_DICT)


def stable_seed(base, *parts):
    text = "|".join(
        map(
            str,
            (
                base,
                *parts,
            ),
        )
    )

    return int(
        hashlib.sha256(text.encode()).hexdigest()[:8],
        16,
    ) % (2**31 - 1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def read_features(
    path,
    id_col,
    meta_n,
):
    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.reader(f)

        header = next(reader)

        assert header[0] == id_col

        values = {}
        metadata = {}

        for row in reader:
            values[row[0]] = np.asarray(
                row[meta_n:],
                dtype=np.float32,
            )

            metadata[row[0]] = row[:meta_n]

    return (
        values,
        metadata,
    )


def load_samples(
    labels,
    cells_path,
    drugs_path,
):
    cells, _ = read_features(
        cells_path,
        "cell_id",
        2,
    )

    drugs, _ = read_features(
        drugs_path,
        "drug_id",
        2,
    )

    samples = {}

    with labels.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:
        for row in csv.DictReader(f):

            if row["is_tie"] == "1":
                continue

            key = (
                row["cancer_type"],
                row["cell_id"],
                row["drug_id"],
            )

            samples[key] = {
                "cancer_type": row["cancer_type"],
                "cell_id": row["cell_id"],
                "drug_id": row["drug_id"],
                "drug_name": row["drug_name"],
                "label": int(row["response_label"]),
                "cell": cells[row["cell_id"]],
                "drug": drugs[row["drug_id"]],
            }

    return samples


def load_graph(path):
    adjacency = torch.eye(
        50,
        dtype=torch.bool,
    )

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:
        for row in csv.DictReader(f):

            adjacency[
                int(row["source_index"]),
                int(row["target_index"]),
            ] = True

    return adjacency


def write(
    path,
    rows,
):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def tensorize(
    keys,
    samples,
):
    cell = torch.tensor(
        np.stack([samples[key]["cell"] for key in keys]),
        dtype=torch.float32,
    )

    drug = torch.tensor(
        np.stack([samples[key]["drug"] for key in keys]),
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [samples[key]["label"] for key in keys],
        dtype=torch.float32,
    )

    return (
        cell,
        drug,
        labels,
    )


class GATLayer(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        heads=1,
        dropout=0.25,
    ):
        super().__init__()

        self.h = heads
        self.o = output_dim

        self.lin = nn.Linear(
            input_dim,
            heads * output_dim,
            bias=False,
        )

        self.a = nn.Parameter(
            torch.empty(
                heads,
                output_dim,
            )
        )

        self.b = nn.Parameter(
            torch.empty(
                heads,
                output_dim,
            )
        )

        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.lin.weight)

        nn.init.xavier_uniform_(self.a)

        nn.init.xavier_uniform_(self.b)

    def forward(
        self,
        x,
        adj,
    ):
        q = self.lin(x).view(
            -1,
            50,
            self.h,
            self.o,
        )

        score_a = (
            (q * self.a)
            .sum(-1)
            .permute(
                0,
                2,
                1,
            )
            .unsqueeze(-1)
        )

        score_b = (
            (q * self.b)
            .sum(-1)
            .permute(
                0,
                2,
                1,
            )
            .unsqueeze(-2)
        )

        scores = score_a + score_b

        attention = nn.functional.leaky_relu(
            scores,
            negative_slope=0.2,
        )

        attention = attention.masked_fill(
            ~adj.view(
                1,
                1,
                50,
                50,
            ),
            float("-inf"),
        )

        attention = torch.softmax(
            attention,
            dim=-1,
        )

        attention = self.drop(attention)

        output = attention @ q.permute(
            0,
            2,
            1,
            3,
        )

        output = output.permute(
            0,
            2,
            1,
            3,
        ).reshape(
            len(x),
            50,
            self.h * self.o,
        )

        return (
            output,
            attention,
        )


class PKGUCL(nn.Module):

    def __init__(
        self,
        dropout=0.25,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(
                50,
                64,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                64,
                64,
            ),
        )

        self.g1 = GATLayer(
            input_dim=1,
            output_dim=16,
            heads=4,
            dropout=dropout,
        )

        self.g2 = GATLayer(
            input_dim=64,
            output_dim=64,
            heads=1,
            dropout=dropout,
        )

        self.gate = nn.Sequential(
            nn.Linear(
                128,
                32,
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                1,
            ),
        )

        self.drug = nn.Sequential(
            nn.Linear(
                2048,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                128,
                64,
            ),
        )

        self.head = nn.Sequential(
            nn.Linear(
                128,
                64,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                64,
                1,
            ),
        )

    def rep(
        self,
        cell,
        drug,
        adj,
        return_interpretation=False,
    ):
        zm = self.mlp(cell)

        g1_output, g1_attention = self.g1(
            cell.unsqueeze(-1),
            adj,
        )

        zg = torch.relu(g1_output)

        g2_output, g2_attention = self.g2(
            zg,
            adj,
        )

        zg = g2_output.mean(dim=1)

        gate = torch.sigmoid(
            self.gate(
                torch.cat(
                    [
                        zm,
                        zg,
                    ],
                    dim=1,
                )
            )
        )

        cell_rep = gate * zm + (1.0 - gate) * zg

        drug_rep = self.drug(drug)

        combined = torch.cat(
            [
                cell_rep,
                drug_rep,
            ],
            dim=1,
        )

        # Parameter-free cell-drug interaction embedding.
        interaction_rep = cell_rep * drug_rep

        representations = (
            combined,
            cell_rep,
            drug_rep,
            interaction_rep,
        )

        if return_interpretation:
            return (
                *representations,
                {
                    "gate": gate.squeeze(1),
                    "g1_attention": g1_attention,
                    "g2_attention": g2_attention,
                },
            )

        return representations

    def forward(
        self,
        cell,
        drug,
        adj,
        return_rep=False,
        return_interpretation=False,
    ):
        if return_rep and return_interpretation:
            raise ValueError(
                "return_rep and return_interpretation cannot both be True."
            )

        if return_interpretation:
            (
                combined,
                _,
                _,
                interaction_rep,
                interpretation,
            ) = self.rep(
                cell,
                drug,
                adj,
                return_interpretation=True,
            )
        else:
            (
                combined,
                _,
                _,
                interaction_rep,
            ) = self.rep(
                cell,
                drug,
                adj,
            )

        logits = self.head(combined).squeeze(1)

        if return_rep:
            return (
                logits,
                interaction_rep,
            )

        if return_interpretation:
            return (
                logits,
                interpretation,
            )

        return logits


def similarity_threshold(
    cell,
    cell_ids,
):
    by_id = {cell_id: cell[i] for i, cell_id in enumerate(cell_ids)}

    unique = torch.stack([by_id[cell_id] for cell_id in sorted(by_id)])

    if len(unique) < 2:
        return 0.0

    normalized = nn.functional.normalize(
        unique,
        dim=1,
    )

    similarity = normalized @ normalized.T

    mask = ~torch.eye(
        len(unique),
        dtype=torch.bool,
    )

    return float(torch.median(similarity[mask]).item())


def adaptive_interaction_contrastive(
    z,
    cell_x,
    drug_x,
    y,
    uncertainty_confidence,
    temperature=0.2,
):
    """
    Continuous adaptive interaction contrastive learning.

    Positive weight:
        0.5 + 0.5 * pair_similarity

    Negative weight:
        1.0 + 0.5 * pair_similarity

    Pair similarity combines raw GSVA similarity and raw drug
    fingerprint similarity using their geometric mean.

    Pair-level uncertainty confidence:
        sqrt(confidence_i * confidence_j)
    """

    if len(y) <= 1:
        return z.sum() * 0.0

    z_normalized = nn.functional.normalize(
        z,
        dim=1,
    )

    similarity_logits = z_normalized @ z_normalized.T / temperature

    cell_normalized = nn.functional.normalize(
        cell_x,
        dim=1,
    )

    cell_similarity = cell_normalized @ cell_normalized.T

    cell_similarity = ((cell_similarity + 1.0) / 2.0).clamp(
        0.0,
        1.0,
    )

    drug_normalized = nn.functional.normalize(
        drug_x,
        dim=1,
    )

    drug_similarity = drug_normalized @ drug_normalized.T

    drug_similarity = ((drug_similarity + 1.0) / 2.0).clamp(
        0.0,
        1.0,
    )

    # Both cell and drug similarity must be high to obtain
    # a high interaction-level biological similarity.
    pair_similarity = torch.sqrt((cell_similarity * drug_similarity).clamp_min(1e-12))

    # Pair-level reliability from MC-dropout confidence.
    pair_confidence = torch.sqrt(
        (uncertainty_confidence[:, None] * uncertainty_confidence[None, :]).clamp_min(
            1e-12
        )
    )

    eye = torch.eye(
        len(y),
        dtype=torch.bool,
        device=y.device,
    )

    same_label = y[:, None] == y[None, :]

    positive_base_weight = 0.5 + 0.5 * pair_similarity

    negative_base_weight = 1.0 + 0.5 * pair_similarity

    positive_weights = torch.where(
        same_label,
        positive_base_weight,
        torch.zeros_like(positive_base_weight),
    )

    negative_weights = torch.where(
        ~same_label,
        negative_base_weight,
        torch.zeros_like(negative_base_weight),
    )

    positive_weights = (positive_weights * pair_confidence).masked_fill(
        eye,
        0.0,
    )

    negative_weights = (negative_weights * pair_confidence).masked_fill(
        eye,
        0.0,
    )

    pair_weights = positive_weights + negative_weights

    weighted_logits = similarity_logits + torch.log(pair_weights.clamp_min(1e-12))

    weighted_logits = weighted_logits.masked_fill(
        eye | (pair_weights <= 0),
        float("-inf"),
    )

    log_denominator = torch.logsumexp(
        weighted_logits,
        dim=1,
    )

    positive_mass = positive_weights.sum(dim=1)

    valid_anchor = positive_mass > 0

    if not valid_anchor.any():
        return z.sum() * 0.0

    per_anchor_loss = -(
        positive_weights * (similarity_logits - log_denominator[:, None])
    ).sum(dim=1) / positive_mass.clamp_min(1e-12)

    return per_anchor_loss[valid_anchor].mean()


@torch.no_grad()
def mc(
    model,
    data,
    adjacency,
    n,
):
    model.train()

    device = adjacency.device

    cell = data[0].to(device)

    drug = data[1].to(device)

    probabilities = torch.stack(
        [
            torch.sigmoid(
                model(
                    cell,
                    drug,
                    adjacency,
                )
            )
            for _ in range(n)
        ]
    )

    return (
        probabilities.mean(dim=0),
        probabilities.var(
            dim=0,
            unbiased=False,
        ),
    )


def calculate_metrics(
    labels,
    probability,
):
    labels = np.asarray(labels).astype(int)

    probability = np.asarray(probability)

    predictions = (probability >= 0.5).astype(int)

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "A held-out CV fold contains only one class. "
            "ROC AUC cannot be calculated. "
            "Please regenerate split_assignments.csv using "
            "stratified 5-fold splitting."
        )

    auc = roc_auc_score(
        labels,
        probability,
    )

    auprc = average_precision_score(
        labels,
        probability,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    return {
        "auc": float(auc),
        "auprc": float(auprc),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }


def evaluate_mc(
    model,
    data,
    adj,
    device,
    draws,
    eval_seed=None,
):
    adj = adj.to(device)

    cpu_rng_state = torch.random.get_rng_state()

    cuda_rng_states = None

    if torch.cuda.is_available():
        cuda_rng_states = torch.cuda.get_rng_state_all()

    try:
        if eval_seed is not None:
            torch.manual_seed(eval_seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed(eval_seed)

                torch.cuda.manual_seed_all(eval_seed)

        probability, variance = mc(
            model,
            data,
            adj,
            draws,
        )

    finally:
        torch.random.set_rng_state(cpu_rng_state)

        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    probability = probability.detach().cpu().numpy()

    variance = variance.detach().cpu().numpy()

    labels = data[2].detach().cpu().numpy().astype(int)

    predictions = (probability >= 0.5).astype(int)

    metrics = calculate_metrics(
        labels,
        probability,
    )

    return (
        metrics,
        labels,
        probability,
        variance,
        predictions,
    )


def train_one_epoch(
    model,
    optimizer,
    train_data,
    threshold,
    hp,
    adj,
    device,
):
    model.train()

    # No hard biological threshold is used by the new
    # continuous contrastive objective.
    _ = threshold

    train_cell_device = train_data[0].to(device)

    train_drug_device = train_data[1].to(device)

    with torch.no_grad():
        probabilities = torch.stack(
            [
                torch.sigmoid(
                    model(
                        train_cell_device,
                        train_drug_device,
                        adj,
                    )
                )
                for _ in range(hp["mc_draws"])
            ]
        )

        variance = probabilities.var(
            dim=0,
            unbiased=False,
        )

        # Raw MC-dropout confidence in (0, 1].
        confidence = 1.0 / (1.0 + hp["scale"] * variance)

        # Mean-normalized sample weights are used for BCE.
        weights = confidence / confidence.mean().clamp_min(1e-8)

    order = torch.randperm(len(train_data[2]))

    epoch_total_loss = 0.0
    epoch_bce_loss = 0.0
    epoch_cl_loss = 0.0

    total_samples = 0

    for start in range(
        0,
        len(order),
        hp["batch_size"],
    ):
        index = order[start : start + hp["batch_size"]]

        cell = train_data[0][index].to(device)

        drug = train_data[1][index].to(device)

        labels = train_data[2][index].to(device)

        device_index = index.to(weights.device)

        batch_weights = weights[device_index]

        batch_confidence = confidence[device_index]

        logits, interaction_rep = model(
            cell,
            drug,
            adj,
            return_rep=True,
        )

        sample_bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )

        bce = (sample_bce * batch_weights).sum() / batch_weights.sum().clamp_min(1e-8)

        contrastive = adaptive_interaction_contrastive(
            z=interaction_rep,
            cell_x=cell,
            drug_x=drug,
            y=labels,
            uncertainty_confidence=batch_confidence,
        )

        loss = bce + hp["lambda_cl"] * contrastive

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_n = len(labels)

        epoch_total_loss += float(loss.item()) * batch_n

        epoch_bce_loss += float(bce.item()) * batch_n

        epoch_cl_loss += float(contrastive.item()) * batch_n

        total_samples += batch_n

    return {
        "train_loss": epoch_total_loss / total_samples,
        "train_bce": epoch_bce_loss / total_samples,
        "train_contrastive": epoch_cl_loss / total_samples,
    }


def train_full_fold(
    train_data,
    train_cell_ids,
    fold_test_data,
    hp,
    adj,
    device,
    eval_seed,
):
    set_seed(hp["seed"])

    adj = adj.to(device)

    model = PKGUCL(dropout=hp["dropout"]).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
    )

    # Continuous similarity replaces the previous hard
    # median threshold. Keep the variable/signature only to
    # minimize unrelated changes to the CV framework.
    threshold = None
    _ = train_cell_ids

    epoch_results = []

    for epoch in range(
        1,
        hp["max_epochs"] + 1,
    ):
        train_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            threshold=threshold,
            hp=hp,
            adj=adj,
            device=device,
        )

        (
            fold_metrics,
            _,
            _,
            _,
            _,
        ) = evaluate_mc(
            model=model,
            data=fold_test_data,
            adj=adj,
            device=device,
            draws=EVAL_MC_DRAWS,
            eval_seed=eval_seed,
        )

        epoch_results.append(
            {
                "epoch": epoch,
                "auc": fold_metrics["auc"],
                "accuracy": fold_metrics["accuracy"],
                "f1": fold_metrics["f1"],
                "train_loss": train_metrics["train_loss"],
                "train_bce": train_metrics["train_bce"],
                "train_contrastive": train_metrics["train_contrastive"],
            }
        )

    return (
        model,
        epoch_results,
    )


def train_final_model(
    train_data,
    hp,
    epochs,
    adj,
    device,
    seed,
):
    """Train once on all development samples without reading the test set."""
    set_seed(seed)

    adj = adj.to(device)

    model = PKGUCL(dropout=hp["dropout"]).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
    )

    for _ in range(epochs):
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            threshold=None,
            hp=hp,
            adj=adj,
            device=device,
        )

    return model


def read_split_assignments(
    path,
    samples,
):
    groups = defaultdict(lambda: defaultdict(list))

    foldmap = {}

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            key = (
                row["cancer_type"],
                row["cell_id"],
                row["drug_id"],
            )

            if key not in samples:
                continue

            groups[row["cancer_type"]][row["set"]].append(key)

            if row.get("cv_fold", ""):
                foldmap[key] = int(row["cv_fold"])

    return (
        groups,
        foldmap,
    )


def validate_cv_assignment(
    cancer,
    development_keys,
    foldmap,
    samples,
):
    missing = [key for key in development_keys if key not in foldmap]

    if missing:
        raise ValueError(
            f"{cancer}: "
            f"{len(missing)} development samples "
            "do not have cv_fold assignments."
        )

    folds_present = sorted({foldmap[key] for key in development_keys})

    expected = list(
        range(
            1,
            N_FOLDS + 1,
        )
    )

    if folds_present != expected:
        raise ValueError(
            f"{cancer}: expected CV folds "
            f"{expected}, but found "
            f"{folds_present}."
        )

    for fold in expected:
        fold_keys = [key for key in development_keys if foldmap[key] == fold]

        labels = [samples[key]["label"] for key in fold_keys]

        if len(set(labels)) < 2:
            raise ValueError(
                f"{cancer}, fold {fold}: "
                "only one response class is present. "
                "AUC cannot be calculated."
            )


def aggregate_epochs(
    cancer,
    grid_id,
    hp,
    fold_rows,
):
    by_epoch = defaultdict(list)

    for row in fold_rows:
        by_epoch[row["epoch"]].append(row)

    mean_rows = []

    for epoch in sorted(by_epoch):
        rows = by_epoch[epoch]

        if len(rows) != N_FOLDS:
            raise ValueError(
                f"{cancer}, grid {grid_id}, "
                f"epoch {epoch}: expected "
                f"{N_FOLDS} fold results, "
                f"got {len(rows)}."
            )

        auc_values = np.asarray(
            [row["auc"] for row in rows],
            dtype=float,
        )

        accuracy_values = np.asarray(
            [row["accuracy"] for row in rows],
            dtype=float,
        )

        f1_values = np.asarray(
            [row["f1"] for row in rows],
            dtype=float,
        )

        loss_values = np.asarray(
            [row["train_loss"] for row in rows],
            dtype=float,
        )

        bce_values = np.asarray(
            [row["train_bce"] for row in rows],
            dtype=float,
        )

        cl_values = np.asarray(
            [row["train_contrastive"] for row in rows],
            dtype=float,
        )

        mean_rows.append(
            {
                "cancer_type": cancer,
                "model": "PKGCL",
                "grid_id": grid_id,
                "epoch": epoch,
                "mean_auc": float(auc_values.mean()),
                "mean_accuracy": float(accuracy_values.mean()),
                "mean_f1": float(f1_values.mean()),
                "std_auc": float(auc_values.std(ddof=1)),
                "std_accuracy": float(accuracy_values.std(ddof=1)),
                "std_f1": float(f1_values.std(ddof=1)),
                "mean_train_loss": float(loss_values.mean()),
                "mean_train_bce": float(bce_values.mean()),
                "mean_train_contrastive": float(cl_values.mean()),
                **hp,
            }
        )

    return mean_rows


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
        f"Device: {device}",
        flush=True,
    )

    print(
        f"Number of parameter groups: " f"{len(FULL_PARAM_GRID)}",
        flush=True,
    )

    print(
        f"CV folds: {N_FOLDS}",
        flush=True,
    )

    print(
        f"Evaluation MC draws: " f"{EVAL_MC_DRAWS}",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    adj = load_graph(GRAPH_PATH).to(device)

    samples = load_samples(
        LABEL_PATH,
        CELL_PATH,
        DRUG_PATH,
    )

    (
        groups,
        foldmap,
    ) = read_split_assignments(
        SPLIT_PATH,
        samples,
    )

    all_fold_epoch_rows = []
    all_mean_epoch_rows = []
    best_epoch_rows = []
    independent_test_metric_rows = []
    independent_test_prediction_rows = []

    for cancer, group in sorted(groups.items()):
        development_keys = group["development"]

        independent_test_keys = group["independent_test"]

        if not development_keys:
            print(
                f"[SKIP] {cancer}: " "no development samples.",
                flush=True,
            )
            continue

        if not independent_test_keys:
            raise ValueError(
                f"{cancer}: no independent_test samples were found in "
                "drug_holdout_split_assignments.csv."
            )

        development_drug_ids = {key[2] for key in development_keys}

        independent_test_drug_ids = {key[2] for key in independent_test_keys}

        shared_drug_ids = development_drug_ids & independent_test_drug_ids

        if shared_drug_ids:
            raise ValueError(
                f"{cancer}: {len(shared_drug_ids)} drug(s) appear in both "
                "development and independent_test."
            )

        validate_cv_assignment(
            cancer,
            development_keys,
            foldmap,
            samples,
        )

        print(
            "\n" + "=" * 70,
            flush=True,
        )

        print(
            f"Cancer type: {cancer}",
            flush=True,
        )

        print(
            f"Development N: " f"{len(development_keys)}",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

        for grid_id, hp in enumerate(
            FULL_PARAM_GRID,
            start=1,
        ):
            print(
                "\n" + "-" * 70,
                flush=True,
            )

            print(
                f"[{cancer}] "
                f"Parameter group "
                f"{grid_id}/"
                f"{len(FULL_PARAM_GRID)}",
                flush=True,
            )

            print(
                hp,
                flush=True,
            )

            grid_fold_rows = []

            for fold in range(
                1,
                N_FOLDS + 1,
            ):
                fold_test_keys = [
                    key for key in development_keys if foldmap[key] == fold
                ]

                train_keys = [key for key in development_keys if foldmap[key] != fold]

                print(
                    f"  Fold {fold}: "
                    f"train={len(train_keys)}, "
                    f"test={len(fold_test_keys)}",
                    flush=True,
                )

                train_data = tensorize(
                    train_keys,
                    samples,
                )

                fold_test_data = tensorize(
                    fold_test_keys,
                    samples,
                )

                train_cell_ids = [key[1] for key in train_keys]

                eval_seed = stable_seed(
                    hp["seed"],
                    cancer,
                    grid_id,
                    fold,
                    "evaluation",
                )

                (
                    _,
                    fold_history,
                ) = train_full_fold(
                    train_data=train_data,
                    train_cell_ids=train_cell_ids,
                    fold_test_data=fold_test_data,
                    hp=hp,
                    adj=adj,
                    device=device,
                    eval_seed=eval_seed,
                )

                for history_row in fold_history:
                    row = {
                        "cancer_type": cancer,
                        "model": "PKGCL",
                        "grid_id": grid_id,
                        "fold": fold,
                        "epoch": history_row["epoch"],
                        "train_n": len(train_keys),
                        "fold_test_n": len(fold_test_keys),
                        "auc": history_row["auc"],
                        "accuracy": history_row["accuracy"],
                        "f1": history_row["f1"],
                        "train_loss": history_row["train_loss"],
                        "train_bce": history_row["train_bce"],
                        "train_contrastive": history_row["train_contrastive"],
                        **hp,
                    }

                    all_fold_epoch_rows.append(row)

                    grid_fold_rows.append(row)

                best_fold_row = max(
                    fold_history,
                    key=lambda x: (
                        x["auc"],
                        -x["epoch"],
                    ),
                )

                print(
                    f"    Best fold AUC="
                    f"{best_fold_row['auc']:.4f} "
                    f"at epoch "
                    f"{best_fold_row['epoch']}",
                    flush=True,
                )

            mean_epoch_rows = aggregate_epochs(
                cancer=cancer,
                grid_id=grid_id,
                hp=hp,
                fold_rows=grid_fold_rows,
            )

            all_mean_epoch_rows.extend(mean_epoch_rows)

            best_mean_row = max(
                mean_epoch_rows,
                key=lambda x: (
                    x["mean_auc"],
                    -x["epoch"],
                ),
            )

            best_epoch_rows.append(
                {
                    "cancer_type": cancer,
                    "model": "PKGCL",
                    "grid_id": grid_id,
                    "best_epoch": best_mean_row["epoch"],
                    "best_mean_auc": best_mean_row["mean_auc"],
                    "auc_std_at_best_epoch": best_mean_row["std_auc"],
                    "mean_accuracy_at_best_epoch": best_mean_row["mean_accuracy"],
                    "accuracy_std_at_best_epoch": best_mean_row["std_accuracy"],
                    "mean_f1_at_best_epoch": best_mean_row["mean_f1"],
                    "f1_std_at_best_epoch": best_mean_row["std_f1"],
                    **hp,
                }
            )

            print(
                f"  >>> Grid {grid_id}: "
                f"best 5-fold mean AUC="
                f"{best_mean_row['mean_auc']:.4f} "
                f"at epoch="
                f"{best_mean_row['epoch']}",
                flush=True,
            )

            write(
                OUT / "full_cv_epoch_fold_metrics.csv",
                all_fold_epoch_rows,
            )

            write(
                OUT / "full_cv_epoch_mean_metrics.csv",
                all_mean_epoch_rows,
            )

            write(
                OUT / "full_cv_param_best_epoch.csv",
                best_epoch_rows,
            )

        cancer_best_rows = [
            row for row in best_epoch_rows if row["cancer_type"] == cancer
        ]

        cancer_best = max(
            cancer_best_rows,
            key=lambda x: (
                x["best_mean_auc"],
                -x["best_epoch"],
            ),
        )

        # The independent test set is only read here, after all CV choices
        # (parameter group and epoch) have been finalized.
        final_train_data = tensorize(
            development_keys,
            samples,
        )

        independent_test_data = tensorize(
            independent_test_keys,
            samples,
        )

        final_seed = stable_seed(
            cancer_best["seed"],
            cancer,
            "drug_holdout_final_training",
        )

        final_model = train_final_model(
            train_data=final_train_data,
            hp=cancer_best,
            epochs=cancer_best["best_epoch"],
            adj=adj,
            device=device,
            seed=final_seed,
        )

        (
            independent_metrics,
            true_labels,
            probabilities,
            variances,
            predictions,
        ) = evaluate_mc(
            model=final_model,
            data=independent_test_data,
            adj=adj,
            device=device,
            draws=EVAL_MC_DRAWS,
            eval_seed=stable_seed(
                cancer_best["seed"],
                cancer,
                "drug_holdout_independent_evaluation",
            ),
        )

        independent_test_metric_rows.append(
            {
                "cancer_type": cancer,
                "model": "PKGCL",
                "independent_test_n": len(independent_test_keys),
                "independent_test_drug_n": len(independent_test_drug_ids),
                "grid_id": cancer_best["grid_id"],
                "best_epoch": cancer_best["best_epoch"],
                "cv_mean_auc": cancer_best["best_mean_auc"],
                "cv_auc_std": cancer_best["auc_std_at_best_epoch"],
                "independent_test_auc": independent_metrics["auc"],
                "independent_test_auprc": independent_metrics["auprc"],
                "independent_test_accuracy": independent_metrics["accuracy"],
                "independent_test_f1": independent_metrics["f1"],
                "seed": cancer_best["seed"],
                "learning_rate": cancer_best["learning_rate"],
                "weight_decay": cancer_best["weight_decay"],
                "dropout": cancer_best["dropout"],
                "lambda_cl": cancer_best["lambda_cl"],
                "batch_size": cancer_best["batch_size"],
                "mc_draws": cancer_best["mc_draws"],
                "scale": cancer_best["scale"],
            }
        )

        for index, key in enumerate(independent_test_keys):
            independent_test_prediction_rows.append(
                {
                    "cancer_type": cancer,
                    "cell_id": key[1],
                    "drug_id": key[2],
                    "drug_name": samples[key]["drug_name"],
                    "true_label": int(true_labels[index]),
                    "probability": float(probabilities[index]),
                    "prediction": int(predictions[index]),
                    "mc_variance": float(variances[index]),
                }
            )

        print(
            "\n" + "=" * 70,
            flush=True,
        )

        print(
            f"[{cancer}] " f"CURRENT BEST PARAMETER GROUP",
            flush=True,
        )

        print(
            f"Grid ID: " f"{cancer_best['grid_id']}",
            flush=True,
        )

        print(
            f"Best epoch: " f"{cancer_best['best_epoch']}",
            flush=True,
        )

        print(
            f"5-fold mean AUC: " f"{cancer_best['best_mean_auc']:.4f}",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

    best_grid_per_cancer = []

    cancer_names = sorted({row["cancer_type"] for row in best_epoch_rows})

    for cancer in cancer_names:
        rows = [row for row in best_epoch_rows if row["cancer_type"] == cancer]

        best_row = max(
            rows,
            key=lambda x: (
                x["best_mean_auc"],
                -x["best_epoch"],
            ),
        )

        best_grid_per_cancer.append(dict(best_row))

    write(
        OUT / "full_cv_epoch_fold_metrics.csv",
        all_fold_epoch_rows,
    )

    write(
        OUT / "full_cv_epoch_mean_metrics.csv",
        all_mean_epoch_rows,
    )

    write(
        OUT / "full_cv_param_best_epoch.csv",
        best_epoch_rows,
    )

    write(
        OUT / "full_cv_best_grid_per_cancer.csv",
        best_grid_per_cancer,
    )

    if independent_test_metric_rows:
        write(
            OUT / "independent_test_metrics.csv",
            independent_test_metric_rows,
        )

    if independent_test_prediction_rows:
        write(
            OUT / "independent_test_predictions.csv",
            independent_test_prediction_rows,
        )

    print(
        "\n" + "=" * 70,
        flush=True,
    )

    print(
        "Finished.",
        flush=True,
    )

    print(
        "Output files:",
        flush=True,
    )

    print(
        "1. full_cv_epoch_fold_metrics.csv",
        flush=True,
    )

    print(
        "2. full_cv_epoch_mean_metrics.csv",
        flush=True,
    )

    print(
        "3. full_cv_param_best_epoch.csv",
        flush=True,
    )

    print(
        "4. full_cv_best_grid_per_cancer.csv",
        flush=True,
    )

    print(
        "5. independent_test_metrics.csv",
        flush=True,
    )

    print(
        "6. independent_test_predictions.csv",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )


if __name__ == "__main__":
    main()
