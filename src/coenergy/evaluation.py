"""Évaluation des prédictions et analyse des erreurs des modèles."""

import numpy as np
import pandas as pd
import torch

from .data import TARGET_NAMES
from .loss import reconstruct_six
from .training import forward_five


TARGET_UNITS = ["W/K", "W/K", "W/K", "h", "h", "h"]


# 1. CALCUL DES MÉTRIQUES


def calculate_metrics(truth, pred, partition, model_name):
    """Calcule les métriques de régression pour les six indicateurs thermiques."""

    true_values = np.asarray(truth, dtype=np.float64)
    predicted_values = np.asarray(pred, dtype=np.float64)

    if true_values.shape != predicted_values.shape or true_values.ndim != 2 or true_values.shape[1] != 6:
        raise ValueError("Les valeurs réelles et prédites doivent avoir six colonnes identiques.")

    results = []

    for target_index, target_name in enumerate(TARGET_NAMES):
        actual = true_values[:, target_index]
        predicted = predicted_values[:, target_index]
        errors = predicted - actual
        absolute_errors = np.abs(errors)

        mae = np.mean(absolute_errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        total_variation = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - np.sum(errors ** 2) / total_variation if total_variation > 0 else np.nan

        actual_std = actual.std()
        predicted_std = predicted.std()
        dispersion_ratio = predicted_std / actual_std if actual_std > 0 else np.nan

        results.append({
            "model": model_name,
            "partition": partition,
            "target": target_name,
            "unit": TARGET_UNITS[target_index],
            "n": len(actual),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "P95_abs": np.quantile(absolute_errors, 0.95),
            "bias": errors.mean(),
            "pred_to_true_std": dispersion_ratio,
        })

    return pd.DataFrame(results)


# 2. PRÉDICTIONS SUR UNE PARTITION


@torch.no_grad()
def predict_partition(experiment, name):
    """Calcule les prédictions et conserve les indices des simulations évaluées."""

    experiment.model.eval()

    true_batches = []
    predicted_batches = []

    for batch in experiment.store.batches(name, experiment.cfg.batch_size):
        with torch.autocast(
            device_type=experiment.device.type,
            enabled=experiment.device.type == "cuda",
        ):
            five_predictions = forward_five(
                experiment.model, batch, experiment.cfg.kind
            )

        six_predictions = reconstruct_six(five_predictions.float())

        true_batches.append(batch[-1].detach().cpu().numpy())
        predicted_batches.append(six_predictions.detach().cpu().numpy())

    simulation_indices = np.sort(experiment.bundle.splits[name]).astype(np.int32)
    true_values = np.concatenate(true_batches)
    predicted_values = np.concatenate(predicted_batches)

    return simulation_indices, true_values, predicted_values


# 3. ÉVALUATION DES MODÈLES


def evaluate_all(experiment, include_validation=True):
    """Évalue un modèle et une référence constante sur les partitions choisies."""

    if include_validation:
        partitions = ["validation", "test", "reserve_u"]
    else:
        partitions = ["test", "reserve_u"]

    target_means = np.asarray(experiment.bundle.prep["target_mean"], dtype=np.float64)
    hlc_mean = target_means[0] + target_means[1]
    baseline_values = np.concatenate(([hlc_mean], target_means))

    predictions_folder = experiment.out / "predictions"
    predictions_folder.mkdir(exist_ok=True)

    metrics_tables = []

    for partition in partitions:
        simulation_indices, true_values, predicted_values = predict_partition(
            experiment, partition
        )

        if not np.isfinite(predicted_values).all() or not (predicted_values > 0).all():
            raise ValueError("Le modèle a produit des prédictions non valides.")

        if not np.allclose(
            predicted_values[:, 0],
            predicted_values[:, 1] + predicted_values[:, 2],
            rtol=1e-5,
            atol=1e-4,
        ):
            raise ValueError("La contrainte physique du HLC n'est pas respectée.")

        model_metrics = calculate_metrics(
            true_values, predicted_values, partition, experiment.cfg.kind
        )
        baseline_predictions = np.broadcast_to(baseline_values, true_values.shape)
        baseline_metrics = calculate_metrics(
            true_values, baseline_predictions, partition, "moyenne_train"
        )

        metrics_tables.extend([model_metrics, baseline_metrics])

        np.savez_compressed(
            predictions_folder / f"{partition}.npz",
            row_id=simulation_indices,
            y_true=true_values.astype(np.float32),
            y_pred=predicted_values.astype(np.float32),
        )

    metrics = pd.concat(metrics_tables, ignore_index=True)
    metrics.to_csv(experiment.out / "metrics.csv", index=False)

    return metrics


# 4. ANALYSE DES VALEURS EXTRÊMES


def extreme_diagnostics(experiment, partitions=("test", "reserve_u")):
    """Analyse les erreurs sur les valeurs extrêmes des constantes de temps."""

    training_indices = np.sort(experiment.bundle.splits["train"])
    training_targets = np.asarray(
        experiment.bundle.targets[training_indices, 1:], dtype=np.float32
    )

    lower_limits = np.quantile(training_targets, 0.05, axis=0)
    upper_limits = np.quantile(training_targets, 0.95, axis=0)

    results = []

    for partition in partitions:
        predictions_path = experiment.out / "predictions" / f"{partition}.npz"

        with np.load(predictions_path, allow_pickle=False) as saved:
            true_values = saved["y_true"]
            predicted_values = saved["y_pred"]

        for target_name, target_column, independent_column in [
            ("tau_b", 3, 2),
            ("tau_inf", 5, 4),
        ]:
            selected_regions = [
                ("5 % faibles", true_values[:, target_column] <= lower_limits[independent_column]),
                ("5 % élevés", true_values[:, target_column] >= upper_limits[independent_column]),
            ]

            for region_name, selected_rows in selected_regions:
                errors = (
                    predicted_values[selected_rows, target_column]
                    - true_values[selected_rows, target_column]
                )

                results.append({
                    "partition": partition,
                    "target": target_name,
                    "region": region_name,
                    "n": len(errors),
                    "RMSE": np.sqrt(np.mean(errors ** 2)) if len(errors) else np.nan,
                    "bias": errors.mean() if len(errors) else np.nan,
                })

    diagnostics = pd.DataFrame(results)
    diagnostics.to_csv(experiment.out / "extremes.csv", index=False)

    return diagnostics
