"""Visualisation des résultats d'entraînement et d'évaluation des modèles."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .data import TARGET_NAMES


def plot_training(history_path, figure_dir, best_epoch):
    """Trace les pertes d'entraînement, de validation et le taux d'apprentissage."""

    history = pd.read_csv(history_path)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Évolution des pertes
    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["validation_loss"], label="Validation")
    axes[0].axvline(
        best_epoch,
        linestyle="--",
        color="gray",
        label="Meilleur checkpoint",
    )
    axes[0].set(
        xlabel="Époque",
        ylabel="Perte",
        title="Apprentissage",
    )
    axes[0].legend()

    # Évolution du taux d'apprentissage
    axes[1].plot(history["epoch"], history["learning_rate"])
    axes[1].set(
        xlabel="Époque",
        ylabel="Learning rate",
        title="Taux d'apprentissage",
    )

    figure.tight_layout()
    figure.savefig(
        figure_dir / "apprentissage.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.show()


def plot_scores(metrics, kind, figure_dir):
    """Compare les coefficients R² du test principal et de la réserve U."""

    target_names = list(TARGET_NAMES)

    test_metrics = metrics[
        (metrics["model"] == kind) & (metrics["partition"] == "test")
    ].set_index("target").loc[target_names]

    reserve_metrics = metrics[
        (metrics["model"] == kind) & (metrics["partition"] == "reserve_u")
    ].set_index("target").loc[target_names]

    positions = np.arange(len(target_names))

    figure, axis = plt.subplots(figsize=(10, 4))

    axis.bar(
        positions - 0.2,
        test_metrics["R2"],
        width=0.4,
        label="Test principal",
    )
    axis.bar(
        positions + 0.2,
        reserve_metrics["R2"],
        width=0.4,
        label="Réserve U",
    )

    axis.set(
        xticks=positions,
        xticklabels=target_names,
        ylabel="R²",
        title="Performance sur les deux tests",
    )
    axis.axhline(0, color="gray")
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        figure_dir / "r2_test_reserve.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.show()


def plot_scatter(prediction_file, figure_dir, seed=2026):
    """Représente les prédictions en fonction des valeurs réelles."""

    with np.load(prediction_file, allow_pickle=False) as predictions:
        true_values = np.asarray(predictions["y_true"])
        predicted_values = np.asarray(predictions["y_pred"])

    sample_size = min(3000, len(true_values))
    sample_indices = np.random.default_rng(seed).choice(
        len(true_values),
        size=sample_size,
        replace=False,
    )

    figure, axes = plt.subplots(2, 3, figsize=(12, 7))

    for target_index, axis in enumerate(axes.flat):
        actual = true_values[sample_indices, target_index]
        predicted = predicted_values[sample_indices, target_index]

        axis.scatter(actual, predicted, s=5, alpha=0.2)

        lower_limit = min(actual.min(), predicted.min())
        upper_limit = max(actual.max(), predicted.max())

        axis.plot(
            [lower_limit, upper_limit],
            [lower_limit, upper_limit],
            linestyle="--",
            color="gray",
        )
        axis.set(
            title=TARGET_NAMES[target_index],
            xlabel="Réel",
            ylabel="Prédit",
        )

    figure.tight_layout()
    figure.savefig(
        figure_dir / "reel_vs_prediction.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.show()
