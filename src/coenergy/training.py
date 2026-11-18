"""Entraînement, validation et sauvegarde des modèles."""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .loss import family_loss


# 1. CONFIGURATION DES EXPÉRIENCES


@dataclass(frozen=True)
class ExperimentConfig:
    """Regroupe les hyperparamètres d'une expérience d'entraînement."""

    kind: str
    run_id: str
    seed: int = 2026
    batch_size: int = 1024
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 2.0
    lr_patience: int = 6
    lr_factor: float = 0.5
    early_patience: int = 0
    min_delta: float = 1e-5
    hlc_weight: float = 0.5
    tau_weight: float = 0.5
    dropout: float = 0.10
    print_every: int = 100
    archive_every: int = 10


def default_config(kind, run_id="run_001"):
    """Définit les hyperparamètres de référence pour chaque architecture."""

    settings = {"kind": kind, "run_id": run_id}

    if kind == "tcn":
        settings.update(
            batch_size=512,
            max_epochs=120,
            dropout=0.05,
            gradient_clip=5.0,
            lr_patience=8,
            early_patience=22,
            min_delta=1e-4,
        )

    elif kind in ("fusion_50_50", "fusion_30_70"):
        settings.update(
            max_epochs=120,
            early_patience=18,
        )

        if kind == "fusion_30_70":
            settings.update(hlc_weight=0.3, tau_weight=0.7)

    elif kind in ("mlp", "cnn", "cnn_gru"):
        settings.update(
            max_epochs=100,
            early_patience=0,
            min_delta=1e-8,
            archive_every=1,
        )

    else:
        raise ValueError(f"Architecture inconnue : {kind}")

    return ExperimentConfig(**settings)


# 2. REPRODUCTIBILITÉ


def seed_everything(seed):
    """Initialise les générateurs aléatoires utilisés pendant l'expérience."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    """Enregistre l'état des générateurs aléatoires pour une reprise."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    """Restaure l'état des générateurs aléatoires enregistré dans un checkpoint."""

    if state is None:
        return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())

    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


# 3. PRÉDICTIONS DES CINQ CIBLES INDÉPENDANTES


def forward_five(model, batch, kind):
    """Retourne les cinq indicateurs indépendants prédits par le modèle."""

    if kind.startswith("fusion_"):
        predictions = model(batch[0], batch[1])
    else:
        predictions = model(batch[0])

    # Le TCN retourne six indicateurs, HLC étant calculé en première colonne.
    if kind == "tcn":
        predictions = predictions[:, 1:]

    return predictions


# 4. ENTRAÎNEMENT ET VALIDATION


class Experiment:
    """Gère l'entraînement, la validation et les checkpoints d'un modèle."""

    def __init__(self, config, bundle, store, model, out_dir, device):
        """Initialise le modèle, l'optimiseur et les chemins de sauvegarde."""

        # Ces attributs sont également utilisés par les notebooks d'évaluation.
        self.cfg = config
        self.bundle = bundle
        self.store = store
        self.model = model
        self.device = torch.device(device)

        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "figures").mkdir(exist_ok=True)

        self.best = self.out / "best.pt"
        self.last = self.out / "last.pt"
        self.history_path = self.out / "history.csv"

        self.settings = {
            **asdict(config),
            "fingerprints": bundle.fingerprints,
            "partition_sizes": {
                name: len(indices)
                for name, indices in bundle.splits.items()
            },
            "n_features": bundle.n_features,
            "n_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        }

        config_path = self.out / "config.json"

        if config_path.exists():
            with open(config_path, encoding="utf-8") as file:
                if json.load(file) != self.settings:
                    raise ValueError(
                        "Ce dossier contient une autre configuration. "
                        "Choisir un nouvel identifiant d'expérience."
                    )
        else:
            with open(config_path, "w", encoding="utf-8") as file:
                json.dump(self.settings, file, indent=2, ensure_ascii=False)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=config.lr_patience,
            factor=config.lr_factor,
            min_lr=1e-5,
        )

        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.device.type == "cuda"
        )

        self.target_std = torch.tensor(
            bundle.prep["target_std"],
            dtype=torch.float32,
            device=self.device,
        )

        self.hist = []
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.no_progress = 0
        self.start_epoch = 1

    def _loss(self, predictions, targets):
        """Calcule la fonction de perte normalisée des indicateurs thermiques."""

        return family_loss(
            predictions,
            targets,
            self.target_std,
            hlc_weight=self.cfg.hlc_weight,
            tau_weight=self.cfg.tau_weight,
        )

    def _train_one_epoch(self, epoch):
        """Effectue une époque d'entraînement et retourne sa perte moyenne."""

        self.model.train()

        total_loss = 0.0
        total_samples = 0

        train_batches = self.store.batches(
            "train",
            batch_size=self.cfg.batch_size,
            shuffle=True,
            seed=self.cfg.seed,
            epoch=epoch,
        )

        for batch_number, batch in enumerate(train_batches, start=1):
            targets = batch[-1]

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda",
            ):
                predictions = forward_five(self.model, batch, self.cfg.kind)
                loss = self._loss(predictions.float(), targets)

            if not torch.isfinite(loss):
                raise ValueError("Perte non finie pendant l'entraînement.")

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.cfg.gradient_clip,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = len(targets)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if batch_number % self.cfg.print_every == 0:
                print(
                    f"Mini-batch {batch_number} | "
                    f"Loss train : {total_loss / total_samples:.5f}",
                    flush=True,
                )

        return total_loss / total_samples

    @torch.no_grad()
    def validation_loss(self):
        """Calcule la perte moyenne sur l'ensemble de validation."""

        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        validation_batches = self.store.batches(
            "validation", batch_size=self.cfg.batch_size
        )

        for batch in validation_batches:
            targets = batch[-1]

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda",
            ):
                predictions = forward_five(self.model, batch, self.cfg.kind)
                loss = self._loss(predictions.float(), targets)

            batch_size = len(targets)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / total_samples

    def resume(self):
        """Reprend l'entraînement à partir du dernier checkpoint sauvegardé."""

        if not self.last.exists():
            if self.best.exists():
                raise FileNotFoundError(
                    "best.pt existe sans last.pt : restaurer la sauvegarde complète."
                )
            return

        checkpoint = torch.load(
            self.last, map_location=self.device, weights_only=False
        )

        if checkpoint["settings"] != self.settings:
            raise ValueError("Le checkpoint appartient à une autre expérience.")

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])

        self.hist = checkpoint["history"]
        self.best_loss = checkpoint["best_loss"]
        self.best_epoch = checkpoint["best_epoch"]
        self.no_progress = checkpoint["no_progress"]
        self.start_epoch = checkpoint["epoch"] + 1

        restore_rng(checkpoint.get("rng"))

        if self.best_epoch > 0 and not self.best.exists():
            raise FileNotFoundError("Le checkpoint best.pt est introuvable.")

        print(f"Reprise à l'époque {self.start_epoch}.")

    def fit(self, resume=True):
        """Entraîne le modèle et sélectionne le meilleur checkpoint sur validation."""

        if resume:
            self.resume()
        elif self.best.exists() or self.last.exists():
            raise FileExistsError(
                "Cette expérience possède déjà des checkpoints. "
                "Utiliser un nouvel identifiant."
            )

        for epoch in range(self.start_epoch, self.cfg.max_epochs + 1):

            # Politique aléatoire conservée pour les expériences TCN et fusion.
            if self.cfg.kind not in ("mlp", "cnn", "cnn_gru"):
                seed_everything(self.cfg.seed + epoch)

            start_time = time.perf_counter()

            train_loss = self._train_one_epoch(epoch)
            val_loss = self.validation_loss()

            self.scheduler.step(val_loss)

            improved = val_loss < self.best_loss - self.cfg.min_delta

            if improved:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.no_progress = 0

                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "settings": self.settings,
                    },
                    self.best,
                )
            else:
                self.no_progress += 1

            epoch_seconds = time.perf_counter() - start_time

            self.hist.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "epoch_seconds": epoch_seconds,
                }
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "history": self.hist,
                    "best_loss": self.best_loss,
                    "best_epoch": self.best_epoch,
                    "no_progress": self.no_progress,
                    "settings": self.settings,
                    "rng": capture_rng(),
                },
                self.last,
            )

            pd.DataFrame(self.hist).to_csv(self.history_path, index=False)

            print(
                f"Époque {epoch:03d} | "
                f"train={train_loss:.5f} | "
                f"validation={val_loss:.5f} | "
                f"meilleure={self.best_epoch} | "
                f"durée={epoch_seconds:.1f}s",
                flush=True,
            )

            # Archive des checkpoints et des résultats, utile sur Kaggle.
            if self.cfg.archive_every and epoch % self.cfg.archive_every == 0:
                shutil.make_archive(str(self.out), "zip", root_dir=self.out)

            if self.cfg.early_patience and self.no_progress >= self.cfg.early_patience:
                print("Arrêt anticipé : la validation ne s'améliore plus.")
                break

        # Sauvegarde finale, même si l'arrêt survient entre deux archives.
        if self.hist:
            shutil.make_archive(str(self.out), "zip", root_dir=self.out)

        return pd.DataFrame(self.hist)

    def load_best(self):
        """Charge les poids du meilleur modèle sélectionné sur validation."""

        if not self.best.exists():
            raise FileNotFoundError("Aucun meilleur checkpoint n'a été sauvegardé.")

        checkpoint = torch.load(
            self.best, map_location=self.device, weights_only=False
        )

        if checkpoint["settings"] != self.settings:
            raise ValueError("Le checkpoint appartient à une autre expérience.")

        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        return checkpoint["epoch"]
