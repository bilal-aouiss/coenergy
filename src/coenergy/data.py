"""Gestion des données et des partitions pour les modèles."""

from pathlib import Path
import hashlib
import json

import numpy as np
import torch


TARGET_NAMES = ["HLC", "HLCext", "HLCmit", "tau_b", "tau_n", "tau_inf"]
SPLIT_NAMES = ["train", "validation", "test", "reserve_u"]


# 1. LOCALISATION DES FICHIERS


def list_raw_batches(project_root):
    """Retourne la liste ordonnée des fichiers de simulations brutes."""
    raw_folder = Path(project_root) / "quentin_house"
    batch_files = sorted(
        raw_folder.glob("batch_*.pt"),
        key=lambda file: int(file.stem.split("_")[-1]),
    )
    if not batch_files:
        raise FileNotFoundError(f"Aucun batch dans {raw_folder}")
    return batch_files


def sha256_file(file_path):
    """Calcule l'empreinte SHA-256 d'un fichier."""
    file_hash = hashlib.sha256()
    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            file_hash.update(block)
    return file_hash.hexdigest()


def get_data_directories(root):
    """Identifie les répertoires des données préparées et des partitions."""
    root = Path(root)
    prepared = root / "prepared"
    splits = root / "splits"

    if prepared.is_dir() and splits.is_dir():
        return prepared, splits
    if (root / "preprocessing.json").is_file() and (root / "partitions_1m.npz").is_file():
        return root, root
    return None


def locate_data_root(root=None):
    """Localise le répertoire des données dans l'environnement d'exécution."""
    if root is not None:
        data_root = Path(root).resolve()
        if get_data_directories(data_root) is None:
            raise FileNotFoundError(f"Données introuvables : {data_root}")
        return data_root

    current = Path.cwd().resolve()
    candidates = [current / "data", current.parent / "data", current, current.parent]
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        for file in kaggle.rglob("preprocessing.json"):
            candidates.extend([file.parent.parent, file.parent])

    for folder in candidates:
        if get_data_directories(folder) is not None:
            return folder

    raise FileNotFoundError("Dossier data introuvable : préciser root dans load_data().")


# 2. CHARGEMENT DES DONNEES PREPAREES


class DataBundle:
    """Regroupe les données, les partitions et les paramètres de prétraitement."""

    def __init__(self, root, prepared, splits_dir, prep, manifest, splits,
                 row_map, sequences, features, targets, fingerprints, group_verified):
        """Initialise les attributs du conteneur de données."""
        self.root = root
        self.prepared = prepared
        self.splits_dir = splits_dir
        self.prep = prep
        self.manifest = manifest
        self.splits = splits
        self.row_map = row_map
        self.sequences = sequences
        self.features = features
        self.targets = targets
        self.fingerprints = fingerprints
        self.group_verified = group_verified

    @property
    def n(self):
        """Retourne le nombre total de simulations."""
        return len(self.row_map)

    @property
    def n_features(self):
        """Retourne le nombre de caractéristiques sélectionnées."""
        return int(sum(self.prep["active_mask"]))


def load_data(root=None, expected_n=1_000_000, require_group_proof=False):
    """Charge les données préparées, les partitions et les paramètres associés."""
    data_root = locate_data_root(root)
    prepared, splits_dir = get_data_directories(data_root)
    split_path = splits_dir / "partitions_1m.npz"
    preprocessing_path = prepared / "preprocessing.json"
    manifest_path = prepared / "cache_manifest.json"

    with open(preprocessing_path, encoding="utf-8") as file:
        preprocessing = json.load(file)
    with open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)
    with np.load(split_path, allow_pickle=False) as file:
        splits = {name: np.asarray(file[name], dtype=np.int64) for name in SPLIT_NAMES}
        row_map = np.asarray(file["row_map"], dtype=np.int32)
        profile_codes = file["u_profile_code"] if "u_profile_code" in file else None

    number_simulations = len(row_map)
    if expected_n is not None and number_simulations != expected_n:
        raise ValueError("Nombre de simulations incorrect.")

    all_indices = np.concatenate(list(splits.values()))
    if not np.array_equal(np.sort(all_indices), np.arange(number_simulations)):
        raise ValueError("Les partitions se chevauchent ou sont incomplètes.")

    # Les identifiants de profils U sont facultatifs dans les fichiers historiques.
    group_verified = profile_codes is not None
    if group_verified:
        other_indices = np.concatenate([splits["train"], splits["validation"], splits["test"]])
        common_profiles = np.intersect1d(profile_codes[splits["reserve_u"]], profile_codes[other_indices])
        if common_profiles.size:
            raise ValueError("Des profils U réservés apparaissent dans une autre partition.")
    elif require_group_proof:
        raise ValueError("u_profile_code absent des partitions.")

    sequences = np.load(prepared / "sequences.npy", mmap_mode="r")
    features = np.load(prepared / "features.npy", mmap_mode="r")
    targets = np.load(prepared / "targets.npy", mmap_mode="r")

    if sequences.shape != (number_simulations, 2, 480):
        raise ValueError("Dimensions incorrectes pour sequences.npy.")
    if features.shape != (number_simulations, 59):
        raise ValueError("Dimensions incorrectes pour features.npy.")
    if targets.shape != (number_simulations, 6):
        raise ValueError("Dimensions incorrectes pour targets.npy.")

    fingerprints = {
        "partitions_sha256": sha256_file(split_path),
        "preprocessing_sha256": sha256_file(preprocessing_path),
        "cache_manifest_sha256": sha256_file(manifest_path),
    }
    if preprocessing.get("split_sha256") not in (None, fingerprints["partitions_sha256"]):
        raise ValueError("Les paramètres de normalisation ne correspondent pas aux partitions.")

    print(f"Données préparées : {number_simulations:,} simulations")
    print(f"Caractéristiques MLP : {sum(preprocessing['active_mask'])}")

    return DataBundle(data_root, prepared, splits_dir, preprocessing, manifest,
                      splits, row_map, sequences, features, targets,
                      fingerprints, group_verified)


# 3. PREPARATION DES MINI-BATCHS


class BatchStore:
    """Prépare les données normalisées et permet leur lecture par mini-batchs."""

    def __init__(self, bundle, mode, device, gpu_cache="auto", chunk_size=4096):
        """Initialise les données et les paramètres de chargement des mini-batchs."""
        if mode not in ("mlp", "cnn", "cnn_gru", "tcn", "fusion"):
            raise ValueError(f"Modèle inconnu : {mode}")

        self.bundle = bundle
        self.mode = mode
        self.device = torch.device(device)

        use_sequences = mode != "mlp"
        use_features = mode in ("mlp", "fusion")
        number_simulations = bundle.n
        number_features = bundle.n_features

        # Sur Kaggle, la T4 peut conserver les séries en mémoire GPU.
        values_per_simulation = 6 + (960 if use_sequences else 0) + (number_features if use_features else 0)
        required_memory = number_simulations * values_per_simulation * 4
        self.cache_device = torch.device("cpu")

        if self.device.type == "cuda" and gpu_cache != "cpu":
            available_memory = torch.cuda.mem_get_info(self.device)[0]
            if available_memory > required_memory + 4 * 1024**3:
                self.cache_device = self.device
            elif gpu_cache == "cuda":
                raise RuntimeError("Mémoire GPU insuffisante pour charger les séries.")

        self.sequences = (torch.empty((number_simulations, 2, 480), device=self.cache_device)
                          if use_sequences else None)
        self.features = (torch.empty((number_simulations, number_features), device=self.cache_device)
                         if use_features else None)
        self.targets = torch.empty((number_simulations, 6), device=self.cache_device)

        parameters = bundle.prep
        scaling = parameters["signal_scaling"]
        temperature_mean = scaling["T_mean"]
        temperature_std = scaling["T_std"]
        excitation_rms = scaling["U_RMS"]

        if use_features:
            feature_mean = torch.tensor(parameters["feature_mean"], device=self.cache_device)
            feature_std = torch.tensor(parameters["feature_std"], device=self.cache_device)
            selected_features = torch.tensor(parameters["active_mask"], device=self.cache_device, dtype=torch.bool)

        for start in range(0, number_simulations, chunk_size):
            end = min(start + chunk_size, number_simulations)

            if use_sequences:
                signals = torch.tensor(np.array(bundle.sequences[start:end]), device=self.cache_device)
                signals[:, 0] = (signals[:, 0] - temperature_mean) / temperature_std
                signals[:, 1] /= excitation_rms
                self.sequences[start:end] = signals

            if use_features:
                statistics = torch.tensor(np.array(bundle.features[start:end]), device=self.cache_device)
                self.features[start:end] = ((statistics - feature_mean) / feature_std)[:, selected_features]

            self.targets[start:end] = torch.tensor(np.array(bundle.targets[start:end]), device=self.cache_device)

        print(f"Données {mode} en mémoire : {self.cache_device}")

    def get_batch(self, row_ids):
        """Extrait les entrées et les cibles correspondant aux indices fournis."""
        indices = torch.as_tensor(np.asarray(row_ids), dtype=torch.long, device=self.cache_device)
        target_values = self.targets.index_select(0, indices).to(self.device)

        if self.mode == "mlp":
            return self.features.index_select(0, indices).to(self.device), target_values

        signals = self.sequences.index_select(0, indices).to(self.device)
        if self.mode == "fusion":
            statistics = self.features.index_select(0, indices).to(self.device)
            return signals, statistics, target_values
        return signals, target_values

    def batches(self, partition, batch_size, shuffle=False, seed=2026, epoch=0):
        """Génère les mini-batchs d'une partition, avec mélange optionnel."""
        indices = self.bundle.splits[partition] if isinstance(partition, str) else np.asarray(partition)
        indices = (np.random.default_rng(seed + epoch).permutation(indices)
                   if shuffle else np.sort(indices))

        for start in range(0, len(indices), batch_size):
            yield self.get_batch(indices[start:start + batch_size])
