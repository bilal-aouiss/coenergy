"""Architectures de réseaux de neurones pour l'identification thermique des bâtiments."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def initialize_positive_output(output_layer):
    """Initialise une couche de sortie pour produire des valeurs proches des moyennes cibles."""
    nn.init.normal_(output_layer.weight, std=1e-3)
    nn.init.constant_(output_layer.bias, math.log(math.expm1(1.0)))


# -----------------------------------------------------------------------------
# 1. MLP : caractéristiques statistiques
# -----------------------------------------------------------------------------

class MLPSeul(nn.Module):
    """Prédit cinq paramètres thermiques à partir de caractéristiques statistiques."""

    def __init__(self, n_features, target_mean, dropout=0.10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.SiLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(32, 96),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 5),
        )

        initialize_positive_output(self.head[-1])
        self.register_buffer(
            "target_mean", torch.as_tensor(target_mean, dtype=torch.float32)
        )

    def forward(self, features):
        """Calcule cinq prédictions positives à partir des caractéristiques d'entrée."""
        encoded_features = self.encoder(features)
        predictions = self.head(encoded_features)
        return self.target_mean * F.softplus(predictions)


# -----------------------------------------------------------------------------
# 2. CNN 1D : séries temporelles de température et de sollicitation
# -----------------------------------------------------------------------------

class CNN1DSeul(nn.Module):
    """Prédit cinq paramètres thermiques à partir de deux séries temporelles."""

    def __init__(self, target_mean, dropout=0.10):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 48, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
        )

        self.project = nn.Sequential(
            nn.Linear(96, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(64, 96),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 5),
        )

        initialize_positive_output(self.head[-1])
        self.register_buffer(
            "target_mean", torch.as_tensor(target_mean, dtype=torch.float32)
        )

    def forward(self, signals):
        """Extrait des représentations temporelles et calcule cinq prédictions positives."""
        convolution_output = self.cnn(signals)

        mean_features = convolution_output.mean(dim=2)
        max_features = convolution_output.amax(dim=2)
        temporal_features = torch.cat([mean_features, max_features], dim=1)

        encoded_features = self.project(temporal_features)
        predictions = self.head(encoded_features)

        return self.target_mean * F.softplus(predictions)


# -----------------------------------------------------------------------------
# 3. CNN-GRU : extraction locale et dépendances temporelles
# -----------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """Encode les séries temporelles à l'aide de convolutions et d'un GRU."""

    def __init__(self, dropout=0.10):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 48, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
        )

        self.gru = nn.GRU(input_size=48, hidden_size=64, batch_first=True)

        self.projection = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, signals):
        """Produit un vecteur de 64 caractéristiques temporelles."""
        convolution_output = self.cnn(signals)
        temporal_sequence = convolution_output.transpose(1, 2)

        gru_sequence, last_hidden_state = self.gru(temporal_sequence)

        final_state = last_hidden_state[-1]
        mean_state = gru_sequence.mean(dim=1)
        temporal_features = torch.cat([final_state, mean_state], dim=1)

        return self.projection(temporal_features)


class CNNGRUSeul(nn.Module):
    """Prédit cinq paramètres thermiques à partir d'un réseau CNN-GRU."""

    def __init__(self, target_mean, dropout=0.10):
        super().__init__()

        # Les sous-modules restent distincts pour conserver les checkpoints existants.
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 48, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
        )

        self.gru = nn.GRU(input_size=48, hidden_size=64, batch_first=True)

        self.projection = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(64, 96),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 5),
        )

        initialize_positive_output(self.head[-1])
        self.register_buffer(
            "target_mean", torch.as_tensor(target_mean, dtype=torch.float32)
        )

    def forward(self, signals):
        """Calcule cinq prédictions positives à partir des séries temporelles."""
        convolution_output = self.cnn(signals)
        temporal_sequence = convolution_output.transpose(1, 2)

        gru_sequence, last_hidden_state = self.gru(temporal_sequence)

        final_state = last_hidden_state[-1]
        mean_state = gru_sequence.mean(dim=1)
        temporal_features = torch.cat([final_state, mean_state], dim=1)

        encoded_features = self.projection(temporal_features)
        predictions = self.head(encoded_features)

        return self.target_mean * F.softplus(predictions)


# -----------------------------------------------------------------------------
# 4. Fusion : CNN-GRU et caractéristiques statistiques
# -----------------------------------------------------------------------------

class StatisticalEncoder(nn.Module):
    """Encode les caractéristiques statistiques dans un vecteur de 32 valeurs."""

    def __init__(self, n_features, dropout=0.10):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.SiLU(),
        )

    def forward(self, features):
        """Calcule une représentation des caractéristiques statistiques."""
        return self.network(features)


class ThermalFusion(nn.Module):
    """Combine les représentations temporelles et statistiques pour la régression."""

    def __init__(self, n_features, target_mean, dropout=0.10):
        super().__init__()

        self.temporal = TemporalEncoder(dropout)
        self.statistical = StatisticalEncoder(n_features, dropout)

        self.regressor = nn.Sequential(
            nn.Linear(96, 96),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 5),
        )

        initialize_positive_output(self.regressor[-1])
        self.register_buffer(
            "target_mean", torch.as_tensor(target_mean, dtype=torch.float32)
        )

    def forward(self, signals, features):
        """Fusionne deux représentations et calcule cinq prédictions positives."""
        temporal_features = self.temporal(signals)
        statistical_features = self.statistical(features)

        combined_features = torch.cat(
            [temporal_features, statistical_features], dim=1
        )
        predictions = self.regressor(combined_features)

        return self.target_mean * F.softplus(predictions)


# -----------------------------------------------------------------------------
# 5. TCN : convolutions temporelles dilatées et connexions résiduelles
# -----------------------------------------------------------------------------

def enrich_inputs(signals):
    """Ajoute les variations temporelles et six statistiques aux signaux d'entrée."""
    differences = torch.zeros_like(signals)
    differences[:, :, 1:] = signals[:, :, 1:] - signals[:, :, :-1]

    enriched_signals = torch.cat([signals, differences], dim=1)

    temperature = signals[:, 0]
    excitation = signals[:, 1]
    temperature_difference = differences[:, 0]
    excitation_difference = differences[:, 1]

    statistics = torch.stack(
        [
            temperature[:, -1] - temperature[:, 0],
            temperature.amax(dim=1) - temperature.amin(dim=1),
            (excitation.abs() > 1e-3).float().mean(dim=1),
            excitation.mean(dim=1),
            temperature_difference.abs().mean(dim=1),
            excitation_difference.abs().mean(dim=1),
        ],
        dim=1,
    )

    return enriched_signals, statistics


class ResidualBlock(nn.Module):
    """Applique deux convolutions dilatées avec une connexion résiduelle."""

    def __init__(self, in_channels, width, dilation, dropout):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, width, kernel_size=3,
            padding=dilation, dilation=dilation,
        )

        self.conv2 = nn.Conv1d(
            width, width, kernel_size=3,
            padding=dilation, dilation=dilation,
        )

        self.projection = (
            nn.Conv1d(in_channels, width, kernel_size=1)
            if in_channels != width
            else nn.Identity()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, signals):
        """Calcule une représentation résiduelle des signaux temporels."""
        first_convolution = self.conv1(signals)
        first_convolution = self.dropout(F.silu(first_convolution))

        second_convolution = self.conv2(first_convolution)
        second_convolution = self.dropout(second_convolution)

        residual = self.projection(signals)
        return F.silu(second_convolution + residual)


class ThermalTCN(nn.Module):
    """Prédit six indicateurs thermiques avec un réseau convolutif temporel."""

    def __init__(
        self,
        target_mean,
        target_std,
        width=16,
        dilations=(1, 2, 4, 8, 16, 32, 64),
        dropout=0.05,
        dense_width=64,
    ):
        super().__init__()

        blocks = []
        in_channels = 4

        for dilation in dilations:
            blocks.append(ResidualBlock(in_channels, width, dilation, dropout))
            in_channels = width

        self.encoder = nn.Sequential(*blocks)

        self.regressor = nn.Sequential(
            nn.Linear(2 * width + 6, dense_width),
            nn.SiLU(),
            nn.Linear(dense_width, 5),
        )

        mean = torch.as_tensor(target_mean, dtype=torch.float32)
        scale = torch.as_tensor(target_std, dtype=torch.float32)

        self.register_buffer("target_scale", scale)
        self.register_buffer("softplus_offset", torch.log(torch.expm1(mean / scale)))

    def forward(self, signals):
        """Reconstitue HLC à partir des cinq paramètres thermiques prédits."""
        enriched_signals, statistics = enrich_inputs(signals)
        encoded_signals = self.encoder(enriched_signals)

        mean_features = encoded_signals.mean(dim=2)
        max_features = encoded_signals.amax(dim=2)
        combined_features = torch.cat(
            [mean_features, max_features, statistics], dim=1
        )

        predictions = self.regressor(combined_features)
        independent_targets = self.target_scale * F.softplus(
            predictions + self.softplus_offset
        )

        total_hlc = independent_targets[:, :2].sum(dim=1, keepdim=True)
        return torch.cat([total_hlc, independent_targets], dim=1)


# -----------------------------------------------------------------------------
# 6. Sélection de l'architecture
# -----------------------------------------------------------------------------

def build_model(kind, preprocessing):
    """Construit une architecture à partir de son nom et du prétraitement."""
    target_mean = preprocessing["target_mean"]
    if kind == "mlp":
        number_of_features = sum(preprocessing["active_mask"])
        return MLPSeul(number_of_features, target_mean, dropout=0.10)

    if kind == "cnn":
        return CNN1DSeul(target_mean, dropout=0.10)

    if kind == "cnn_gru":
        return CNNGRUSeul(target_mean, dropout=0.10)

    if kind in ("fusion_50_50", "fusion_30_70"):
        number_of_features = sum(preprocessing["active_mask"])
        return ThermalFusion(number_of_features, target_mean, dropout=0.10)

    if kind == "tcn":
        return ThermalTCN(target_mean, preprocessing["target_std"])

    raise ValueError(f"Architecture inconnue : {kind}")
