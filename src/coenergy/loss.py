"""Reconstruction des indicateurs thermiques et calcul de la fonction de perte."""

import torch


def reconstruct_six(five):
    """Reconstruit les six indicateurs à partir des cinq sorties indépendantes."""

    hlc = five[:, :2].sum(dim=1, keepdim=True)

    return torch.cat([hlc, five], dim=1)


def family_loss(five, truth_six, target_std, hlc_weight=0.5, tau_weight=0.5):
    """Calcule la perte quadratique normalisée des deux familles de cibles."""

    if five.shape[1] != 5 or truth_six.shape[1] != 6:
        raise ValueError("Les prédictions doivent avoir 5 colonnes et les cibles 6.")

    if abs(hlc_weight + tau_weight - 1.0) > 1e-8:
        raise ValueError("La somme des deux poids doit être égale à 1.")

    # Normalisation par les écarts-types calculés sur l'entraînement.
    standard_deviations = torch.as_tensor(
        target_std,
        dtype=five.dtype,
        device=five.device
    )

    normalized_errors = (five - truth_six[:, 1:]) / standard_deviations

    # Les deux premiers indicateurs sont les déperditions thermiques.
    hlc_loss = normalized_errors[:, :2].square().mean()

    # Les trois derniers indicateurs sont les constantes de temps.
    tau_loss = normalized_errors[:, 2:].square().mean()

    return hlc_weight * hlc_loss + tau_weight * tau_loss
