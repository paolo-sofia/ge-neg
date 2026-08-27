import numpy as np


def inversion_and_density_balance(
    transmittance_balanced: np.ndarray,
) -> np.ndarray:
    """MODULO 3: Conversione in densità logaritmica, calcolo Density Balance e Inversione.

    - transmittance_balanced: Array Float32/Float64 [0.0, 1.0] dal Modulo 2

    Returns:
    - positive_img: Immagine positiva invertita e normalizzata [0.0, 1.0]
    - best_gammas: I tre coefficienti di Density Balance calcolati [gamma_R, gamma_G, gamma_B]
    """
    t_safe = np.clip(transmittance_balanced, 1e-5, 1.0)
    density = -np.log10(t_safe)

    # 2. Calcolo dei percentili 1% e 99% UNA SOLA VOLTA sui 3 canali
    p1: np.ndarray = np.percentile(density, 1, axis=(0, 1))  # [p1_R, p1_G, p1_B]
    p99: np.ndarray = np.percentile(density, 99, axis=(0, 1))  # [p99_R, p99_G, p99_B]

    # Il contrasto (gamma) di ciascun canale è l'ampiezza dell'intervallo dinamico (p99 - p1)
    contrast: np.ndarray = p99 - p1

    # Normalizziamo i Gamma rispetto al canale Verde (G) come riferimento
    best_gammas: np.ndarray = contrast[1] / contrast

    # 3. Inversione e Normalizzazione
    balanced_density: np.ndarray = density * best_gammas
    d_min: float = balanced_density.min()
    d_max: float = balanced_density.max()

    positive_img = (balanced_density - d_min) / (d_max - d_min + 1e-6)

    return positive_img
