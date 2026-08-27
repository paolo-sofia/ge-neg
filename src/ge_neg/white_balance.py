import cv2
import numpy as np


def film_base_wb(img: np.ndarray, film_base_color: np.ndarray) -> np.ndarray:
    """MODULO 1: Calcola la Trasmittanza relativa eliminando il colore della base.

    - img: Array RGB float32 [0.0, 1.0]
    - film_base_color: Array (3,) ottenuto dalla mediana dei bordi [0.0, 1.0]
    """
    # 1. Evitiamo divisioni per zero o valori non validi (clipping di sicurezza)
    img_safe = np.clip(img, 1e-6, 1.0)
    base_safe = np.clip(film_base_color, 1e-6, 1.0)

    # 2. Applicazione della formula T = I / Base
    transmittance = img_safe / base_safe

    # 3. Rinormalizzazione Globale: riporta il valore massimo dell'immagine a 1.0
    # mantenendo intatti i rapporti cromatici tra i canali senza tagliare (clip) i dati!
    max_val = np.max(transmittance)
    transmittance = transmittance / max(1e-6, max_val)

    transmittance = np.clip(transmittance, 1e-6, 1.0)

    return transmittance


def scene_wb(transmittance: np.ndarray, sat_threshold: float = 0.15) -> np.ndarray:
    """MODULO 2: Bilanciamento del bianco della scena basato su pixel neutri (HSV).

    - transmittance: Array float32 [0.0, 1.0] dal Modulo 1
    """
    # 1. Conversione temporanea in HSV per misurare la saturazione
    hsv = cv2.cvtColor(transmittance, cv2.COLOR_RGB2HSV_FULL)
    saturation = hsv[:, :, 1]

    # 2. Maschera dei pixel cromaticamente neutri
    neutral_mask = saturation < sat_threshold

    # Fallback se ci sono troppi pochi pixel neutri (es. foto ultra satura)
    if np.sum(neutral_mask) < (transmittance.shape[0] * transmittance.shape[1] * 0.01):
        neutral_mask = saturation < 0.35

    # 3. Calcolo del valore medio nei 3 canali sui soli pixel neutri
    neutral_pixels = transmittance[neutral_mask]

    channel_means: np.ndarray = np.mean(neutral_pixels, axis=0)

    # 4. Normalizzazione rispetto al verde (G è il canale di riferimento di luminanza)
    wb_factors: np.ndarray = channel_means[1] / np.clip(channel_means, 1e-6, 1.0)

    # 5. Applicazione del bilanciamento e clipping
    balanced: np.ndarray = transmittance * wb_factors

    return np.clip(balanced, 1e-6, 1.0)
