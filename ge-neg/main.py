import pathlib
from time import time

import cv2
import numpy as np
import tifffile as tiff
from border_identifier import BorderIdentifier
from contrast_booster import ContrastBoosterGenetic, apply_s_curve
from preprocessing import expose_to_the_right, load_and_normalize, remove_scanner_light
from scipy.optimize import minimize


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


def downsample_for_optimizer(
    img: np.ndarray, target_pixels: int = 1_000_000
) -> np.ndarray:
    """Riduce l'immagine di densità a ~2 Megapixel usando Average Pooling (INTER_AREA)."""
    height, width, channel = img.shape
    total_pixels = height * width

    if total_pixels <= target_pixels:
        return img  # Già sotto i 2MP

    # Calcolo del fattore di scala esatto
    scale = np.sqrt(target_pixels / total_pixels)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # INTER_AREA esegue un Average Pooling perfetto sui blocchi di pixel
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


def inversion_and_density_balance(
    transmittance_balanced: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """MODULO 3: Conversione in densità logaritmica, calcolo Density Balance e Inversione.

    - transmittance_balanced: Array Float32/Float64 [0.0, 1.0] dal Modulo 2

    Returns:
    - positive_img: Immagine positiva invertita e normalizzata [0.0, 1.0]
    - best_gammas: I tre coefficienti di Density Balance calcolati [gamma_R, gamma_G, gamma_B]
    """
    t_safe = np.clip(transmittance_balanced, 1e-5, 1.0)
    density = -np.log10(t_safe)

    # 2. Calcolo dei percentili 1% e 99% UNA SOLA VOLTA sui 3 canali
    p1 = np.percentile(density, 1, axis=(0, 1))  # [p1_R, p1_G, p1_B]
    p99 = np.percentile(density, 99, axis=(0, 1))  # [p99_R, p99_G, p99_B]

    # Il contrasto (gamma) di ciascun canale è l'ampiezza dell'intervallo dinamico (p99 - p1)
    contrast = p99 - p1

    # Normalizziamo i Gamma rispetto al canale Verde (G) come riferimento
    best_gammas = contrast[1] / contrast

    # 3. Inversione e Normalizzazione
    balanced_density = density * best_gammas
    d_min = balanced_density.min()
    d_max = balanced_density.max()

    positive_img = (balanced_density - d_min) / (d_max - d_min + 1e-6)

    return positive_img, best_gammas


def scene_wb(transmittance: np.ndarray, sat_threshold: float = 0.15) -> np.ndarray:
    """MODULO 2: Bilanciamento del bianco della scena basato su pixel neutri (HSV).

    - transmittance: Array float32 [0.0, 1.0] dal Modulo 1
    """
    # 1. Conversione temporanea in HSV per misurare la saturazione
    hsv = cv2.cvtColor(transmittance, cv2.COLOR_RGB2HSV_FULL)
    saturation = hsv[:, :, 1]
    print(
        f"saturation -> shape: {saturation.shape} - min: {saturation.min()} - max: {saturation.max()} - mean: {saturation.mean()} "
    )

    # 2. Maschera dei pixel cromaticamente neutri
    neutral_mask = saturation < sat_threshold

    # Fallback se ci sono troppi pochi pixel neutri (es. foto ultra satura)
    if np.sum(neutral_mask) < (transmittance.shape[0] * transmittance.shape[1] * 0.01):
        neutral_mask = saturation < 0.35

    # 3. Calcolo del valore medio nei 3 canali sui soli pixel neutri
    neutral_pixels = transmittance[neutral_mask]
    print(
        f"neutral_pixels -> shape: {neutral_pixels.shape} - min: {neutral_pixels.min()} - max: {neutral_pixels.max()} - mean: {neutral_pixels.mean()} "
    )
    channel_means = np.mean(neutral_pixels, axis=0)
    print(
        f"channel_means -> shape: {channel_means.shape} - min: {channel_means.min()} - max: {channel_means.max()} - mean: {channel_means.mean()} "
    )

    # 4. Normalizzazione rispetto al verde (G è il canale di riferimento di luminanza)
    wb_factors = channel_means[1] / np.clip(channel_means, 1e-6, 1.0)

    print(
        f"transmittance - min: {transmittance.min()} - max: {transmittance.max()} - mean: {transmittance.mean()}"
    )
    print(f"wb_factors - {wb_factors}")

    # 5. Applicazione del bilanciamento e clipping
    balanced = transmittance * wb_factors
    print(
        f"balanced - min: {balanced.min()} - max: {balanced.max()} - mean: {balanced.mean()}"
    )
    return np.clip(balanced, 1e-6, 1.0)


def save_to_file(
    img: np.ndarray, image_type: str, image_description: str, suffix: str
) -> bool:

    img_clip = np.clip(img, 0.0, 1.0)
    img_16bit = (img_clip * 65535.0).astype(np.uint16)
    return cv2.imwrite(
        f"/home/paolo/git/ge-neg/tests/output_images/{image_type}/{image_description}_{suffix}.tiff",
        cv2.cvtColor(img_16bit, cv2.COLOR_RGB2BGR),
    )


image_type = "bianco_nero"  # colori bianco_nero
image_description = "correttamente_esposta"

path = pathlib.Path(
    f"/home/paolo/git/ge-neg/tests/input_images/{image_type}/{image_description}.tif"
)

img = load_and_normalize(path)

print(f"img.shape: {img.shape}")

trimmed_image = remove_scanner_light(img)
save_to_file(trimmed_image, image_type, image_description, "scanner_crop")

ettr = expose_to_the_right(trimmed_image)
save_to_file(ettr, image_type, image_description, "exposure_compensation")


border_identifier = BorderIdentifier(img=ettr)
border_identifier.find_borders()
photo_pixels = border_identifier.get_image()
borders = border_identifier.get_borders()

print(
    f"photo_pixels shape: {photo_pixels.shape} - Area ratio: {border_identifier.get_area_ratio()}"
)
print(f"borders shape: {borders.shape}")
print(f"film base is: {border_identifier.get_film_base()}")
save_to_file(photo_pixels, image_type, image_description, "tagliata")

image_wb = film_base_wb(photo_pixels, film_base_color=border_identifier.get_film_base())
print(
    f"image_wb min: {image_wb.min()} - max: {image_wb.max()} - mean: {image_wb.mean()}"
)
save_to_file(image_wb, image_type, image_description, "wb")

positive_img, best_gammas = inversion_and_density_balance(image_wb)
save_to_file(positive_img, image_type, image_description, "positive")

img_scene_wb = scene_wb(positive_img)
save_to_file(img_scene_wb, image_type, image_description, "wb_scena")

contrast_booster = ContrastBoosterGenetic(img_scene_wb)
contrast_booster.run()
contrasted_image = apply_s_curve(
    positive_img, *contrast_booster.genetic_optimizer.best_solution()[0]
)
save_to_file(contrasted_image, image_type, image_description, "contrast_booster")
