import pathlib

import cv2
import numpy as np


def normalize_image(img: np.ndarray, is_linear: bool) -> np.ndarray:
    if img.ndim == 2:  # Conversione B/N -> RGB a 3 canali
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.shape[2] == 4:  # Se presente canale Alfa
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    # Scala in base alla profondità di bit del file
    img_float: np.ndarray = img.astype(np.float32) / (2 ** (img.dtype.itemsize * 8) - 1)

    # 2. Gestione della Linearizzazione Spaziale
    if is_linear:
        return img_float

    return np.where(
        img_float <= 0.04045,
        img_float / 12.92,
        np.power((img_float + 0.055) / 1.055, 2.4),
    )


def expose_to_the_right(img: np.ndarray, target_white: float = 99.95) -> np.ndarray:
    # 2. Identificazione del punto di bianco della luminanza
    y_max: float = np.percentile(img, target_white)

    # Evitiamo divisioni per zero o valori nulli
    if y_max < 1e-6:
        return img

    # 3. Calcolo del guadagno scalare UNICO per tutta l'immagine (Stile slider Exposure)
    gain = 1.0 / y_max

    # 4. Applicazione dello stesso guadagno su tutti i canali per non alterare la tinta
    img_scaled = img * gain

    # 3. Identificazione del valore massimo per ogni singolo PIXEL tra i 3 canali
    # Shape: (H, W, 1)
    pixel_max = np.max(img_scaled, axis=-1, keepdims=True)

    # 4. Calcolo del fattore di scala locale:
    # Se pixel_max > 1.0, dividiamo per pixel_max (cioè moltiplichiamo per 1 / pixel_max)
    # Se pixel_max <= 1.0, il fattore rimane 1.0 (nessuna modifica)
    scale_factor = np.where(pixel_max > 1.0, 1.0 / pixel_max, 1.0)

    # 5. Applicazione del descaling locale per preservare la tinta (R:G:B invariato)
    img_out = img_scaled * scale_factor

    return np.clip(img_out, 0.0, 1.0)


def remove_noise_from_scanned_light_mask(
    mask_indexes: set[int],
) -> list[int]:
    idxs = sorted(mask_indexes)
    idxs_cleaned = []

    for i in range(len(idxs)):
        if i == len(idxs) - 1:
            if idxs[i] - idxs[i - 1] == 1:
                idxs_cleaned.append(idxs[i])

            break

        if idxs[i + 1] - idxs[i] == 1:
            idxs_cleaned.append(idxs[i])

    return idxs_cleaned


def find_farthest_point(points, reference_pt=(0, 0), function=np.argmax):
    """
    Trova il punto più distante da un punto di riferimento custom (ref_x, ref_y).

    :param points: Array NumPy di punti (N, 2) o (N, 1, 2)
    :param reference_pt: Tupla o lista con le coordinate (x, y) del punto di riferimento
    :return: Coordinate [x, y] del punto con la distanza massima
    """
    # 1. Normalizziamo la forma a (N, 2)
    pts = points.squeeze()

    if pts.ndim == 1:
        return pts

    # 2. Convertiamo il punto di riferimento in un array NumPy
    ref = np.array(reference_pt, dtype=np.float32)

    # 3. Traslazione dei punti e calcolo della distanza quadratica: (X - X0)^2 + (Y - Y0)^2
    # NumPy applica la sottrazione in broadcast su tutte le righe
    dist_squared = np.sum((pts - ref) ** 2, axis=1)

    # 4. Indice del punto più distante
    idx = function(dist_squared)

    return pts[idx]


def find_biggest_masked_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    height, width = mask.shape

    is_reversed = False
    length_to_loop = height
    if height > width:
        is_reversed = True
        length_to_loop = width

    # iteration bottom to top
    axis_start_white_pixels = set()
    axis_end_white_pixels = set()
    for idx_axis in range(length_to_loop):
        if not is_reversed:
            axis_values = mask[idx_axis, :]
        else:
            axis_values = mask[:, idx_axis]

        mask_lenght = np.where(axis_values > 128)[0]
        axis_start_white_pixels.add(mask_lenght.min())
        axis_end_white_pixels.add(mask_lenght.max())

    if not axis_start_white_pixels:
        start_point = 0
    else:
        start_point = np.max(sorted(axis_start_white_pixels))

    if not axis_end_white_pixels:
        end_point = 0
    else:
        end_point = np.max(sorted(axis_end_white_pixels))

    if is_reversed:
        return start_point, end_point, 0, height
    else:
        return 0, width, start_point, end_point
