import pathlib
from turtle import left

import cv2
import numpy as np
from numpy._typing import _16Bit
from numpy.testing._private.utils import break_cycles


def load_and_normalize(
    image_path: pathlib.Path, is_linear_raw: bool = True
) -> np.ndarray:
    # 1. Caricamento / Normalizzazione iniziale a float32 [0.0, 1.0]
    # Usiamo IMREAD_UNCHANGED per preservare i 16-bit reali dei TIFF di VueScan

    img: np.ndarray | None = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Impossibile caricare l'immagine: {image_path}")

    if img.ndim == 2:  # Conversione B/N -> RGB a 3 canali
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.shape[2] == 4:  # Se presente canale Alfa
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    # Scala in base alla profondità di bit del file
    if img.dtype == np.uint16:
        print("File è scansione a 16 bit")
        img_float = img.astype(np.float32) / 65535.0
        # Di default, un TIFF a 16-bit da VueScan viene trattato come Lineare
        is_linear_raw = True
    else:
        print("File è scansione a 8 bit")
        img_float = img.astype(np.float32) / 255.0

    # 2. Gestione della Linearizzazione Spaziale
    if is_linear_raw:
        linear_rgb = img_float
        print(
            "   ✓ [MODULO 0] Input riconosciuto come TIFF LINEARE (VueScan RAW). Nessuna de-gamma applicata."
        )
    else:
        # Rimozione della gamma sRGB per immagini convenzionali
        linear_rgb = np.where(
            img_float <= 0.04045,
            img_float / 12.92,
            np.power((img_float + 0.055) / 1.055, 2.4),
        )
        print(
            "   ✓ [MODULO 0] Input riconosciuto come sRGB. Convertito in spazio Lineare."
        )

    return linear_rgb


def expose_to_the_right(img: np.ndarray, target_white: float = 99.95) -> np.ndarray:

    print(
        f"input image - Pixel where green more dominant than red: {np.where(img[:, :, 0] < img[:, :, 1])}"
    )

    # 2. Identificazione del punto di bianco della luminanza
    y_max = np.percentile(img, target_white)

    print(f"y_max percentile: {y_max} - y_max full: {img.max()}")

    # Evitiamo divisioni per zero o valori nulli
    if y_max < 1e-6:
        return img

    # 3. Calcolo del guadagno scalare UNICO per tutta l'immagine (Stile slider Exposure)
    gain = 1.0 / y_max

    print(f"gain: {gain}")

    # 4. Applicazione dello stesso guadagno su tutti i canali per non alterare la tinta
    img_scaled = img * gain
    print(
        f"img_scaled- Pixel where green more dominant than red: {np.where(img_scaled[:, :, 0] < img_scaled[:, :, 1])}"
    )

    # 3. Identificazione del valore massimo per ogni singolo PIXEL tra i 3 canali
    # Shape: (H, W, 1)
    pixel_max = np.max(img_scaled, axis=-1, keepdims=True)

    print(f"pixel max shape: {pixel_max.shape}")

    # 4. Calcolo del fattore di scala locale:
    # Se pixel_max > 1.0, dividiamo per pixel_max (cioè moltiplichiamo per 1 / pixel_max)
    # Se pixel_max <= 1.0, il fattore rimane 1.0 (nessuna modifica)
    scale_factor = np.where(pixel_max > 1.0, 1.0 / pixel_max, 1.0)

    print(f"scale_factor: min: {scale_factor.min()} - max: {scale_factor.max()}")

    # 5. Applicazione del descaling locale per preservare la tinta (R:G:B invariato)
    img_out = img_scaled * scale_factor
    print(f"img_out: min: {img_out.min()} - max: {img_out.max()}")
    print(
        f"img_out - Pixel where green more dominant than red: {np.where(img_out[:, :, 0] < img_out[:, :, 1])}"
    )

    print(np.sum(np.where(img_out > 1)), img_out.max())

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


def remove_scanner_light(img: np.ndarray, white_threshold: float = 0.97) -> np.ndarray:
    """
    Rimuove il vuoto dello scanner binarizzando l'immagine e prendendo
    il bounding box interno più conservativo per eliminare i tagli obliqui.

    :param rgb_float: Immagine RGB float32 [0.0, 1.0]
    :param white_threshold: Soglia per considerare un pixel come Bianco Puro (Vuoto Scanner)
    :return: Immagine ritagliata priva di qualsiasi pixel di bianco puro
    """

    gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    film_mask = ((gray < white_threshold) * 255).astype(np.uint8)

    kernel = np.ones((10, 10), np.uint8)
    # 3. Chiusura (Morphological Closing: Dilation -> Erosion)
    mask_morph = cv2.morphologyEx(film_mask, cv2.MORPH_CLOSE, kernel)

    for i in range(5):
        # 2. Apertura (Morphological Opening: Erosion -> Dilation)
        mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_OPEN, kernel)

    borders: tuple[int, int, int, int] = find_biggest_masked_rectangle(mask_morph)
    x_min, x_max, y_min, y_max = borders

    # 3. Ritaglia l'immagine originale usando slicing NumPy
    cropped_img = img[x_min:x_max, y_min:y_max]
    return cropped_img
