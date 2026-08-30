import pathlib
from datetime import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import tifffile


def compute_final_image_metrics(
    img: np.ndarray, bit_depth_str: str
) -> dict[str, str | float | int]:
    """Calcola metriche fotometriche, densitometriche, SNR, analisi colore ed esposizione in EV."""
    if img is None or img.size == 0:
        raise ValueError("Immagine non valida o vuota.")

    img = img.astype(np.float32)
    bit_depth: int = np.dtype(bit_depth_str).itemsize * 8
    max_val = 1.0

    # 1. Conversione in scala di grigi
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        b, g, r = (
            img[:, :, 0],
            img[:, :, 1],
            img[:, :, 2],
        )
    else:
        gray = img
        b = g = r = img

    gray_float = gray.astype(np.float64)

    # -------------------------------------------------------------------------
    # 2. Stima dell'Esposizione (EV Shift & Densità Media)
    # -------------------------------------------------------------------------
    eps = 1e-6
    # Normalizzazione in intervallo [0, 1]
    norm_gray = np.clip(gray_float / max_val, eps, 1.0)

    # Media della luminanza
    signal_mean = float(np.mean(norm_gray))

    # Target ideale: 18% di grigio medio in spazio lineare (~0.18) o ~50% (0.5) se già compresso in gamma
    # Usiamo 0.18 come riferimento standard della fotografia fotometrica:
    target_mid_gray = 0.18
    ev_shift = float(np.log2(signal_mean / target_mid_gray))

    # MAPPA DI DENSITÀ: D = -log10(I / I_max)
    density_map = -np.log10(norm_gray)
    d_min = float(np.percentile(density_map, 0.1))
    d_max = float(np.percentile(density_map, 99.9))
    d_avg = float(np.mean(density_map))
    dynamic_range = d_max - d_min

    # -------------------------------------------------------------------------
    # 3. Signal-to-Noise Ratio (SNR in dB)
    # -------------------------------------------------------------------------
    noise_std = float(np.std(gray_float))
    if noise_std > 0 and signal_mean > 0:
        snr_db = float(20 * np.log10(np.mean(gray_float) / noise_std))
    else:
        snr_db = 0.0

    # -------------------------------------------------------------------------
    # 4. Temperatura Colore e Tinta (CIE Lab)
    # -------------------------------------------------------------------------
    temperature_score, temperature_label = calculate_temperature_score(img)

    # -------------------------------------------------------------------------
    # 5. Clipping e Nitidezza
    # -------------------------------------------------------------------------
    total_pixels = float(gray.size)
    clipped_shadows_pct = float(np.sum(gray == 0) / total_pixels * 100.0)
    clipped_highlights_pct = float(np.sum(gray >= (max_val - 1)) / total_pixels * 100.0)

    gray_8u = (gray_float / max_val * 255.0).astype(np.uint8) if bit_depth > 8 else gray
    sharpness_score = float(cv2.Laplacian(gray_8u, cv2.CV_64F).var())

    return {
        "ev_shift": round(
            ev_shift, 2
        ),  # Scostamento in stop di luce (+1.0 = +1 EV, -0.8 = -0.8 EV)
        "d_avg": round(d_avg, 4),  # Densità media della pellicola
        "d_min": round(d_min, 4),
        "d_max": round(d_max, 4),
        "dynamic_range": round(dynamic_range, 4),
        "snr_db": round(snr_db, 2),
        "temperature_score": round(temperature_score, 2),
        "temperature_label": temperature_label,
        "brightness_mean": round(signal_mean * max_val, 2),
        "contrast_rms": round(noise_std, 2),
        "clipped_shadows_pct": round(clipped_shadows_pct, 3),
        "clipped_highlights_pct": round(clipped_highlights_pct, 3),
        "sharpness_score": round(sharpness_score, 2),
        "final_mean_r": round(float(np.mean(r)), 2),
        "final_mean_g": round(float(np.mean(g)), 2),
        "final_mean_b": round(float(np.mean(b)), 2),
    }


def calculate_temperature_score(
    img: np.ndarray,
) -> tuple[float, str]:
    """Calcola un singolo valore per la temperatura colore.

    Ritorna:
        - temp_score (float): >0 Calda, <0 Fredda, ~0 Bilanciata.
        - temp_label (str): "WARM", "COOL", o "BALANCED".
    """
    if len(img.shape) < 3 or img.shape[2] != 3:
        # Se l'immagine è in scala di grigi pura è perfettamente bilanciata
        return 0.0, "BALANCED"

    # OpenCV usa BGR
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    mean_r = float(np.mean(r))
    mean_g = float(np.mean(g))
    mean_b = float(np.mean(b))

    eps = 1e-6
    # Score normalizzato tra -1.0 e +1.0 circa
    temp_score = (mean_r - mean_b) / (mean_g + eps)

    # Definizione delle soglie di tolleranza
    threshold = 0.05

    if temp_score > threshold:
        temp_label = "WARM"
    elif temp_score < -threshold:
        temp_label = "COOL"
    else:
        temp_label = "BALANCED"

    return round(temp_score, 4), temp_label


def downsample_for_optimizer(
    img: np.ndarray, target_pixels: int = 1_000_000
) -> np.ndarray:
    """Riduce l'immagine di densità a ~2 Megapixel usando Average Pooling (INTER_AREA)."""
    height, width = img.shape[:2]
    total_pixels: float = height * width

    if total_pixels <= target_pixels:
        return img  # Già sotto i 2MP

    # Calcolo del fattore di scala esatto
    scale = np.sqrt(target_pixels / total_pixels)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # INTER_AREA esegue un Average Pooling perfetto sui blocchi di pixel
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


def save_to_file(img: np.ndarray, output_path: pathlib.Path, suffix: str) -> pathlib.Path:
    if output_path.is_dir():
        if suffix:
            output_path = output_path.parent / f"{suffix}.tif"
        else:
            output_path = output_path / "contrast_boosted.tif"
    else:
        if suffix:
            new_filename: str = f"{output_path.stem}_{suffix}{output_path.suffix}"
            output_path = output_path.parent / new_filename

    if output_path.exists():
        current_timestamp = datetime.now(ZoneInfo("Europe/Amsterdam")).strftime(
            "%Y%m%d_%H_%M_%S"
        )
        output_path = (
            output_path.parent
            / f"{output_path.stem}__{current_timestamp}{output_path.suffix}"
        )

    img_clip = np.clip(img, 0.0, 1.0)
    img_16bit = (img_clip * 65535.0).astype(np.uint16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = tifffile.imwrite(
        output_path,
        img_16bit,
        photometric="rgb",
        compression="zstd",  # Opzionale: riduce la dimensione del file senza perdere dati,
        returnoffset=False,
    )

    print(f"Successfully saved image at path {output_path}")
    return output_path


def predict_film_type(film_base_rgb: tuple[float, float, float]) -> str:
    """Predice se il rullino è Bianco e Nero ("BW") o a Colori ("COLOR")

    basandosi sul colore RGB della base della pellicola.
    """
    # 1. Normalizzazione a uint8 per OpenCV (0-255)
    r, g, b = film_base_rgb

    # Se i valori sono in range [0, 1]
    if max(r, g, b) <= 1.0:
        r, g, b = r * 255.0, g * 255.0, b * 255.0
    # Se i valori sono a 16-bit [0, 65535]
    elif max(r, g, b) > 255.0:
        r, g, b = r / 257.0, g / 257.0, b / 257.0

    # Creiamo un piccolo pixel 1x1 in BGR per OpenCV
    pixel_bgr = np.uint8([[[int(b), int(g), int(r)]]])

    # Conversione in HSV (In OpenCV: H in [0, 180], S in [0, 255], V in [0, 255])
    pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)[0][0]
    hue, saturation, value = pixel_hsv[0], pixel_hsv[1], pixel_hsv[2]

    # 2. Soglia sulla Saturazione
    # Una base B&W ha pochissima saturazione (solitamente S < 20-25 su 255)
    SATURATION_THRESHOLD = 22.0

    if saturation < SATURATION_THRESHOLD:
        return "BW"
    else:
        return "COLOR"


def apply_s_curve(img: np.ndarray, x0: float, k: float, h: float) -> np.ndarray:
    """Applica una curva a S parametrica nell'intervallo [0, 1].

    - x0: Punto medio/perno (0.3 - 0.6)
    - k: Pendenza/Contrasto nei toni medi (1.0 - 5.0)
    - h: Compressione/Spalla delle luci (0.7 - 1.3)
    """
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # 1. Calcola la Luminanza sRGB
    Y_orig = 0.2126 * R + 0.7152 * G + 0.0722 * B

    # print(f"Apply s curve with param: x0:{x0} - k:{k} - h:{h}")
    # Formula sigmoidale modificata per il controllo dinamico della spalla
    s_shaped: np.ndarray = 1.0 / (1.0 + np.exp(-k * (Y_orig - x0)))
    # print(f"s_shaped: {s_shaped.shape}")

    # Normalizzazione per garantire che la curva mappi esattamente [0, 1] -> [0, 1]
    y_min: np.ndarray = 1.0 / (1.0 + np.exp(-k * (0.0 - x0)))
    # print(f"y_min; {y_min}")
    y_max: np.ndarray = 1.0 / (1.0 + np.exp(-k * (1.0 - x0)))
    # print(f"y_max; {y_max}")
    normalized: np.ndarray = (s_shaped - y_min) / (y_max - y_min + 1e-6)
    # print(f"normalized; {normalized.shape}")

    # Luminanza trasformata dalla curva
    Y_boosted = np.power(np.clip(normalized, 1e-6, 1.0), h)
    Y_boosted = np.clip(Y_boosted, 0.0, 1.0)

    # 3. Fattore di scala proporzionale (2D)
    scale = np.where(Y_orig > 1e-6, Y_boosted / Y_orig, 1.0)

    # 4. Applicazione del guadagno ai canali RGB e ricomposizione
    img_out = np.dstack([R * scale, G * scale, B * scale])

    return np.clip(img_out, 0.0, 1.0)


def get_luminance(img: np.ndarray) -> np.ndarray:
    """Restituisce la luminanza 2D indipendentemente da B&N o RGB."""
    if img.ndim == 3 and img.shape[2] == 3:
        # Pesi standard percettivi per RGB
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    elif img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    return img