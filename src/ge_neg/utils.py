import pathlib
from datetime import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import tifffile

def clean_image_for_border_detection(
    img: np.ndarray,
    blur_kernel: tuple[int, int] = (5, 5),
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    gamma: float = 1.2,
) -> np.ndarray:
    img_work = img.copy()
    if img_work.dtype != np.uint8:
        img_work = (np.clip(img_work, 0, 1) * 255).astype(np.uint8)

    blurred = img_work
    # 1. Gaussian Blur tridimensionale (sfoca ogni canale mantenendo il colore)
    for i in range(3):
        blurred = cv2.GaussianBlur(blurred, blur_kernel, 0)

    img_float = (blurred / 255.0).astype(float)

    # 2. Stretching del contrasto basato sui percentili globali
    # Mantiene la proporzione esatta dei canali RGB
    p_low = np.percentile(img_float, low_percentile)
    p_high = np.percentile(img_float, high_percentile)

    # Taglio dei picchi ed espansione lineare nell'intervallo [0, 1]
    img_stretched = np.clip((img_float - p_low) / (p_high - p_low + 1e-7), 0, 1)

    # 3. Correzione Gamma per scurire/definire i toni scuri senza sbiancare la maschera
    # Un valore gamma > 1.0 (es. 1.2 o 1.5) scurisce le ombre mantenendo la saturazione
    img_enhanced = np.power(img_stretched, gamma)
    return img_enhanced

def zonal_system_fitness_penalty(
    img: np.ndarray,
    alpha: float = 1.0,  # Peso per la continuità/morbidezza (Smoothness)
    beta: float = 1.5,  # Peso per la penalizzazione dell'appiattimento (Concentrazione)
    gamma: float = 3.0,  # Peso per il clipping alle estremità (Zone 0 e X)
    tau_ext: float = 0.015,  # Soglia max tollerata per Zone 0 e X (1.5%)
) -> float:
    """
    Calcola la penalità basata sul Sistema Zonale di Ansel Adams (11 Zone).
    L'immagine di input 'img' deve avere valori in intervallo [0.0, 1.0].
    Ritorna un valore float >= 0 che rappresenta la penalità da SOTTRARRE alla fitness.
    """
    # 1. Calcolo dell'istogramma zonale su 11 bin da 0.0 a 1.0
    # np.histogram restituisce i conteggi; dividiamo per il numero totale di pixel
    counts, _ = np.histogram(img, bins=11, range=(0.0, 1.0))
    p = counts / img.size  # Istogramma zonale normalizzato (somma = 1.0)

    # 2. Termine 1: Smoothness (Differenze tra zone adiacenti)
    # Calcola le differenze prime: p[k+1] - p[k]
    diffs = np.diff(p)
    l_smooth = np.sum(diffs**2)

    # 3. Termine 2: Concentrazione (Indice HHI per evitare che 1-2 zone dominino)
    l_conc = np.sum(p**2)

    # 4. Termine 3: Ancoraggio alle estremità (Zone 0 e Zone 10)
    # Penalizza solo se la frazione di pixel supera tau_ext
    p_zone_0_excess = max(0.0, p[0] - tau_ext)
    p_zone_10_excess = max(0.0, p[10] - tau_ext)
    l_ext = p_zone_0_excess + p_zone_10_excess

    # 5. Penalità totale combinata
    unweighted_score = l_smooth + l_conc + l_ext
    total_penalty = 1.0 - np.exp(-unweighted_score)

    return float(total_penalty)


def image_entropy(image: np.ndarray, normalize: bool = False) -> float:
    """Calcola l'entropia media sui 3 canali colore per una fetta di immagine."""
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    num_bins: int = 256

    entropies: list[float] = []
    for i in range(len(image.shape)):
        channel = image[..., i].ravel()
        histogram, _ = np.histogram(
            channel,
            bins=num_bins,
            range=(0.0, 1.0) if channel.max() <= 1.0 else (0, 255),
        )
        p = histogram / histogram.sum()
        p = p[p > 0]

        entropy = -np.sum(p * np.log2(p))

        if normalize:
            # Dividiamo per log2(256) ovvero 8.0
            entropy /= np.log2(num_bins)

        entropies.append(entropy)
    return float(np.mean(entropies))


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


def save_to_file(
    img: np.ndarray, output_path: pathlib.Path, suffix: str
) -> pathlib.Path:
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


def get_hue_angle(img: np.ndarray) -> np.ndarray:
    R, G, B = img[..., 0], img[..., 1], img[..., 2]

    # Componenti cromatiche cartesiane (seno e coseno dell'angolo di Hue)
    # alpha e beta rappresentano le coordinate nello spazio cromatico di CIELAB/HSV
    alpha = 2.0 * R - G - B
    beta = np.sqrt(3.0) * (G - B)

    # Calcola l'angolo di Hue in radianti [-pi, pi]
    hue_angle = np.arctan2(beta, alpha + 1e-7)
    return hue_angle


def compute_hue_shift(img_orig: np.ndarray, img_new: np.ndarray) -> float:
    """
    Calcola la penalità per lo spostamento cromatico/viraggio.

    - Per immagini a COLORE: misura la distorsione della tonalità (Hue Shift).
    - Per immagini B&N (o zone grigie): misura l'introduzione di dominanti di colore (viraggio).
    """
    # 1. Calcola la cromaticità (distanza dall'asse del grigio neutro R=G=B)
    chroma_orig = np.max(img_orig, axis=-1) - np.min(img_orig, axis=-1)

    # Controllo: L'immagine originale è in Bianco e Nero (o priva di colore)?
    is_black_and_white = np.mean(chroma_orig) < 1e-3

    if is_black_and_white:
        # FOTO B&N: Penalizza qualsiasi divergenza tra i canali RGB nell'immagine finale.
        # Se img_new ha R != G != B, significa che la curva ha introdotto una dominante di colore.
        chroma_new = np.max(img_new, axis=-1) - np.min(img_new, axis=-1)
        return float(np.mean(chroma_new))

    hue_orig = get_hue_angle(img_orig)
    hue_new = get_hue_angle(img_new)

    angle_diff = np.abs(hue_orig - hue_new)
    angular_distance = np.minimum(angle_diff, 2.0 * np.pi - angle_diff)

    # Valuta lo shift solo sui pixel con un minimo di colore
    valid_mask = chroma_orig > 0.05

    if not np.any(valid_mask):
        return float(np.mean(angular_distance))

    return float(np.mean(angular_distance[valid_mask]))


def apply_log_logistic_curve(
    img: np.ndarray, x0: float, k: float, h: float
) -> np.ndarray:
    """Applica la curva Log-Logistica con 3 parametri direttamente su ogni canale RGB (0-1).

    - x0: Punto perno/mezze tinte (0.3 - 0.6) -> determina alpha
    - k:  Contrasto/Pendenza (1.0 - 5.0) -> determina beta
    - h:  Controllo della spalla/luci (0.7 - 1.3) -> modula l'uscita
    """
    # 1. Clip di sicurezza sull'immagine e su x0 per evitare errori di calcolo
    img_safe = np.clip(img, 1e-6, 1.0 - 1e-6)
    x0_safe = np.clip(x0, 1e-3, 0.999)

    # 2. Log-Logistica applicata direttamente all'array 3D dei canali RGB
    # Formula esatta di Wikipedia (x^k / (x0^k + x^k)) applicata sui canali R, G, B
    x_k = np.power(img_safe, k)
    x0_k = np.power(x0_safe, k)
    s_shaped = x_k / (x0_k + x_k + 1e-6)

    # 3. Normalizzazione min-max su [0, 1]
    y_min = 0.0
    y_max = 1.0 / (x0_k + 1.0 + 1e-6)
    normalized = (s_shaped - y_min) / (y_max - y_min + 1e-6)

    # 4. Applicazione del parametro h per modulare la spalla delle luci
    img_boosted = np.power(np.clip(normalized, 1e-6, 1.0), h)

    return np.clip(img_boosted, 0.0, 1.0)


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
