import pathlib
from datetime import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import tifffile


def clean_image_for_border_detection(
    img: np.ndarray,
    film_type: str,
    blur_kernel: tuple[int, int] = (9, 9),
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    gamma: float = 1.2,
) -> np.ndarray:
    img_work: np.ndarray = img.copy()
    if img_work.dtype != np.uint8:
        img_work = (np.clip(img_work, 0, 1) * 255).astype(np.uint8)

    # 1. Gaussian Blur tridimensionale (sfoca ogni canale mantenendo il colore)
    blurred = cv2.GaussianBlur(img_work, blur_kernel, 0)

    img_float: np.ndarray = (blurred / 255.0).astype(float)

    # 2. Stretching del contrasto basato sui percentili globali
    # Mantiene la proporzione esatta dei canali RGB
    p_low: float = float(np.percentile(img_float, low_percentile))
    p_high: float = float(np.percentile(img_float, high_percentile))

    # Taglio dei picchi ed espansione lineare nell'intervallo [0, 1]
    img_stretched: np.ndarray = np.clip(
        (img_float - p_low) / (p_high - p_low + 1e-7), 0, 1
    )

    # 3. Correzione Gamma per scurire/definire i toni scuri senza sbiancare la maschera
    # Un valore gamma > 1.0 (es. 1.2 o 1.5) scurisce le ombre mantenendo la saturazione
    img_enhanced: np.ndarray = np.power(img_stretched, gamma)

    if film_type == "BW":
        return cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2GRAY)
    return img_enhanced[..., 0]


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
    orig_img: np.ndarray, img: np.ndarray, bit_depth_str: str
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
    eps: float = 0.003
    clipped_shadows_pct = float(np.sum(gray <= eps) / total_pixels * 100.0)
    clipped_highlights_pct = float(np.sum(gray >= (1.0 - eps)) / total_pixels * 100.0)

    gray_8u = (gray_float / max_val * 255.0).astype(np.uint8) if bit_depth > 8 else gray
    sharpness_score = float(cv2.Laplacian(gray_8u, cv2.CV_64F).var())

    median: np.ndarray = np.median(img, axis=(0, 1))
    mean: np.ndarray = np.mean(img, axis=(0, 1))

    orig_median: np.ndarray = np.median(orig_img, axis=(0, 1))

    return {
        "ev_shift": round(
            ev_shift, 3
        ),  # Scostamento in stop di luce (+1.0 = +1 EV, -0.8 = -0.8 EV)
        "d_avg": round(d_avg, 4),  # Densità media della pellicola
        "d_min": round(d_min, 4),
        "d_max": round(d_max, 4),
        "dynamic_range": round(dynamic_range, 4),
        "snr_db": round(snr_db, 3),
        "temperature_score": round(temperature_score, 3),
        "temperature_label": temperature_label,
        "brightness_mean": round(signal_mean * max_val, 3),
        "contrast_rms": round(noise_std, 2),
        "clipped_shadows_pct": round(clipped_shadows_pct, 3),
        "clipped_highlights_pct": round(clipped_highlights_pct, 3),
        "sharpness_score": round(sharpness_score, 3),
        "final_mean_r": round(float(mean[0]), 3),
        "final_mean_g": round(float(mean[1]), 3),
        "final_mean_b": round(float(mean[2]), 3),
        "final_median_r": round(float(median[0]), 3),
        "final_median_g": round(float(median[1]), 3),
        "final_median_b": round(float(median[2]), 3),
        "pre_median_r": round(float(orig_median[0]), 3),
        "pre_median_g": round(float(orig_median[1]), 3),
        "pre_median_b": round(float(orig_median[2]), 3),
    }


def fitness_function_components(
    orig_img: np.ndarray,
    new_img: np.ndarray,
    film_type: str,
) -> dict[str, float]:
    """Calcola la fitness: massimizza la Deviazione Standard e penalizza il Clipping."""

    # 1. Target di contrasto a 0.21 (Premia la vicinanza a 0.21 con la radice)
    sigma = np.std(new_img)
    sigma_score = np.exp(-15 * abs(sigma - 0.21))

    target_median = 0.48
    current_median = np.median(new_img)
    # Scala esponenziale: cresce velocemente se ci si allontana da 0.46
    median_score = np.exp(-15 * abs(current_median - target_median))

    # Parametro di severità per la crescita esponenziale delle penalità
    # Più è alto, più la barriera contro il clipping è rigida e precoce
    k_penalty = 3.0

    # 2. Penalità Ombre
    shadows_threshold: float = 0.05
    orig_shadows = np.mean(orig_img < shadows_threshold)
    new_shadows = np.mean(new_img < shadows_threshold)
    shadow_diff = new_shadows - orig_shadows
    shadow_penalty = np.exp(k_penalty * max(0, shadow_diff)) - 1

    # 3. Penalità Luci
    highlight_threshold: float = 0.98
    orig_highlights = np.mean(orig_img > highlight_threshold)
    new_highlights = np.mean(new_img > highlight_threshold)
    highlight_diff = new_highlights - orig_highlights
    highlight_penalty = np.exp(k_penalty * max(0, highlight_diff)) - 1

    # pentaly on loss of information
    original_entropy = image_entropy(orig_img, normalize=True)
    new_entropy = image_entropy(new_img, normalize=True)
    entropy_diff: float = (
        original_entropy - new_entropy
    )  # if new entropy is higher, then it's a bonus, not a penalty
    entropy_penalty = np.exp(k_penalty * max(0, entropy_diff)) - 1

    zonal_system_penalty = zonal_system_fitness_penalty(
        new_img, alpha=1.0, beta=1.2, gamma=2.5
    )

    hue_shift_penalty = compute_hue_shift(orig_img, new_img, film_type)

    # Fitness finale
    fitness_value: float = float(
        sigma_score
        + median_score
        - shadow_penalty
        - highlight_penalty
        - entropy_penalty
        - zonal_system_penalty
        - hue_shift_penalty
    )
    if fitness_value > 1.4:
        print("=" * 100)
        print(
            f"median: {np.round(current_median, 3)} - std dev: {np.round(sigma, 3)} - entropy: {np.round(new_entropy, 3)}"
        )
        print(
            f"fitness_value: {np.round(fitness_value, 3)} = {np.round(sigma_score, 3)} + {np.round(median_score, 3)} - ({np.round(shadow_penalty, 3)}) - ({np.round(highlight_penalty, 3)}) - ({np.round(entropy_penalty, 3)}) - ({np.round(zonal_system_penalty, 3)}) - ({np.round(hue_shift_penalty, 3)})"
        )

    return {
        "fitness_score": fitness_value,
        "sigma_score": sigma_score,
        "median_score": median_score,
        "shadow_penalty": shadow_penalty,
        "highlight_penalty": highlight_penalty,
        "entropy_penalty": entropy_penalty,
        "zonal_system_penalty": zonal_system_penalty,
        "hue_shift_penalty": hue_shift_penalty,
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


def predict_film_type(img: np.ndarray) -> str:
    """Predice se il rullino è Bianco e Nero ("BW") o a Colori ("COLOR")

    basandosi sul colore RGB della base della pellicola.
    """
    # 1. Normalizzazione a uint8 per OpenCV (0-255)

    # Se i valori sono in range [0, 1]
    if img.max() <= 1.0:
        img_u8 = (img * 255).astype(np.uint8)
    # Se i valori sono a 16-bit [0, 65535]
    elif img.dtype == np.uint16:
        img_u8 = (img / 255).astype(np.uint8)
    else:
        img_u8 = img

    # Conversione in HSV (In OpenCV: H in [0, 180], S in [0, 255], V in [0, 255])
    pixel_hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)[0][0]
    _, saturation, _ = pixel_hsv[0], pixel_hsv[1], pixel_hsv[2]

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


def compute_hue_shift(
    img_orig: np.ndarray, img_new: np.ndarray, film_type: str
) -> float:
    """
    Calcola la penalità per lo spostamento cromatico/viraggio.

    - Per immagini a COLORE: misura la distorsione della tonalità (Hue Shift).
    - Per immagini B&N (o zone grigie): misura l'introduzione di dominanti di colore (viraggio).
    """
    # 1. Calcola la cromaticità (distanza dall'asse del grigio neutro R=G=B)
    chroma_orig = np.max(img_orig, axis=-1) - np.min(img_orig, axis=-1)

    # Controllo: L'immagine originale è in Bianco e Nero (o priva di colore)?

    if film_type == "BW":
        # FOTO B&N: Penalizza qualsiasi divergenza tra i canali RGB nell'immagine finale.
        # Se img_new ha R != G != B, significa che la curva ha introdotto una dominante di colore.
        # 1. Calcolo della mediana per ciascun canale su scala [0.0, 1.0]
        med_r: float = float(np.median(img_new[:, :, 0]))
        med_g: float = float(np.median(img_new[:, :, 1]))
        med_b: float = float(np.median(img_new[:, :, 2]))

        # 2. Differenze a coppie tra le mediane
        diff_rg_sq: float = (med_r - med_g) ** 2
        diff_rb_sq: float = (med_r - med_b) ** 2
        diff_gb_sq: float = (med_g - med_b) ** 2

        # 3. Drift score normalizzato in [0.0, 1.0]
        # Il divisore 2.0 garantisce che con la massima divergenza il valore sia 1.0
        drift_score: float = np.sqrt((diff_rg_sq + diff_rb_sq + diff_gb_sq) / 2.0)

        # 4. Muro di penalità esponenziale:
        # Se drift_score è < 0.01 la penalità è quasi zero.
        # Se il canale Blu scappa via rispetto agli altri, la penalità tende rapidamente a 1.0.
        penalty: float = 1.0 - np.exp(-drift_score)

        return float(penalty)

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
