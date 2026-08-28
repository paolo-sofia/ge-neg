import pathlib

import cv2
import numpy as np
import tifffile


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


def save_to_file(img: np.ndarray, output_path: pathlib.Path, suffix: str) -> bool:
    if output_path.is_dir():
        if suffix:
            output_path = output_path.parent / f"{suffix}.tif"
        else:
            output_path = output_path / "contrast_boosted.tif"
    else:
        if suffix:
            new_filename: str = f"{output_path.stem}_{suffix}{output_path.suffix}"
            output_path = output_path.parent / new_filename

    img_clip = np.clip(img, 0.0, 1.0)
    img_16bit = (img_clip * 65535.0).astype(np.uint16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return tifffile.imwrite(
        output_path,
        img_16bit,
        photometric="rgb",
        compression="zstd",  # Opzionale: riduce la dimensione del file senza perdere dati
    )


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
