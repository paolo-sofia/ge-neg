import argparse
import pathlib
from time import time

import cv2
import numpy as np
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

    return positive_img


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

    channel_means = np.mean(neutral_pixels, axis=0)

    # 4. Normalizzazione rispetto al verde (G è il canale di riferimento di luminanza)
    wb_factors = channel_means[1] / np.clip(channel_means, 1e-6, 1.0)

    # 5. Applicazione del bilanciamento e clipping
    balanced = transmittance * wb_factors

    return np.clip(balanced, 1e-6, 1.0)


def save_to_file(img: np.ndarray, output_path: pathlib.Path, suffix: str) -> bool:

    img_clip = np.clip(img, 0.0, 1.0)
    img_16bit = (img_clip * 65535.0).astype(np.uint16)
    return cv2.imwrite(
        output_path / f"{suffix}.tiff",
        cv2.cvtColor(img_16bit, cv2.COLOR_RGB2BGR),
    )


def full_process(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    print(f"[MODULO 0] - Caricamento e normalizzazione immagine")
    img = load_and_normalize(input_path)

    print(f"[MODULO 0] - Rimozione luce scanner")
    trimmed_image = remove_scanner_light(img)
    save_to_file(trimmed_image, output_path, "scanner_crop")

    print(f"[MODULO 1] - Compensazione dell'esposizione con ETTR")
    ettr = expose_to_the_right(trimmed_image)
    save_to_file(ettr, output_path, "exposure_compensation")

    border_identifier = BorderIdentifier(img=ettr)
    border_identifier.find_borders()
    photo_pixels = border_identifier.get_image()

    save_to_file(photo_pixels, output_path, "tagliata")

    print("[MODULO 2] - Bilanciamento del bianco")
    image_wb = film_base_wb(
        photo_pixels, film_base_color=border_identifier.get_film_base()
    )
    save_to_file(image_wb, output_path, "wb")

    print("[MODULO 3] - Inversione in positivo e bilanciamento della densità")
    positive_img = inversion_and_density_balance(image_wb)
    save_to_file(positive_img, output_path, "positive")

    print("[MODULO 3] - Bilanciamento del bianco della scena")
    img_scene_wb = scene_wb(positive_img)
    save_to_file(img_scene_wb, output_path, "wb_scena")

    print("[MODULO 4] - Miglioramento del contrasto")
    contrast_booster = ContrastBoosterGenetic(img_scene_wb)
    contrast_booster.run()
    x0, k, h = contrast_booster.genetic_optimizer.best_solution()[0]
    print(
        f"[MODULO 4] - Parametri migliori per curva di contrasto: x0 = {x0} - k = {k} - h = {h}"
    )
    contrasted_image = apply_s_curve(
        img_scene_wb, *contrast_booster.genetic_optimizer.best_solution()[0]
    )
    save_to_file(contrasted_image, output_path, "contrast_booster")


def main(
    image_type_to_process: str = "", image_description_to_process: str = ""
) -> None:
    input_path = pathlib.Path("/home/paolo/git/ge-neg/tests/input_images/")
    output_folder = pathlib.Path("/home/paolo/git/ge-neg/tests/output_images/")

    VALID_EXTENSIONS = [".tiff", ".tif"]
    paths = input_path.rglob("**")
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        image_type: str = path.parents[0].stem
        image_description: str = path.stem

        if image_type_to_process and image_type != image_type_to_process:
            continue

        if (
            image_description_to_process
            and image_description != image_description_to_process
        ):
            continue

        output_path = output_folder / image_type / image_description
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            print(
                f" ============================== Processing image {image_type} - {image_description} =============================="
            )
            start_time = time()
            full_process(path, output_path)
            elapsed = time() - start_time

            print(
                f"Image {path.stem} processed in {(elapsed // 60)}:{int(elapsed % 60)} minutes"
            )
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ge-neg",
        description="Processa automaticamente i negativi",
    )
    _ = parser.add_argument("-t", "--type")
    _ = parser.add_argument("-d", "--description")

    args = parser.parse_args()
    main(args.type, args.description)
