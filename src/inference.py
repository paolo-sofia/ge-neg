import argparse
from pathlib import Path

import cv2
import numpy as np
from db import get_parameters


def apply_scurve(img: np.ndarray, x0: float, k: float, h: float) -> np.ndarray:
    """Applica la funzione sigmoidale/S-Curve direttamente con i tre parametri."""
    # Sostituisci con la formula esatta del tuo ContrastBooster
    return h / (1.0 + np.exp(-k * (img - x0)))


def run_inference(input_path: Path, output_path: Path) -> None:
    params = get_parameters(input_path)

    if params is None:
        raise ValueError(
            f"[-] Parametri non trovati nel DB per l'immagine: {input_path}"
        )

    print(
        f"[*] Parametri trovati su DB: x0={params['x0']}, k={params['k']}, h={params['h']}"
    )

    # 1. Caricamento Immagine
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"Impossibile caricare {input_path}")

    # 2. Applicazione eventuale Crop salvato
    if params["crop_top"] is not None:
        img = img[
            params["crop_top"] : params["crop_bottom"],
            params["crop_left"] : params["crop_right"],
        ]

    # 3. Applicazione diretta della curva (Inferenza ultra-rapida)
    img_float = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_out = apply_scurve(img_float, params["x0"], params["k"], params["h"])

    # 4. Salvataggio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = (
        cv2.cvtColor(np.clip(img_out, 0, 1).astype(np.float32), cv2.COLOR_RGB2BGR)
        * 255.0
    ).astype(np.uint8)
    cv2.imwrite(str(output_path), img_bgr)

    print(f"[✔] Inferenza completata e salvata in: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferenza rapida da parametri SQL.")
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    run_inference(args.input, args.output)
