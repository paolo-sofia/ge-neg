import json
import subprocess
from pathlib import Path
from typing import Any


def get_metadata(image_path: Path) -> dict[str, str | int]:
    """Estrae metadati chiave da una scansione VueScan usando ExifTool."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {image_path}")

    # Tag specifici da estrarre per ottimizzare la lettura
    tags = [
        "-BitsPerSample",  # es. "16 16 16" o 16
        "-Software",  # es. "VueScan 9.7.85"
        "-ColorSpace",  # es. "Uncalibrated" o "sRGB"
        "-ICCProfileName",  # Nome del profilo colore incorporato (se presente)
        "-Make",  # Marca dello scanner
        "-Model",  # Modello dello scanner
        "-PhotometricInterpretation",  # es. RGB o MinIsBlack
    ]

    # Eseguiamo ExifTool richiedendo l'output in formato JSON (-j)
    cmd = ["exiftool", "-j"] + tags + [str(path)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data: dict[str, Any] = json.loads(result.stdout)[0]
    except FileNotFoundError:
        raise RuntimeError(
            "ExifTool non risulta installato o non è presente nel PATH di sistema."
        )
    except Exception as e:
        raise RuntimeError(f"Errore durante l'estrazione dei metadati: {e}") from e

    # Normalizzazione dei dati estratti
    bits_per_sample = data.get("BitsPerSample")
    # Se viene restituito come stringa "16 16 16", prendiamo solo il primo valore
    if isinstance(bits_per_sample, str):
        bits_per_sample = int(bits_per_sample.split()[0])

    return {
        "bits_per_sample": bits_per_sample,
        "software": data.get("Software"),
        "color_space": data.get("ColorSpace"),
        "icc_profile": data.get("ICCProfileName", "N/A"),
        "scanner_make": data.get("Make"),
        "scanner_model": data.get("Model"),
        "photometric": data.get("PhotometricInterpretation"),
    }
