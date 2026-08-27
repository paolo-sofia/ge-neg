import hashlib
import os
import sqlite3
from typing import Optional, Tuple

import cv2
import numpy as np

DB_PATH = "image_processing.db"


def init_db():
    """Inizializza il database e crea le tabelle se non esistono."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        # Tabella 1: Metadati statici (immutabili)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_info (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pixel_hash TEXT NOT NULL UNIQUE,
                file_hash TEXT NOT NULL,
                file_path TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                channels INTEGER NOT NULL,
                bit_depth TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Tabella 2: Storico parametri di processo (relazione 1:N)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS process_parameters (
                param_id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                x0 REAL NOT NULL,
                k REAL NOT NULL,
                h REAL NOT NULL,
                fitness_score REAL,
                shadow_threshold REAL,
                generations_run INTEGER,
                output_path TEXT,
                execution_time_ms INTEGER,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES image_info (image_id) ON DELETE CASCADE
            )
        """)
        conn.commit()


def compute_hashes_and_metadata(file_path: str) -> dict:
    """Calcola hash del file, hash dei pixel e metadati dell'immagine."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File non trovato: {file_path}")

    # 1. Calcolo dell'hash del file (lettura binaria)
    file_hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            file_hasher.update(chunk)
    file_hash = file_hasher.hexdigest()
    file_size_bytes = os.path.getsize(file_path)

    # 2. Calcolo dell'hash dei pixel (OpenCV)
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Impossibile decodificare l'immagine: {file_path}")

    img_bytes = np.ascontiguousarray(img).tobytes()
    pixel_hash = hashlib.sha256(img_bytes).hexdigest()

    # 3. Estrazione dimensioni
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1

    return {
        "file_path": file_path,
        "file_hash": file_hash,
        "pixel_hash": pixel_hash,
        "file_size_bytes": file_size_bytes,
        "width": width,
        "height": height,
        "channels": channels,
        "bit_depth": str(img.dtype),
    }


def get_or_create_image_info(metadata: dict) -> int:
    """Inserisce l'immagine se nuova, altrimenti restituisce l'ID esistente basato sul pixel_hash."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        # Verifica se i pixel esistono già a prescindere dal nome file
        cursor.execute(
            "SELECT image_id FROM image_info WHERE pixel_hash = ?",
            (metadata["pixel_hash"],),
        )
        result = cursor.fetchone()

        if result:
            return result[0]

        cursor.execute(
            """
            INSERT INTO image_info
            (pixel_hash, file_hash, file_path, width, height, channels, bit_depth, file_size_bytes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metadata["pixel_hash"],
                metadata["file_hash"],
                metadata["file_path"],
                metadata["width"],
                metadata["height"],
                metadata["channels"],
                metadata["bit_depth"],
                metadata["file_size_bytes"],
            ),
        )
        return cursor.lastrowid


def insert_process_run(image_id: int, process_data: dict) -> int:
    """Salva una nuova esecuzione dell'algoritmo senza sovrascrivere i dati storici."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO process_parameters
            (image_id, x0, k, h, fitness_score, shadow_threshold, generations_run, output_path, execution_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                image_id,
                process_data["x0"],
                process_data["k"],
                process_data["h"],
                process_data.get("fitness_score"),
                process_data.get("shadow_threshold"),
                process_data.get("generations_run"),
                process_data.get("output_path"),
                process_data.get("execution_time_ms"),
            ),
        )
        return cursor.lastrowid


# Inizializza il DB al momento dell'importazione
init_db()
