import pathlib
import sqlite3

from src.ge_neg.config_loader import db_path


def get_connection(db_path: str | pathlib.Path) -> sqlite3.Connection:
    """Crea una connessione attivando il supporto per le Foreign Keys."""
    conn = sqlite3.connect(db_path)
    # Di default SQLite disabilita il vincolo delle Foreign Keys; lo attiviamo esplicitamente
    _ = conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str | pathlib.Path):
    """Inizializza il DB creando tabelle in modalità STRICT."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Tabella 1: Metadati statici in modalità STRICT
        _ = cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_info (
                pixel_hash TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                file_path TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                channels INTEGER NOT NULL,
                bit_depth TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                is_linear TEXT NOT NULL,
                scanning_software TEXT,
                color_space TEXT,
                scanner_make TEXT,
                scanner_model TEXT,
                created_at TEXT DEFAULT (CURRENT_TIMESTAMP)
            ) STRICT;
        """)

        # Tabella 2: Storico esecuzioni STRICT (PK Composta: pixel_hash + processed_at)
        _ = cursor.execute("""
            CREATE TABLE IF NOT EXISTS process_parameters (
                pixel_hash TEXT NOT NULL,
                processed_at TEXT DEFAULT (CURRENT_TIMESTAMP),
                status TEXT,
                failure_reason TEXT,

                -- Coordinate luce scanner
                scanner_light_start_x INTEGER,
                scanner_light_end_x INTEGER,
                scanner_light_start_y INTEGER,
                scanner_light_end_y INTEGER,

                -- Coordinate foto ritagliata senza bordi
                img_start_x INTEGER,
                img_end_x INTEGER,
                img_start_y INTEGER,
                img_end_y INTEGER,

                -- Colore base del film (RGB)
                film_base_red REAL,
                film_base_green REAL,
                film_base_blue REAL,

                -- White Balance della scena (RGB)
                wb_red REAL,
                wb_green REAL,
                wb_blue REAL,

                -- Parametri Algoritmo Genetico & Fitness
                x0 REAL NOT NULL,
                k REAL NOT NULL,
                h REAL NOT NULL,
                fitness_score REAL,

                -- Output ed esecuzione
                output_path TEXT,
                execution_time_ms INTEGER,

                PRIMARY KEY (pixel_hash, processed_at),
                FOREIGN KEY (pixel_hash) REFERENCES image_info (pixel_hash) ON DELETE CASCADE
            ) STRICT;
        """)
        conn.commit()


def save_to_db(
    db_path: str | pathlib.Path, data: dict[str, str | int | float]
) -> tuple[str, str]:
    with get_connection(db_path) as conn:
        pixel_hash = get_or_create_image_info(conn, data)
        pixel_hash, processed_at = insert_process_run(conn, pixel_hash, data)

    return pixel_hash, processed_at


def get_or_create_image_info(
    conn: sqlite3.Connection, data: dict[str, str | int | float]
) -> str:
    """Inserisce i metadati se l'immagine è nuova, o aggiorna il path se spostata."""
    cursor = conn.cursor()

    metadata: dict[str, str] = data.get("metadata", {})

    _ = cursor.execute(
        """
        INSERT INTO image_info
        (pixel_hash, file_hash, file_path, width, height, channels, bit_depth, file_size_bytes, is_linear, scanning_software, color_space, scanner_make, scanner_model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pixel_hash) DO UPDATE SET file_path = excluded.file_path
    """,
        (
            data.get("pixel_hash", ""),
            data.get("file_hash", ""),
            str(data.get("image_path", "")),
            data.get("image_width", -1),
            data.get("image_height", -1),
            data.get("channels", -1),
            data.get("bit_depth", ""),
            data.get("file_size_bytes", -1),
            data.get("is_linear", False),
            metadata.get("software", ""),
            metadata.get("color_space", ""),
            metadata.get("scanner_make", ""),
            metadata.get("scanner_model", ""),
        ),
    )
    conn.commit()
    return str(data.get("pixel_hash", ""))


def insert_process_run(
    conn: sqlite3.Connection, pixel_hash: str, process_data: dict[str, str | int]
) -> tuple[str, str]:
    """Inserisce un nuovo record nello storico garantendo il rispetto dei tipi STRICT."""
    print(f"Inserting data into process_parameters table. data: {process_data}")
    cursor = conn.cursor()

    # Estrazione tuple coordinate con casting difensivo per la modalita STRICT
    sc_box: tuple[int, int, int, int] = process_data.get(
        "scanner_light_box", (-1, -1, -1, -1)
    )
    cr_box: tuple[int, int, int, int] = process_data.get("crop_box", (-1, -1, -1, -1))
    fb_rgb = process_data.get("film_base_rgb", (-1, -1, -1))
    wb_rgb = process_data.get("white_balance_rgb", (-1, -1, -1))

    _ = cursor.execute(
        """
        INSERT INTO process_parameters (
            pixel_hash,
            status,
            failure_reason,
            scanner_light_start_x, scanner_light_end_x, scanner_light_start_y, scanner_light_end_y,
            img_start_x, img_end_x, img_start_y, img_end_y,
            film_base_red, film_base_green, film_base_blue,
            wb_red, wb_green, wb_blue,
            x0, k, h, fitness_score,
            output_path, execution_time_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING processed_at
    """,
        (
            pixel_hash,
            process_data.get("status", "UNKNOWN"),
            process_data.get("failure_reason", "UNKNOWN"),
            int(sc_box[0]),
            sc_box[1],
            sc_box[2],
            sc_box[3],
            cr_box[0],
            cr_box[1],
            cr_box[2],
            cr_box[3],
            fb_rgb[0],
            fb_rgb[1],
            fb_rgb[2],
            wb_rgb[0],
            wb_rgb[1],
            wb_rgb[2],
            float(process_data.get("x0", -1)),
            float(process_data.get("k", -1)),
            float(process_data.get("h", -1)),
            float(process_data.get("fitness_score", -1)),
            str(process_data.get("output_path")),
            int(process_data["execution_time_ms"]),
        ),
    )

    processed_at = cursor.fetchone()[0]
    conn.commit()
    return pixel_hash, processed_at


# Inizializza il DB all'importazione
init_db(db_path)
