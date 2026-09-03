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
                film_roll_id TEXT NOT NULL,
                frame_number TEXT NOT NULL,
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
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            ) STRICT;
        """)

        # Tabella 2: Storico esecuzioni STRICT (PK Composta: pixel_hash + processed_at)
        _ = cursor.execute("""
            CREATE TABLE IF NOT EXISTS process_parameters (
                pixel_hash TEXT NOT NULL,
                processed_at TEXT DEFAULT (datetime('now', 'localtime')),
                status TEXT,
                failure_reason TEXT,
                output_path TEXT,
                filename TEXT,
                execution_time_ms INTEGER,

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

                pre_median_r REAL,
                pre_median_g REAL,
                pre_median_b REAL,

                final_mean_r REAL,
                final_mean_g REAL,
                final_mean_b REAL,
                final_median_r REAL,
                final_median_g REAL,
                final_median_b REAL,

                -- Parametri Algoritmo Genetico & Fitness
                ga_best_solution_generation INTEGER,
                ga_generations_completed INTEGER,
                random_seed INTEGER,
                x0 REAL NOT NULL,
                k REAL NOT NULL,
                h REAL NOT NULL,

                fitness_score REAL,
                fitness_sigma_score REAL,
                fitness_median_score REAL,
                fitness_shadow_penalty REAL,
                fitness_highlight_penalty REAL,
                fitness_entropy_penalty REAL,
                fitness_zonal_system_penalty REAL,
                fitness_hue_shift_penalty REAL,

                -- Image features
                film_type TEXT,
                ev_shift REAL,
                d_avg REAL,
                d_min REAL,
                d_max REAL,
                dynamic_range REAL,
                snr_db REAL,
                brightness_mean REAL,
                contrast_rms REAL,
                clipped_shadows_pct REAL,
                clipped_highlights_pct REAL,
                sharpness_score REAL,
                temperature_score REAL,
                temperature_label TEXT,

                PRIMARY KEY (pixel_hash, processed_at),
                FOREIGN KEY (pixel_hash) REFERENCES image_info (pixel_hash) ON DELETE CASCADE
            ) STRICT;
        """)
        conn.commit()


def save_to_db(
    db_path: str | pathlib.Path, data: dict[str, str | int | float]
) -> tuple[str, str]:
    try:
        with get_connection(db_path) as conn:
            pixel_hash = get_or_create_image_info(conn, data)
            pixel_hash, processed_at = insert_process_run(conn, pixel_hash, data)

        return pixel_hash, processed_at
    except Exception as e:
        print(f"Error while saving to db. {e}")
        return "", ""


def get_or_create_image_info(
    conn: sqlite3.Connection, data: dict[str, str | int | float]
) -> str:
    """Inserisce i metadati se l'immagine è nuova, o aggiorna il path se spostata."""
    cursor = conn.cursor()

    metadata: dict[str, str] = data.get("metadata", {})

    _ = cursor.execute(
        """
        INSERT INTO image_info
        (pixel_hash, file_hash, file_path, film_roll_id, frame_number, width, height, channels, bit_depth, file_size_bytes, is_linear, scanning_software, color_space, scanner_make, scanner_model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(pixel_hash) DO UPDATE SET file_path = excluded.file_path
    """,
        (
            data.get("pixel_hash", ""),
            data.get("file_hash", ""),
            str(data.get("image_path", "")),
            data.get("roll_number", ""),
            data.get("frame_number", ""),
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
    conn: sqlite3.Connection,
    pixel_hash: str,
    process_data: dict[str, str | int],
) -> tuple[str, str]:
    """Inserisce un nuovo record nello storico garantendo il rispetto dei tipi STRICT."""
    print("Inserting data into process_parameters table.")
    cursor = conn.cursor()

    sc_box = process_data.get("scanner_light_borders", (-1, -1, -1, -1))
    cr_box = process_data.get("borders", (-1, -1, -1, -1))
    fb_rgb = process_data.get("film_base", (-1, -1, -1))
    x0, k, h = process_data.get("contrast_booster_solution", (-1.0, -1.0, -1.0))
    feat = process_data.get("processed_image_features", {})

    # Mappa esplicita Nome Colonna -> Valore Tipizzato
    row_data = {
        "pixel_hash": pixel_hash,
        "status": process_data.get("processing_status", "UNKNOWN"),
        "failure_reason": process_data.get("error_message", ""),
        "output_path": str(process_data.get("output_path")),
        "filename": str(process_data.get("output_filename", "")),
        "execution_time_ms": int(process_data.get("execution_time_ms", 0)),
        # Coordinate
        "scanner_light_start_x": int(sc_box[0]),
        "scanner_light_end_x": int(sc_box[1]),
        "scanner_light_start_y": int(sc_box[2]),
        "scanner_light_end_y": int(sc_box[3]),
        "img_start_x": int(cr_box[0]),
        "img_end_x": int(cr_box[1]),
        "img_start_y": int(cr_box[2]),
        "img_end_y": int(cr_box[3]),
        # Base film
        "film_base_red": float(fb_rgb[0]),
        "film_base_green": float(fb_rgb[1]),
        "film_base_blue": float(fb_rgb[2]),
        # Parametri GA e Curva
        "ga_best_solution_generation": int(
            process_data.get("contrast_booster_best_solution_generation", -1)
        ),
        "ga_generations_completed": int(
            process_data.get("contrast_booster_generations_completed", -1)
        ),
        "random_seed": int(process_data.get("seed", -1)),
        "x0": float(x0),
        "k": float(k),
        "h": float(h),
        "fitness_score": float(process_data.get("contrast_booster_fitness", -1.0)),
        # Mediane e Medie
        "pre_median_r": float(feat.get("pre_median_r", -1.0)),
        "pre_median_g": float(feat.get("pre_median_g", -1.0)),
        "pre_median_b": float(feat.get("pre_median_b", -1.0)),
        "final_mean_r": float(feat.get("final_mean_r", -1.0)),
        "final_mean_g": float(feat.get("final_mean_g", -1.0)),
        "final_mean_b": float(feat.get("final_mean_b", -1.0)),
        "final_median_r": float(feat.get("final_median_r", -1.0)),
        "final_median_g": float(feat.get("final_median_g", -1.0)),
        "final_median_b": float(feat.get("final_median_b", -1.0)),
        # Scomposizione Fitness
        "fitness_sigma_score": float(feat.get("fitness_sigma_score", -1.0)),
        "fitness_median_score": float(feat.get("fitness_median_score", -1.0)),
        "fitness_shadow_penalty": float(feat.get("fitness_shadow_penalty", -1.0)),
        "fitness_highlight_penalty": float(feat.get("fitness_highlight_penalty", -1.0)),
        "fitness_entropy_penalty": float(feat.get("fitness_entropy_penalty", -1.0)),
        "fitness_zonal_system_penalty": float(
            feat.get("fitness_zonal_system_penalty", -1.0)
        ),
        "fitness_hue_shift_penalty": float(feat.get("fitness_hue_shift_penalty", -1.0)),
        # Metriche Immagine
        "film_type": str(feat.get("film_type", "UNKNOWN")),
        "ev_shift": float(feat.get("ev_shift", -1.0)),
        "d_avg": float(feat.get("d_avg", -1.0)),
        "d_min": float(feat.get("d_min", -1.0)),
        "d_max": float(feat.get("d_max", -1.0)),
        "dynamic_range": float(feat.get("dynamic_range", -1.0)),
        "snr_db": float(feat.get("snr_db", -1.0)),
        "brightness_mean": float(feat.get("brightness_mean", -1.0)),
        "contrast_rms": float(feat.get("contrast_rms", -1.0)),
        "clipped_shadows_pct": float(feat.get("clipped_shadows_pct", -1.0)),
        "clipped_highlights_pct": float(feat.get("clipped_highlights_pct", -1.0)),
        "sharpness_score": float(feat.get("sharpness_score", -1.0)),
        "temperature_score": float(feat.get("temperature_score", -1.0)),
        "temperature_label": str(feat.get("temperature_label", "UNKNOWN")),
    }

    # Generazione dinamica della query senza possibilità di disallineamento
    columns_str = ", ".join(row_data.keys())
    placeholders_str = ", ".join([f":{k}" for k in row_data.keys()])

    query = f"""
        INSERT INTO process_parameters ({columns_str})
        VALUES ({placeholders_str})
        RETURNING processed_at
    """

    cursor.execute(query, row_data)
    processed_at = cursor.fetchone()[0]
    conn.commit()

    return pixel_hash, processed_at


# Inizializza il DB all'importazione
init_db(db_path)
