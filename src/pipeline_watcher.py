import pathlib
import signal
import sys
from pathlib import Path
from time import sleep
from typing import Any

import tomllib
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.ge_neg.config_loader import db_path, valid_extensions
from src.pipeline import ImageProcessor

# Caricamento configurazione TOML
config_path: pathlib.Path = Path(__file__).parent.parent / "config.toml"
watch_folder: pathlib.Path
output_folder: pathlib.Path
valid_extensions: set[str]

if config_path.exists():
    print(f"Reading config at path: {str(config_path)}")
    with open(config_path, "rb") as f:
        config: dict[str, Any] = tomllib.load(f)

    print(f"config: {config}")
    watch_folder = (
        pathlib.Path(config.get("paths", {}).get("watch_dir", Path(__file__)))
        .expanduser()
        .resolve()
    )
    output_folder = (
        pathlib.Path(
            config.get("paths", {}).get("output_dir", Path(__file__) / "output")
        )
        .expanduser()
        .resolve()
    )
    valid_extensions = set(
        config.get("settings", {}).get("valid_extensions", [".tiff", ".tif"])
    )
else:
    watch_folder = Path(__file__)
    output_folder = Path(__file__) / "output"
    valid_extensions = {".tiff", ".tif"}

watch_folder.mkdir(exist_ok=True, parents=True)
output_folder.mkdir(exist_ok=True, parents=True)


class VueScanFileHandler(FileSystemEventHandler):
    def __init__(self) -> None:
        super().__init__()
        self.processed_hashes: list[str] = []

    def on_closed(self, event):
        # Ripristina l'evento di chiusura scrittura file (IN_CLOSE_WRITE)
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        if file_path.suffix.lower() not in valid_extensions:
            return

        print(f"[*] Rilevato nuovo file completo: {file_path}")
        try:
            relative_path = file_path.relative_to(watch_folder)
        except ValueError:
            relative_path = Path(file_path.name)

        # Costruisci l'output path rispettando la struttura originale del rullino
        out_path = output_folder / relative_path
        sleep(2)
        try:
            print(
                f" ============================== Processing image {file_path.stem} =============================="
            )
            image_processor: ImageProcessor = ImageProcessor(
                self.processed_hashes, file_path, out_path
            )
            self.processed_hashes.append(image_processor.run(db_path))

        except Exception as e:
            print(e)


if __name__ == "__main__":
    event_handler = VueScanFileHandler()
    observer = Observer()
    observer.schedule(event_handler, str(watch_folder), recursive=True)
    observer.start()

    print(f"[*] Monitoring attivo su: {watch_folder}")
    print(f"[*] Cartella di destinazione: {output_folder}")

    def stop_signal_handler(signum, frame):
        print("[*] Arresto del watcher in corso...")
        observer.stop()
        observer.join()
        sys.exit(0)

    # Collega CTRL+C (SIGINT) e la terminazione systemd (SIGTERM)
    signal.signal(signal.SIGINT, stop_signal_handler)
    signal.signal(signal.SIGTERM, stop_signal_handler)

    # Mette il thread principale in attesa passiva di segnali (Zero polling)
    signal.pause()
