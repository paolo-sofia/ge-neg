import hashlib
import os
import pathlib
import random
import re
from time import perf_counter_ns
from typing import Any

import cv2
import numpy as np

from src.ge_neg.border_identifier import BorderIdentifier
from src.ge_neg.contrast_booster import ContrastBoosterGenetic
from src.ge_neg.db import save_to_db
from src.ge_neg.inversion import inversion_and_density_balance
from src.ge_neg.preprocessing import (
    expose_to_the_right,
    find_biggest_masked_rectangle,
    normalize_image,
)
from src.ge_neg.utils import (
    apply_log_logistic_curve,
    compute_final_image_metrics,
    fitness_function_components,
    predict_film_type,
    save_to_file,
)
from src.ge_neg.white_balance import film_base_wb, scene_wb
from src.metadata import get_metadata


class ImageProcessor:
    def __init__(
        self,
        processed_hashes: list[str],
        image_path: pathlib.Path,
        output_path: pathlib.Path,
        apply_genetic_algorithm: bool = True,
    ) -> None:
        self.processed_hashes: list[str] = processed_hashes
        self.image_path: pathlib.Path = image_path
        self.output_path: pathlib.Path = output_path
        self.apply_genetic_algorithm: bool = apply_genetic_algorithm
        self.output_filename: str = ""
        self.processed_image: np.ndarray = np.empty(shape=(0, 0))
        self.execution_time_ms: float = -1
        self.error_message: str | None = None
        self.processing_status: str = "TO_PROCESS"
        self.roll_number: str = ""
        self.frame_number: str = ""
        self.seed: int = random.randint(1, 1_000_000)

        self.metadata: dict[str, Any] = {}
        self.is_linear: bool = False
        self.channels: int = -1
        self.bit_depth: str = ""
        self.pixel_hash: str = ""
        self.file_hash: str = ""
        self.image_width: int = -1
        self.image_height: int = -1
        self.file_size_bytes: int = -1

        self.scanner_light_borders: tuple[int, int, int, int] = (-1, -1, -1, -1)
        self.borders: tuple[int, int, int, int] = (-1, -1, -1, -1)
        self.film_base: tuple[float, float, float] = (-1.0, -1.0, -1.0)
        self.contrast_booster_solution: tuple[float, float, float] = (-1.0, -1.0, -1.0)
        self.contrast_booster_fitness: float = -1.0
        self.processed_image_features: dict[str, str | int | float] = {}
        self.film_type: str = ""

        self._get_roll_info()

    def _get_roll_info(self) -> None:
        film_roll_folder: str = self.image_path.parent.stem
        match = re.match(r"film_(\d+)", film_roll_folder)
        if match:
            self.roll_number = match.group(1)

        frame_number: str = self.image_path.stem
        match = re.match(r"img_(\d+)_(\d+)", frame_number)
        if match:
            self.frame_number = match.group().split("_")[-1]

    def load_image_and_compute_hashes_and_metadata(self) -> np.ndarray:
        """Calcola hash del file, hash dei pixel grezzi e metadati dell'immagine."""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"File non trovato: {self.image_path}")

        # 1. Hash del file fisico
        file_hasher = hashlib.sha256()
        with open(self.image_path, "rb") as f:
            while chunk := f.read(8192):
                file_hasher.update(chunk)
        self.file_hash = file_hasher.hexdigest()
        self.file_size_bytes = int(os.path.getsize(self.image_path))

        # 2. Hash dei pixel grezzi (OpenCV)
        img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Impossibile decodificare l'immagine: {self.image_path}")

        img_bytes = np.ascontiguousarray(img).tobytes()
        self.pixel_hash = hashlib.sha256(img_bytes).hexdigest()

        self.image_width = img.shape[1]
        self.image_height = img.shape[0]
        self.channels = int(img.shape[2]) if len(img.shape) > 2 else 1
        self.bit_depth = str(img.dtype)

        self.metadata = get_metadata(self.image_path)

        self.is_linear = self.metadata.get("color_space", "") == "Uncalibrated"

        return img

    def remove_scanner_light(
        self, img: np.ndarray, white_threshold: float = 0.97
    ) -> np.ndarray:
        """
        Rimuove il vuoto dello scanner binarizzando l'immagine e prendendo
        il bounding box interno più conservativo per eliminare i tagli obliqui.

        :param rgb_float: Immagine RGB float32 [0.0, 1.0]
        :param white_threshold: Soglia per considerare un pixel come Bianco Puro (Vuoto Scanner)
        :return: Immagine ritagliata priva di qualsiasi pixel di bianco puro
        """

        gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        film_mask = ((gray < white_threshold) * 255).astype(np.uint8)

        kernel = np.ones((10, 10), np.uint8)
        # 3. Chiusura (Morphological Closing: Dilation -> Erosion)
        mask_morph = cv2.morphologyEx(film_mask, cv2.MORPH_CLOSE, kernel)

        for _ in range(5):
            # 2. Apertura (Morphological Opening: Erosion -> Dilation)
            mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_OPEN, kernel)

        borders: tuple[int, int, int, int] = find_biggest_masked_rectangle(mask_morph)
        x_min, x_max, y_min, y_max = borders
        self.scanner_light_borders = (int(x_min), int(x_max), int(y_min), int(y_max))

        # 3. Ritaglia l'immagine originale usando slicing NumPy
        cropped_img = img[x_min:x_max, y_min:y_max]
        return cropped_img

    def run(self, db_path: pathlib.Path) -> str:
        start_time = perf_counter_ns()

        try:
            _ = self.process_image()

        except Exception as e:
            print(e)
            self.error_message = str(e)
            self.processing_status = "FAILURE"
        finally:
            if self.processing_status in ("SUCCESS", "FAILURE"):
                stop_time = perf_counter_ns()
                self.execution_time_ms = (stop_time - start_time) / 1_000_000
                pixel_hash, processed_at = save_to_db(db_path, self.__dict__)
                print(
                    f"[MAIN MODULE] - Image with hash {pixel_hash} processed at time: {processed_at} with status {self.processing_status}"
                )
            else:
                print(
                    f"[MAIN MODULE] - Image has status {self.processing_status}. Skipping save to db"
                )

        # for k, v in self.__dict__.items():
        #     print(k, v)
        return self.pixel_hash

    def process_image(self, debug: bool = False) -> None:
        print("[MODULO 0] - Caricamento e normalizzazione immagine")
        img: np.ndarray = self.load_image_and_compute_hashes_and_metadata()

        if self.pixel_hash in self.processed_hashes:
            self.processing_status = "ALREADY_PROCESSED"
            return

        img = normalize_image(img, self.is_linear)

        print(f"[MODULO 0] - Rimozione luce scanner")
        trimmed_image = self.remove_scanner_light(img)
        if debug:
            _ = save_to_file(trimmed_image, self.output_path, "scanner_crop")

        print("[MODULO 0] - Predicting film type")
        self.film_type = predict_film_type(trimmed_image)
        print("[MODULO 0] - Compensazione dell'esposizione con ETTR")
        ettr = expose_to_the_right(trimmed_image)
        if debug:
            _ = save_to_file(ettr, self.output_path, "exposure_compensation")

        print("[MODULO 1] - Identify borders and compute film base")
        border_identifier = BorderIdentifier(img=ettr, film_type=self.film_type)
        border_identifier.find_borders()
        photo_pixels = border_identifier.get_image()
        film_base = border_identifier.get_film_base()
        self.film_base = tuple(film_base.tolist())
        self.borders = border_identifier.get_image_coordinates()
        print(f"[MODULE 1] - Film base is: {self.film_base}")

        if debug:
            _ = save_to_file(photo_pixels, self.output_path, "tagliata")

        print("[MODULO 2] - Film base white balance")
        image_wb = film_base_wb(photo_pixels, film_base_color=film_base)
        if debug:
            _ = save_to_file(image_wb, self.output_path, "wb")

        print("[MODULO 3] - Inversion and density balance")
        positive_img = inversion_and_density_balance(image_wb)

        if debug:
            _ = save_to_file(positive_img, self.output_path, "positive")

        print("[MODULO 3] - Scene white balance")
        img_scene_wb = scene_wb(positive_img)

        if debug:
            _ = save_to_file(img_scene_wb, self.output_path, "wb_scena")

        if self.apply_genetic_algorithm:
            print("[MODULO 4] - Contrast booster")
            contrast_booster = ContrastBoosterGenetic(
                img_scene_wb, seed=self.seed, film_type=self.film_type
            )
            contrast_booster.run()
            solution = contrast_booster.genetic_optimizer.best_solution()[0]

            # denormalize values
            x0 = float(
                contrast_booster.bounds[0][0]
                + solution[0]
                * (contrast_booster.bounds[0][1] - contrast_booster.bounds[0][0])
            )
            k = float(
                contrast_booster.bounds[1][0]
                + solution[1]
                * (contrast_booster.bounds[1][1] - contrast_booster.bounds[1][0])
            )
            h = float(
                contrast_booster.bounds[2][0]
                + solution[2]
                * (contrast_booster.bounds[2][1] - contrast_booster.bounds[2][0])
            )

            self.contrast_booster_solution = (x0, k, h)
            self.contrast_booster_fitness = float(
                contrast_booster.genetic_optimizer.best_solution()[1]
            )
            print(
                f"""[MODULO 4] - Parametri migliori per curva di contrasto (x0, k, h) = {self.contrast_booster_solution} - Fitness = {self.contrast_booster_fitness}
                ==========================================================================================================================================================="""
            )
            self.processed_image = apply_log_logistic_curve(img_scene_wb, x0, k, h)
        else:
            self.processed_image = img_scene_wb

        output_path: pathlib.Path = save_to_file(
            self.processed_image,
            self.output_path,
            self.image_path.stem if debug else "",
        )

        self.output_filename = output_path.name

        self.processed_image_features = compute_final_image_metrics(
            img_scene_wb, self.processed_image, self.bit_depth
        )

        fitness_fn_components: dict[str, float] = fitness_function_components(
            orig_img=img_scene_wb,
            new_img=self.processed_image,
            film_type=self.film_type,
        )

        self.processed_image_features = (
            self.processed_image_features | fitness_fn_components
        )

        self.processing_status = "SUCCESS"
