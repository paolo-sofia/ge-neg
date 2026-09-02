import cv2
import numpy as np

from src.ge_neg.utils import clean_image_for_border_detection, image_entropy


class DynamicEdgeDetector:
    def __init__(
        self,
        mad_threshold: float = 3.0,
        min_abs_delta: float = 0.08,
        min_mad: float = 0.003,
        window_size: int = 5,
        verbose: bool = False,
    ):
        """z_threshold: quanti sigma sopra la media indicano un outlier.

        min_abs_delta: salto minimo assoluto di entropia per considerare un
        bordo (evita falsi positivi su fluttuazioni microscopiche). min_std: std
        minima per le prime slice dove la varianza è zero.
        """
        self.mad_threshold: float = mad_threshold
        self.min_abs_delta: float = min_abs_delta
        self.min_mad: float = min_mad
        self.window_size: int = window_size
        self.history: list[float] = []
        self.verbose: bool = verbose

    def process_slice(self, current_value: float) -> bool:
        # 1. GESTIONE LIGHT LEAK INIZIALE:
        # Se la storia ha valori alti e la slice attuale STA SCENDENDO,
        # significa che il valore precedente era un light leak. Lo ripuliamo.
        if self.verbose:
            print("=" * 200)
            print(f"current_value: {current_value}")

        # 1. GESTIONE LIGHT LEAK INIZIALE
        if (
            self.history
            and current_value < self.history[-1]
            and (len(self.history) == 1 or self.history[0] > current_value * 0.1)
        ):
            # Se la prima slice era un picco isolato (es. 19.99), la sostituiamo
            print(
                f"[Light Leak Detected] Pulizia valore anomalo iniziale: {self.history[0]:.4f} -> {current_value:.4f}"
            )
            self.history.clear()
            self.history.append(current_value)
            return False

        # Caso 1: Primo elemento in assoluto. Non possiamo fare confronti.

        if self.verbose:
            print(f"history: {self.history}")
        if not self.history:
            self.history.append(current_value)
            return False

        recent_history = self.history[-self.window_size :]
        hist_array = np.array(recent_history)

        median = float(np.median(hist_array))
        current_mad = float(np.median(np.abs(hist_array - median)))

        # Usiamo il fattore 1.4826 per rendere la MAD equivalente alla Deviazione Standard
        mad: float = max(current_mad, self.min_mad)
        robust_std: float = mad * 1.4826
        dynamic_threshold: float = median + (self.mad_threshold * robust_std)

        # 3. Calcolo della derivata/delta rispetto al valore precedente
        delta = current_value - self.history[-1]

        # 4. CONDIZIONE DI BORDO:
        # Il valore supera la soglia dinamica AND il salto è significativo (evita rumore)
        if self.verbose:
            print(f"current_mad: {current_mad} - mad: {mad} - std: {robust_std}")
            print(
                f"dynamic_threshold: {dynamic_threshold} and delta: {delta} > min_abs_delta {self.min_abs_delta}"
            )

        if len(self.history) < 3:
            if delta > self.min_abs_delta:
                print(
                    f"[MODULE 1] - BORDO RILEVATO ALLA SLICE #{len(self.history) + 1}! ({current_value:.4f} > {dynamic_threshold:.4f})"
                )
                return True
        else:
            if current_value > dynamic_threshold and delta > self.min_abs_delta:
                print(
                    f"[MODULE 1] - BORDO RILEVATO ALLA SLICE #{len(self.history) + 1}! ({current_value:.4f} > {dynamic_threshold:.4f})"
                )
                return True

        # Se è rumore/variabilità normale, aggiungiamo alla baseline
        self.history.append(current_value)
        return False


class BorderIdentifier:
    def __init__(
        self,
        img: np.ndarray[tuple[int, int, int], np.dtype[np.uint8 | np.float32]],
        film_type: str,
        step_size_width: float = 0.002,
        step_size_height: float = 0.005,
        delta_entropy_threshold: float = 3.5,
        max_plateau_iterations: int = 15,
    ) -> None:
        self.img: np.ndarray = img
        self.cleaned_image: np.ndarray = clean_image_for_border_detection(
            img, film_type
        )
        self.image_shape: tuple[int, int, int] = img.shape
        self.step_size_width: float = step_size_width
        self.step_size_height: float = step_size_height
        self.delta_entropy_threshold: float = delta_entropy_threshold
        self.max_plateau_iterations: int = max_plateau_iterations

        self.step_size_x_px: int = int(self.image_shape[1] * self.step_size_width)
        self.step_size_y_px: int = int(self.image_shape[0] * self.step_size_height)

        self.borders: dict[str, int] = {
            "left": 0,
            "right": self.image_shape[1],
            "top": 0,
            "bottom": self.image_shape[0],
        }

    def _find_scanner_frame_border(self, direction: str):
        direction = direction.lower().strip()
        if direction not in ("left", "right"):
            print(
                f"Wrong direction passed as input. pass one of these values as parameters: 'left', 'right'. Value passed to function is : {direction}"
            )
            return

        print(f"[MODULE 1] - Finding {direction} border...")
        limit: int = round(
            self.img.shape[1] * 0.05
        )  # 3.5% is a safe number, usually the border is around 90px, 5% ~= 145px
        number_of_steps: int = max(1, round(limit / self.step_size_x_px))
        slice_processor: DynamicEdgeDetector = DynamicEdgeDetector(
            verbose=False, min_abs_delta=0.03, min_mad=0.0
        )

        for i in range(number_of_steps):
            if direction == "left":
                start = self.step_size_x_px * i
                stop = self.step_size_x_px * (i + 1)
            else:
                stop = self.image_shape[1] - (self.step_size_x_px * i)
                start = self.image_shape[1] - (self.step_size_x_px * (i + 1))

            slice: np.ndarray = self.cleaned_image[:, start:stop]
            is_border: bool = slice_processor.process_slice(float(np.std(slice)))
            if not is_border:
                continue

            border_start_px: int = (i + 1) * self.step_size_x_px
            if direction == "left":
                self.borders[direction] = border_start_px
            else:
                self.borders[direction] = self.image_shape[1] - border_start_px

            print(
                f"Bordo trovato all'indice: {i}. Nuovo bordo {direction}: {self.borders[direction]}"
            )
            return

        return

    def _find_film_border(self, direction: str, verbose: bool = False) -> None:
        direction = direction.lower().strip()
        if direction != "top" and direction != "bottom":
            print(
                f"Wrong direction passed as input. pass one of these values as parameters: 'top', 'bottom'. Value passed to function is : {direction}"
            )
            return

        print(f"[MODULE 1] - Finding {direction} border...")

        limit: int = round(
            self.image_shape[0] * 0.1316
        )  # nikon coolscan scans 38mm, so in theory 1mm per side. there could be a problem with scanning and we end up taking a bit of the old frame + film base + current frame, so to be safe we scan 5mm, which is ~13%

        number_of_steps: int = max(1, round(limit / self.step_size_y_px))
        slice_processor: DynamicEdgeDetector = DynamicEdgeDetector(
            verbose=verbose, min_abs_delta=0.1, min_mad=0.005
        )
        for i in range(number_of_steps):
            if direction == "top":
                start = self.step_size_y_px * i
                stop = self.step_size_y_px * (i + 1)
            else:
                stop = self.image_shape[0] - (self.step_size_y_px * i)
                start = self.image_shape[0] - (self.step_size_y_px * (i + 1))

            slice: np.ndarray = self.cleaned_image[start:stop, :]

            is_border: bool = slice_processor.process_slice(float(np.std(slice)))
            if not is_border:
                continue

            border_start_px: int = (i + 1) * self.step_size_y_px
            if direction == "top":
                self.borders[direction] = border_start_px
            else:
                self.borders[direction] = self.image_shape[0] - border_start_px

            print(
                f"Bordo trovato all'indice: {i}. Nuovo bordo {direction}: {self.borders[direction]}"
            )
            return

        return

    def find_borders(self) -> None:
        print("[MODULO 1] - Find borders")

        self._find_scanner_frame_border(direction="left")
        self._find_scanner_frame_border(direction="right")
        self.cleaned_image = self.cleaned_image[
            self.borders["top"] : self.borders["bottom"],
            self.borders["left"] : self.borders["right"],
        ]
        self._find_film_border(direction="top")
        self._find_film_border(direction="bottom", verbose=True)
        print("[MODULO 1] - All borders found")

    def get_image_coordinates(self) -> tuple[int, int, int, int]:
        return (
            self.borders["top"],
            self.borders["bottom"],
            self.borders["left"],
            self.borders["right"],
        )

    def get_film_base(self) -> np.ndarray:
        crop_mask = np.zeros(shape=self.img.shape, dtype=bool)
        crop_mask[
            self.borders["top"] : self.borders["bottom"],
            self.borders["left"] : self.borders["right"],
        ] = True
        crop_mask[
            :,
            : self.borders["left"],
        ] = False  # we remove the side border which is pure black
        crop_mask[
            :,
            self.borders["right"] :,
        ] = False
        borders_array = self.img[~crop_mask].reshape(-1, 3)
        return np.median(borders_array, axis=0)

    def get_image(self) -> np.ndarray:
        return self.img[
            self.borders["top"] : self.borders["bottom"],
            self.borders["left"] : self.borders["right"],
            :,
        ]

    def get_area_ratio(self) -> float:
        cropped_image: np.ndarray = self.img[
            self.borders["top"] : self.borders["bottom"],
            self.borders["left"] : self.borders["right"],
            :,
        ]

        cropped_image_shape: tuple[int, int, int] = cropped_image.shape
        image_shape: tuple[int, int, int] = self.img.shape
        return float(
            (cropped_image_shape[0] * cropped_image_shape[1])
            / (image_shape[0] * image_shape[1])
        )
