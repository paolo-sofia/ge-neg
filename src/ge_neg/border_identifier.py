import cv2
import numpy as np

from src.ge_neg.utils import clean_image_for_border_detection, image_entropy


class DynamicEdgeDetector:
    def __init__(
        self,
        mad_threshold: float = 3.0,
        min_abs_delta: float = 1.0,
        min_mad: float = 0.15,
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
        self.history: list[float] = []
        self.verbose: bool = verbose

    def _calculate_mad(self, data: np.ndarray, median: float) -> float:
        """Calcola la Median Absolute Deviation (MAD), alternativa robusta alla
        Std."""
        return float(np.median(np.abs(data - median)))

    def process_slice(self, current_entropy: float) -> bool:
        # 1. GESTIONE LIGHT LEAK INIZIALE:
        # Se la storia ha valori alti e la slice attuale STA SCENDENDO,
        # significa che il valore precedente era un light leak. Lo ripuliamo.
        if (
            self.history
            and current_entropy < self.history[-1]
            and (len(self.history) == 1 or self.history[0] > current_entropy + 2.0)
        ):
            # Se la prima slice era un picco isolato (es. 19.99), la sostituiamo
            # print(
            #     f"   [Light Leak Detected] Pulizia valore anomalo iniziale: {self.history[0]:.4f} -> {current_entropy:.4f}"
            # )
            self.history.clear()
            self.history.append(current_entropy)
            return False

        # Caso 1: Primo elemento in assoluto. Non possiamo fare confronti.

        if self.verbose:
            print(f"history: {self.history}")
        if not self.history:
            self.history.append(current_entropy)
            return False

        hist_array = np.array(self.history)
        median = float(np.median(hist_array))
        mad = max(self._calculate_mad(hist_array, median), self.min_mad)

        # Usiamo il fattore 1.4826 per rendere la MAD equivalente alla Deviazione Standard
        robust_std = mad * 1.4826
        dynamic_threshold = median + (self.mad_threshold * robust_std)

        # 3. Calcolo della derivata/delta rispetto al valore precedente
        delta = current_entropy - self.history[-1]

        # 4. CONDIZIONE DI BORDO:
        # Il valore supera la soglia dinamica AND il salto è significativo (evita rumore)
        if self.verbose:
            print(f"mean: {mad} - std: {robust_std}")
            print(
                f"current_entropy: {current_entropy} > dynamic_threshold: {dynamic_threshold} and delta: {delta} > min_abs_delta {self.min_abs_delta}"
            )
        if current_entropy > dynamic_threshold and delta > self.min_abs_delta:
            # print(
            #     f"[MODULE 1] - BORDO RILEVATO ALLA SLICE #{len(self.history) + 1}! ({current_entropy:.4f} > {dynamic_threshold:.4f})"
            # )
            return True

        # Se è rumore/variabilità normale, aggiungiamo alla baseline
        self.history.append(current_entropy)
        return False


class BorderIdentifier:
    def __init__(
        self,
        img: np.ndarray[tuple[int, int, int], np.dtype[np.uint8 | np.float32]],
        step_size: float = 0.002,
        delta_entropy_threshold: float = 3.5,
        max_plateau_iterations: int = 15,
    ) -> None:
        self.img: np.ndarray = img
        self.cleaned_image: np.ndarray = clean_image_for_border_detection(img)
        self.image_shape: tuple[int, int, int] = img.shape
        self.step_size: float = step_size
        self.delta_entropy_threshold: float = delta_entropy_threshold
        self.max_plateau_iterations: int = max_plateau_iterations

        self.step_x: int = int(self.image_shape[1] * self.step_size)
        self.step_y: int = int(self.image_shape[0] * self.step_size)

        self.borders: dict[str, int] = {
            "left": 0,
            "right": self.image_shape[1],
            "top": 0,
            "bottom": self.image_shape[0],
        }

    def _find_scanner_frame_border(self, direction: str):
        direction = direction.lower().strip()
        if direction != "left" and direction != "right":
            print(
                f"Wrong direction passed as input. pass one of these values as parameters: 'left', 'right'. Value passed to function is : {direction}"
            )
            return

        

    def _find_border(self, direction: str) -> None:
        direction = direction.lower().strip()
        if (
            direction != "left"
            and direction != "right"
            and direction != "top"
            and direction != "bottom"
        ):
            print(
                f"Wrong direction passed as input. pass one of these values as parameters: 'left', 'right', 'top', 'bottom'. Value passed to function is : {direction}"
            )
            return

        # print(
        #     f"[MODULE 0] - Find {direction} border - Starting value: {self.borders[direction]}"
        # )
        previous_entropies: list[float] = []

        i: int = 1
        new_direction_value: int = 0
        old_direction_value: int = self.borders[direction]
        is_last_iteration: bool = False

        edge_detector = DynamicEdgeDetector()  # verbose=direction == "bottom"

        image: np.ndarray
        limit_perc_width: float = 0.066  # film is 36mm, vuescan scans 38mm. so let's scan the first 2.5mm of the image, which is around 6.6%
        limit_perc_height: float = 0.035  # nikon coolscan has a black border of around 91 pixel per side. total scan is 2869 pixel, ~3.5% of the image

        limit: int
        while True:
            if direction == "left":
                new_direction_value = self.borders[direction] + (self.step_x * i)
                limit = int(self.image_shape[1] * limit_perc_height)

                is_last_iteration = new_direction_value >= limit
                new_direction_value = min(new_direction_value, limit)

                image = self.cleaned_image[
                    self.borders["top"] : self.borders["bottom"],
                    old_direction_value:new_direction_value,
                    :,
                ]
            elif direction == "right":
                new_direction_value = self.borders[direction] - (self.step_x * i)
                limit = int(self.image_shape[1] * (1 - limit_perc_height))

                is_last_iteration = new_direction_value <= limit
                new_direction_value = max(new_direction_value, limit)

                image = self.cleaned_image[
                    self.borders["top"] : self.borders["bottom"],
                    new_direction_value:old_direction_value,
                    :,
                ]
            elif direction == "top":
                new_direction_value = self.borders[direction] + (self.step_y * i)
                limit = int(self.image_shape[0] * limit_perc_width)

                is_last_iteration = new_direction_value >= limit
                new_direction_value = min(new_direction_value, limit)

                if (
                    old_direction_value == new_direction_value
                    or old_direction_value > new_direction_value
                ):
                    return

                image = self.cleaned_image[
                    old_direction_value:new_direction_value,
                    self.borders["left"] : self.borders["right"],
                    :,
                ]
            elif direction == "bottom":
                new_direction_value = self.borders[direction] - (self.step_y * i)

                limit = int(self.image_shape[0] * (1 - limit_perc_width))

                is_last_iteration = new_direction_value <= limit
                new_direction_value = max(new_direction_value, limit)

                image = self.cleaned_image[
                    new_direction_value:old_direction_value,
                    self.borders["left"] : self.borders["right"],
                    :,
                ]

            entropy: float = image_entropy(image)
            if edge_detector.process_slice(entropy):
                self.borders[direction] = (
                    new_direction_value + (self.step_x * i)
                )  # safe addition, we might crop a tiny part of the image, but at least we don't have borders
                return

            if is_last_iteration:
                return

            previous_entropies.append(entropy)
            old_direction_value = new_direction_value
            i += 1

    def find_borders(self) -> None:
        print("[MODULO 1] - Find borders")

        self._find_border(direction="left")
        self._find_border(direction="right")
        self._find_border(direction="top")
        self._find_border(direction="bottom")
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
