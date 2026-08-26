import numpy as np


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
        self.mad_threshold = mad_threshold
        self.min_abs_delta = min_abs_delta
        self.min_mad = min_mad
        self.history = []
        self.verbose = verbose

    def _calculate_mad(self, data: np.ndarray, median: float) -> float:
        """Calcola la Median Absolute Deviation (MAD), alternativa robusta alla
        Std."""
        return float(np.median(np.abs(data - median)))

    def process_slice(self, current_entropy: float) -> bool:
        # 1. GESTIONE LIGHT LEAK INIZIALE:
        # Se la storia ha valori alti e la slice attuale STA SCENDENDO,
        # significa che il valore precedente era un light leak. Lo ripuliamo.
        if self.history and current_entropy < self.history[-1]:
            # Se la prima slice era un picco isolato (es. 19.99), la sostituiamo
            if len(self.history) == 1 or self.history[0] > current_entropy + 2.0:
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


def image_entropy(image: np.ndarray) -> float:
    """Calcola l'entropia media sui 3 canali colore per una fetta di immagine."""
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    entropies: list[float] = []
    for i in range(len(image.shape)):
        channel = image[..., i].ravel()
        histogram, _ = np.histogram(
            channel, bins=256, range=(0.0, 1.0) if channel.max() <= 1.0 else (0, 255)
        )
        p = histogram / histogram.sum()
        p = p[p > 0]
        entropies.append(-np.sum(p * np.log2(p)))
    return float(np.sum(entropies))


class BorderIdentifier:
    def __init__(
        self,
        img: np.ndarray,
        step_size: float = 0.005,
        delta_entropy_threshold: float = 3.5,
        max_plateau_iterations: int = 15,
    ) -> None:
        self.img: np.ndarray = img
        self.step_size: float = step_size
        self.delta_entropy_threshold: float = delta_entropy_threshold
        self.max_plateau_iterations: int = max_plateau_iterations

        self.step_x: int = int(self.img.shape[1] * self.step_size)
        self.step_y: int = int(self.img.shape[0] * self.step_size)

        self.borders: dict[str, int] = {
            "left": 0,
            "right": self.img.shape[1],
            "top": 0,
            "bottom": self.img.shape[0],
        }

        self.max_entropy = image_entropy(
            self.img[
                self.borders["top"] : self.borders["bottom"],
                self.borders["left"] : self.borders["right"],
                :,
            ]
        )

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
        is_last_iteration = False

        edge_detector = DynamicEdgeDetector()  # verbose=direction == "bottom"

        image: np.ndarray
        while True:
            if direction == "left":
                new_direction_value = self.borders[direction] + (self.step_x * i)

                if new_direction_value > self.img.shape[1] // 2:
                    is_last_iteration = True
                    new_direction_value = self.img.shape[1] // 2

                image = self.img[
                    self.borders["top"] : self.borders["bottom"],
                    old_direction_value:new_direction_value,
                    :,
                ]
            elif direction == "right":
                new_direction_value = self.borders[direction] - (self.step_x * i)

                if new_direction_value < self.img.shape[1] // 2:
                    is_last_iteration = True
                    new_direction_value = self.img.shape[1] // 2

                image = self.img[
                    self.borders["top"] : self.borders["bottom"],
                    new_direction_value:old_direction_value,
                    :,
                ]
            elif direction == "top":
                new_direction_value = self.borders[direction] + (self.step_y * i)

                if new_direction_value > self.img.shape[0] // 2:
                    is_last_iteration = True
                    new_direction_value = self.img.shape[0] // 2

                if (
                    old_direction_value == new_direction_value
                    or old_direction_value > new_direction_value
                ):
                    return

                image = self.img[
                    old_direction_value:new_direction_value,
                    self.borders["left"] : self.borders["right"],
                    :,
                ]
            elif direction == "bottom":
                new_direction_value = self.borders[direction] - (self.step_y * i)

                if new_direction_value < self.img.shape[0] // 2:
                    is_last_iteration = True
                    new_direction_value = self.img.shape[0] // 2

                image = self.img[
                    new_direction_value:old_direction_value,
                    self.borders["left"] : self.borders["right"],
                    :,
                ]

            entropy: float = image_entropy(image)
            if edge_detector.process_slice(entropy):
                self.borders[direction] = new_direction_value
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

    def get_borders(self) -> np.ndarray:
        crop_mask = np.zeros(shape=self.img.shape, dtype=bool)
        crop_mask[
            self.borders["top"] : self.borders["bottom"],
            self.borders["left"] : self.borders["right"],
        ] = True
        return self.img[~crop_mask].reshape(-1, 3)

    def get_film_base(self) -> np.ndarray:
        return np.median(self.get_borders(), axis=0)

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
        return (cropped_image.shape[0] * cropped_image.shape[1]) / (
            self.img.shape[0] * self.img.shape[1]
        )
