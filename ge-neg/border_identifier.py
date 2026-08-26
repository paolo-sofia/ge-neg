import os

import numpy as np
import pygad


class DynamicEdgeDetector:
    def __init__(
        self,
        z_threshold: float = 3.0,
        min_abs_delta: float = 1.0,
        min_std: float = 0.15,
    ):
        """z_threshold: quanti sigma sopra la media indicano un outlier.

        min_abs_delta: salto minimo assoluto di entropia per considerare un
        bordo (evita falsi positivi su fluttuazioni microscopiche). min_std: std
        minima per le prime slice dove la varianza è zero.
        """
        self.z_threshold = z_threshold
        self.min_abs_delta = min_abs_delta
        self.min_std = min_std
        self.history = []

    def process_slice(self, current_entropy: float) -> bool:
        # Caso 1: Primo elemento in assoluto. Non possiamo fare confronti.
        if not self.history:
            self.history.append(current_entropy)
            return False

        # 1. Calcolo Statistica Dinamica sugli elementi finora accumulati
        mean = np.mean(self.history)

        # Se abbiamo solo 1 elemento, std è 0, quindi usiamo il min_std di sicurezza
        calc_std = np.std(self.history) if len(self.history) > 1 else 0.0
        std = max(calc_std, self.min_std)

        # 2. Soglia dinamica (media + Z * std)
        dynamic_threshold = mean + (self.z_threshold * std)

        # 3. Calcolo della derivata/delta rispetto al valore precedente
        delta = current_entropy - self.history[-1]

        # 4. CONDIZIONE DI BORDO:
        # Il valore supera la soglia dinamica AND il salto è significativo (evita rumore)
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
        num_plateau_iterations: int = 0
        new_direction_value: int = 0
        old_direction_value: int = self.borders[direction]
        is_last_iteration = False

        edge_detector = DynamicEdgeDetector()

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
