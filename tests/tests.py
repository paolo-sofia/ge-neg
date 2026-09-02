import pathlib

import cv2
import numpy as np

from src.ge_neg import border_identifier

path = pathlib.Path(
    "/home/paolo/Immagini/analog_images/bronze/nikon_coolscan/test/img_060_02.tif"
)
output_path = pathlib.Path(
    "/home/paolo/Immagini/analog_images/silver/nikon_coolscan/test/img_059_02.tif"
)


def _clean_image(
    img: np.ndarray,
    blur_kernel: tuple[int, int] = (5, 5),
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    gamma: float = 1.2,
) -> np.ndarray:
    img_work = img.copy()
    if img_work.dtype != np.uint8:
        img_work = (np.clip(img_work, 0, 1) * 255).astype(np.uint8)

    blurred = img_work
    # 1. Gaussian Blur tridimensionale (sfoca ogni canale mantenendo il colore)
    for i in range(3):
        blurred = cv2.GaussianBlur(blurred, blur_kernel, 0)

    img_float = (blurred / 255.0).astype(float)

    # 2. Stretching del contrasto basato sui percentili globali
    # Mantiene la proporzione esatta dei canali RGB
    p_low = np.percentile(img_float, low_percentile)
    p_high = np.percentile(img_float, high_percentile)

    # Taglio dei picchi ed espansione lineare nell'intervallo [0, 1]
    img_stretched = np.clip((img_float - p_low) / (p_high - p_low + 1e-7), 0, 1)

    # 3. Correzione Gamma per scurire/definire i toni scuri senza sbiancare la maschera
    # Un valore gamma > 1.0 (es. 1.2 o 1.5) scurisce le ombre mantenendo la saturazione
    img_enhanced = np.power(img_stretched, gamma)
    return img_enhanced


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
        self.cleaned_image: np.ndarray = _clean_image(img)
        self.image_shape: tuple[int, int, int] = img.shape
        self.step_size: float = step_size
        self.delta_entropy_threshold: float = delta_entropy_threshold
        self.max_plateau_iterations: int = max_plateau_iterations

        self.step_size_x_px: int = int(self.image_shape[1] * self.step_size)
        self.step_size_y_px: int = int(self.image_shape[0] * self.step_size)

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

        limit: int = round(
            self.img.shape[1] * 0.05
        )  # 3.5% is a safe number, usually the border is around 90px, 5% ~= 145px
        number_of_steps: int = round(limit / self.step_size_x_px)
        std_devs: list[float] = []
        for i in range(number_of_steps):
            slice: np.ndarray = self.cleaned_image[
                :, self.step_size_x_px * i : self.step_size_x_px * (i + 1)
            ]
            std_devs.append(float(np.std(slice)))

        std_devs_arr: np.ndarray = np.array(std_devs)
        diff: np.ndarray = np.diff(std_devs_arr)
        border_start_px: int = (int(diff.argmax()) + 1) * self.step_size_x_px


if __name__ == "__main__":
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float: np.ndarray = img.astype(np.float32) / (2 ** (img.dtype.itemsize * 8) - 1)

    border_identifier = BorderIdentifier(img_float)
    border_identifier._find_scanner_frame_border(direction="left")
