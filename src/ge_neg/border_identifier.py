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

    def process_slice(self, slice: np.ndarray) -> bool:
        # 1. GESTIONE LIGHT LEAK INIZIALE:
        # Se la storia ha valori alti e la slice attuale STA SCENDENDO,
        # significa che il valore precedente era un light leak. Lo ripuliamo.
        current_value: float = float(np.std(slice))

        if self.verbose:
            print("=" * 200)
            print(
                f"current_value: {current_value:.4f} - min: {slice.min():.4f} - max: {slice.max():.4f}"
            )

        # 1. GESTIONE LIGHT LEAK INIZIALE
        if (
            self.history
            and current_value * 1.2 < self.history[-1]
            and (len(self.history) == 1)  # or self.history[0] > current_value * 0.1)
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
            print(
                f"current_mad: {current_mad:.4f} - mad: {mad:.4f} - std: {robust_std:.4f}"
            )
            print(
                f"dynamic_threshold: {dynamic_threshold:.4f} and delta: {delta:.4f} > min_abs_delta {self.min_abs_delta:.4f}"
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


import math


def analyze_slices(
    img: np.ndarray, limit_px: int, step_size_px: int, direction: str = "left"
) -> tuple[np.ndarray, np.ndarray]:
    """Calcola mean, std e max per TUTTE le slice sul bordo in un'unica operazione vettorializzata."""
    # 1. Quante slice intere possiamo estrarre?
    num_slices: int = math.ceil(limit_px / step_size_px)
    exact_limit: int = num_slices * step_size_px

    # 2. Ritaglio del bordo (Left o Right)
    if direction == "left":
        crop = img[:, :exact_limit]
    elif direction == "right":  # right
        crop = img[:, -exact_limit:]
        # Invertiamo l'ordine orizzontale se andiamo da destra verso sinistra
        crop = crop[:, ::-1]
    elif direction == "top":
        crop = img[:exact_limit, :]
    else:  # bottom
        crop = img[-exact_limit:, :]
        # Invertiamo l'ordine orizzontale se andiamo da destra verso sinistra
        crop = crop[::-1, :]

    has_channels = crop.ndim == 3
    channels = crop.shape[2] if has_channels else 1

    # 2. Reshape corretto in base all'orientamento
    if direction in ("left", "right"):
        # Le slice dividono la LARGHEZZA (W)
        height = crop.shape[0]
        if has_channels:
            reshaped = crop.reshape(height, num_slices, step_size_px, channels)
            slices_tensor = reshaped.transpose(1, 0, 2, 3)  # (Slice, H, Step, C)
        else:
            reshaped = crop.reshape(height, num_slices, step_size_px)
            slices_tensor = reshaped.transpose(1, 0, 2)  # (Slice, H, Step)
    else:
        # Direzioni TOP / BOTTOM: Le slice dividono l'ALTEZZA (H)
        width = crop.shape[1]
        if has_channels:
            # (Num_Slice, Step_Size, W, C)
            reshaped = crop.reshape(num_slices, step_size_px, width, channels)
            slices_tensor = reshaped  # L'asse delle slice è già all'indice 0!
        else:
            # (Num_Slice, Step_Size, W)
            reshaped = crop.reshape(num_slices, step_size_px, width)
            slices_tensor = reshaped  # L'asse delle slice è già all'indice 0!

    # 3. Calcolo delle metriche sull'asse corretto
    # L'asse 0 identifica la slice, riduciamo su tutti gli altri assi (1, 2, ...)
    reduce_axes = tuple(range(1, slices_tensor.ndim))

    means = np.mean(slices_tensor, axis=reduce_axes)
    stds = np.std(slices_tensor, axis=reduce_axes)
    # medians = np.median(slices_tensor, axis=reduce_axes)

    return means, stds


def find_border_index_safe(
    means: np.ndarray,
    stds: np.ndarray,
    black_threshold: float = 0.005,  # Tolleranza per il "nero da scanner"
    min_signal_jump: float = 0.015,  # Salto minimo della media per considerare un bordo
    gradient_prominence: float = 3.0,  # Quanto deve risaltare il picco della derivata rispetto al rumore
) -> int | None:
    """Restituisce l'indice della slice del bordo o None se il bordo non è presente."""
    # 1. CONTROLLO 1: Esiste del nero puro nelle prime slice?
    # Se le prime 2-3 slice non sono nere, significa che l'immagine parte da subito (NO BORDO)
    if np.any(means[:2] > black_threshold):
        return None  # Nessun bordo nero presente

    # 2. CONTROLLO 2: C'è un'escursione/salto di segnale sufficiente?
    # Se la media rimane quasi piatta in tutte le slice, è solo bordo continuo o rumore
    signal_range = np.ptp(means)  # Peak-to-peak (max - min)
    if signal_range < min_signal_jump:
        return None  # Nessun salto significativo verso l'immagine

    # 3. CALCOLO DELLA DERIVATA PRIMA DELLA MEDIA
    mean_gradient = np.gradient(means)
    max_grad_idx = int(np.argmax(mean_gradient))
    max_grad_val = mean_gradient[max_grad_idx]

    # 4. CONTROLLO 3: Il picco del gradiente è isolato e rilevante?
    # Calcoliamo la mediana del gradiente per misurare il rumore della curva
    grad_noise = np.median(np.abs(mean_gradient)) + 1e-6
    if (max_grad_val / grad_noise) < gradient_prominence:
        return None  # Il salto non è abbastanza marcato da essere un bordo pulito

    # --- BORDO CONFERMATO ---
    # Per non tagliare dentro l'immagine ma posizionarsi all'inizio del gradiente,
    # valutiamo se usare l'indice del picco o l'indice immediatamente precedente.
    return max_grad_idx


def find_edge_by_gradient(
    means: np.ndarray,
    stds: np.ndarray,
    min_gradient_significance: float = 0.05,
) -> int | None:
    """Trova il punto di transizione del bordo basandosi sui picchi di derivata

    senza assumere che il bordo sia nero o perfettamente omogeneo.
    """
    # 1. Gradiente della media (può essere positivo o negativo a seconda che il bordo sia più chiaro o più scuro)
    mean_grad = np.gradient(means)

    # 2. Prendiamo il valore assoluto della derivata della media: misura la "velocità di cambiamento"
    abs_mean_grad = np.abs(mean_grad)

    # 3. Gradiente della deviazione standard: nel punto di transizione la std si impenna per via del contrasto
    std_grad = np.gradient(stds)

    # 4. Combiniamo i due segnali in un unico indicatore di transizione
    # Il prodotto o la somma pesata amplifica l'effetto "scalino"
    combined_transition_signal = abs_mean_grad + std_grad

    # 5. Troviamo la slice con il picco di transizione
    max_idx = int(np.argmax(combined_transition_signal))
    max_value = combined_transition_signal[max_idx]

    # 6. CONTROLLO DI SICUREZZA (nessun bordo presente se il picco è trascurabile)
    # Se l'immagine è omogenea e non ha bordi, la variazione sarà bassissima lungo tutto l'array
    if max_value < min_gradient_significance:
        return None  # Nessun bordo rilevato, l'immagine occupa già tutto il margine

    return max_idx


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

    def _correct_scanner_border(self) -> None:
        """Corrects the side borders by applying the same border size on one side, if one border is found but not the other"""
        left_border: int = self.borders.get("left", 0)
        right_border = self.borders.get("right", 0)
        if 100 > left_border > 80 and right_border == self.image_shape[1]:
            print(
                "[MODULE 1] - Correcting right border. Left border found but not right"
            )
            self.borders["right"] = self.image_shape[1] - left_border
        elif (
            left_border == 0
            and self.image_shape[1] - 80 < right_border < self.image_shape[1] - 100
        ):
            print(
                "[MODULE 1] - Correcting left border. Right border found but not right"
            )
            self.borders["left"] = self.image_shape[1] - right_border

    def _find_scanner_frame_border(self, direction: str, verbose: bool = False) -> None:
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

        means, stds = analyze_slices(
            self.cleaned_image,
            limit_px=limit,
            step_size_px=self.step_size_x_px,
            direction=direction,
        )

        border_idx: int | None = find_border_index_safe(
            means,
            stds,
            black_threshold=0.005,  # Tolleranza per il "nero da scanner"
            min_signal_jump=0.015,  # Salto minimo della media per considerare un bordo
            gradient_prominence=3.0,  # Quanto deve risaltare il picco della derivata rispetto al rumore
        )

        if border_idx:
            border_px: int = self.step_size_x_px * (
                border_idx + 1
            )  # add extra pixel for safety

            self.borders[direction] = (
                border_px if direction == "left" else self.image_shape[1] - border_px
            )
            print(
                f"""Bordo trovato all'indice: {border_idx}. Nuovo bordo {direction}: {self.borders[direction]}"""
            )
            print("=" * 150)
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

        means, stds = analyze_slices(
            self.cleaned_image,
            limit_px=limit,
            step_size_px=self.step_size_y_px,
            direction=direction,
        )

        border_idx: int | None = find_edge_by_gradient(means, stds)
        if border_idx:
            border_px: int = self.step_size_y_px * (
                border_idx + 1
            )  # add extra pixel for safety

            self.borders[direction] = (
                border_px if direction == "top" else self.image_shape[0] - border_px
            )
            print(
                f"""Bordo trovato all'indice: {border_idx}. Nuovo bordo {direction}: {self.borders[direction]}"""
            )
            print("=" * 150)
            return
        return

    # def _find_film_border(self, direction: str, verbose: bool = False) -> None:
    #     direction = direction.lower().strip()
    #     if direction != "top" and direction != "bottom":
    #         print(
    #             f"Wrong direction passed as input. pass one of these values as parameters: 'top', 'bottom'. Value passed to function is : {direction}"
    #         )
    #         return

    #     print(f"[MODULE 1] - Finding {direction} border...")

    #     limit: int = round(
    #         self.image_shape[0] * 0.1316
    #     )  # nikon coolscan scans 38mm, so in theory 1mm per side. there could be a problem with scanning and we end up taking a bit of the old frame + film base + current frame, so to be safe we scan 5mm, which is ~13%

    #     number_of_steps: int = max(1, round(limit / self.step_size_y_px))
    #     slice_processor: DynamicEdgeDetector = DynamicEdgeDetector(verbose=verbose)
    #     for i in range(number_of_steps):
    #         if direction == "top":
    #             start = self.step_size_y_px * i
    #             stop = self.step_size_y_px * (i + 1)
    #         else:
    #             stop = self.image_shape[0] - (self.step_size_y_px * i)
    #             start = self.image_shape[0] - (self.step_size_y_px * (i + 1))

    #         slice: np.ndarray = self.cleaned_image[start:stop, :]

    #         is_border: bool = slice_processor.process_slice(slice)
    #         if not is_border:
    #             continue

    #         border_start_px: int = (i + 1) * self.step_size_y_px
    #         if direction == "top":
    #             self.borders[direction] = border_start_px
    #         else:
    #             self.borders[direction] = self.image_shape[0] - border_start_px

    #         print(
    #             f"Bordo trovato all'indice: {i}. Nuovo bordo {direction}: {self.borders[direction]}"
    #         )
    #         return

    #     return

    def find_borders(self) -> None:
        print("[MODULO 1] - Find borders")

        self._find_scanner_frame_border(direction="left", verbose=False)
        print("=" * 150)
        self._find_scanner_frame_border(direction="right", verbose=False)
        print("=" * 150)
        self._correct_scanner_border()
        self.cleaned_image = self.cleaned_image[
            self.borders["top"] : self.borders["bottom"],
            self.borders["left"] : self.borders["right"],
        ]
        import cv2

        cv2.imwrite(
            "cropped_scanner_border.png", (self.cleaned_image * 255).astype(np.uint8)
        )
        self._find_film_border(direction="top", verbose=False)
        self._find_film_border(direction="bottom", verbose=False)
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
