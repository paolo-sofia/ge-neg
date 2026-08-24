import os

import cv2
import numpy as np
import pygad


def downsample_for_optimizer(
    img: np.ndarray, target_pixels: int = 1_000_000
) -> np.ndarray:
    """Riduce l'immagine di densità a ~2 Megapixel usando Average Pooling (INTER_AREA)."""
    height, width, channel = img.shape
    total_pixels = height * width

    if total_pixels <= target_pixels:
        return img  # Già sotto i 2MP

    # Calcolo del fattore di scala esatto
    scale = np.sqrt(target_pixels / total_pixels)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # INTER_AREA esegue un Average Pooling perfetto sui blocchi di pixel
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


def apply_s_curve(img: np.ndarray, x0: float, k: float, h: float) -> np.ndarray:
    """Applica una curva a S parametrica nell'intervallo [0, 1].

    - x0: Punto medio/perno (0.3 - 0.6)
    - k: Pendenza/Contrasto nei toni medi (1.0 - 5.0)
    - h: Compressione/Spalla delle luci (0.7 - 1.3)
    """
    # print(f"Apply s curve with param: x0:{x0} - k:{k} - h:{h}")
    # Formula sigmoidale modificata per il controllo dinamico della spalla
    s_shaped: np.ndarray = 1.0 / (1.0 + np.exp(-k * (img - x0)))
    # print(f"s_shaped: {s_shaped.shape}")

    # Normalizzazione per garantire che la curva mappi esattamente [0, 1] -> [0, 1]
    y_min: np.ndarray = 1.0 / (1.0 + np.exp(-k * (0.0 - x0)))
    # print(f"y_min; {y_min}")
    y_max: np.ndarray = 1.0 / (1.0 + np.exp(-k * (1.0 - x0)))
    # print(f"y_max; {y_max}")
    normalized: np.ndarray = (s_shaped - y_min) / (y_max - y_min + 1e-6)
    # print(f"normalized; {normalized.shape}")

    # Modulazione spalla/luci
    out: np.ndarray = np.power(np.clip(normalized, 1e-6, 1.0), h)
    # print(f"out; {out.shape}")
    clipped_output = np.clip(out, 0.0, 1.0)
    # print("s curve applied")
    return clipped_output


class ContrastBoosterGenetic:
    def __init__(
        self,
        img: np.ndarray,
        alpha: float = 3,
        population_size: int = 50,
        num_generations: int = 50,
        num_parents_mating: int = 4,
        mutation_rate: float = 0.1,
    ) -> None:
        self.img: np.ndarray = downsample_for_optimizer(img)
        self.alpha: float = alpha
        self.bounds: list[list[float]] = [
            [0.35, 0.55],  # x0
            [1.2, 4.0],  # k
            [0.8, 1.2],  # h
        ]

        self.population: np.ndarray = self._initialize_population(population_size)
        self.num_generations: int = num_generations
        self.num_parents_mating: int = num_parents_mating
        self.mutation_rate: float = mutation_rate
        self.genetic_optimizer: pygad.GA | None = None
        self.fitness_values: list[float] = []

    def _initialize_genetic_optimizer(self) -> pygad.GA:
        return pygad.GA(
            initial_population=self.population,
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            fitness_func=self.fitness_func,
            gene_space=self.bounds,
            parent_selection_type="tournament",
            mutation_probability=self.mutation_rate,
            on_generation=self._on_gen,
            parallel_processing=["thread", os.cpu_count()],
            stop_criteria="saturate_10",
        )

    def _initialize_population(self, population_size: int) -> np.ndarray:
        bounds: np.ndarray = np.array(self.bounds)

        return np.random.uniform(bounds[:, 0], bounds[:, 1], size=(population_size, 3))

    def _on_gen(self, ga_instance: pygad.GA) -> None:
        print("Generation : ", ga_instance.generations_completed)
        print("Fitness of the best solution :", ga_instance.best_solution()[1])

    def run(self) -> None:
        print(f"✓ [MODULO 4] - Inizializzazione algoritmo genetico")
        self.genetic_optimizer = self._initialize_genetic_optimizer()
        print(f"✓ [MODULO 4] - Esecuzione algoritmo genetico")
        self.genetic_optimizer.run()
        print(f"✓ [MODULO 4] - Algoritmo genetico eseguito")

    def _get_image(self) -> np.ndarray: ...

    def _get_borders(self, merge: bool = False) -> list[np.ndarray]: ...

    def fitness_func(
        self, genetic_optimizer: pygad.GA, solution: list[float], solution_idx: int
    ) -> float:
        """Calcola la fitness: massimizza la Deviazione Standard e penalizza il Clipping."""
        # 1. Obiettivo principale: Massimizzare il contrasto (Deviazione Standard)
        # print(f"solution: {solution}")
        image = apply_s_curve(self.img, *solution)
        # print("s curve applied")
        sigma = np.std(image)
        # print("sigma computed")

        # 2. Penalità per Clipping (Pixel bruciati nelle ombre o nelle luci)
        shadow_clipping = np.mean(image < 0.01)
        # print("shadow_clipping computed")
        highlight_clipping = np.mean(image > 0.99)
        # print("highlight_clipping computed")

        # Pesi per le penalità
        penalty = 3.0 * (shadow_clipping + highlight_clipping)
        # print(f"solution: {solution} --> sigma: {sigma} - penalty: {penalty}")
        return float(sigma - penalty)
