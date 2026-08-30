import os

import cv2
import numpy as np
import pygad

from src.ge_neg.utils import apply_s_curve, downsample_for_optimizer, get_luminance


class ContrastBoosterGenetic:
    def __init__(
        self,
        img: np.ndarray,
        seed: int,
        alpha: float = 3,
        population_size: int = 50,
        num_generations: int = 50,
        num_parents_mating: int = 12,
        mutation_rate: float = 0.25,
    ) -> None:
        self.img: np.ndarray = downsample_for_optimizer(img)
        self.seed = seed
        self.alpha: float = alpha
        self.bounds: list[list[float]] = [
            [0.05, 0.85],  # x0
            [1.0, 12.0],  # k
            [0.4, 1.6],  # h
        ]

        self.population: np.ndarray = self._initialize_population(population_size)
        self.num_generations: int = num_generations
        self.num_parents_mating: int = num_parents_mating
        self.mutation_rate: float = mutation_rate
        self.genetic_optimizer: pygad.GA = self._initialize_genetic_optimizer()
        self.fitness_values: list[float] = []

    def _initialize_genetic_optimizer(self) -> pygad.GA:
        return pygad.GA(
            initial_population=self.population,
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            fitness_func=self.fitness_func,
            gene_space=self.bounds,
            allow_duplicate_genes=False,
            on_generation=self._on_gen,
            parallel_processing=["thread", os.cpu_count()],
            stop_criteria="saturate_10",
            # random_seed=self.seed,
            random_seed=42,
            parent_selection_type="tournament",
            K_tournament=3,
            crossover_type="uniform",  # Uniforme invece di single_point per mescolare meglio i 3 geni
            mutation_type="random",
            mutation_probability=self.mutation_rate,
            keep_elitism=2,
            mutation_by_replacement=True,
        )

    def _initialize_population(self, population_size: int) -> np.ndarray:
        bounds: np.ndarray = np.array(self.bounds)

        return np.random.uniform(bounds[:, 0], bounds[:, 1], size=(population_size, 3))

    def _on_gen(self, ga_instance: pygad.GA) -> None:
        print("Generation : ", ga_instance.generations_completed)
        print(f"Best solution found: {ga_instance.best_solution()[0]}")
        print("Fitness of the best solution :", ga_instance.best_solution()[1])

    def run(self) -> None:
        print(f"[MODULO 4] - Inizializzazione algoritmo genetico per aumento contrasto")
        self.genetic_optimizer = self._initialize_genetic_optimizer()
        print(f"[MODULO 4] - Esecuzione algoritmo genetico")
        self.genetic_optimizer.run()
        print(f"[MODULO 4] - Algoritmo genetico eseguito")

    def _get_image(self) -> np.ndarray: ...

    def _get_borders(self, merge: bool = False) -> list[np.ndarray]: ...

    def fitness_func(
        self, genetic_optimizer: pygad.GA, solution: list[float], solution_idx: int
    ) -> float:
        """Calcola la fitness: massimizza la Deviazione Standard e penalizza il Clipping."""
        # 1. Obiettivo principale: Massimizzare il contrasto (Deviazione Standard)
        # print(f"solution: {solution}")
        x0, k, h = solution
        image = apply_s_curve(self.img, x0, k, h)

        # 1. Target di contrasto a 0.21 (Premia la vicinanza a 0.21 con la radice)
        sigma = np.std(image)
        sigma_error = np.sqrt(abs(sigma - 0.21))
        sigma_score = 1.0 - sigma_error

        # 2. Penalità Ombre Alzata a 0.12 (Intercetta i toni scuri prima del nero puro)
        shadows_threshold: float = 0.12
        orig_shadows = np.mean(self.img < shadows_threshold)
        new_shadows = np.mean(image < shadows_threshold)
        # Se aumenta la percentuale di pixel sotto 0.12, applica un peso severo (es. 50.0)
        shadow_penalty = new_shadows - orig_shadows
        shadow_penalty = shadow_penalty if shadow_penalty < 0 else shadow_penalty * 100

        # 3. Penalità Luci
        highlight_threshold: float = 0.98
        orig_highlights = np.mean(self.img > highlight_threshold)
        new_highlights = np.mean(image > highlight_threshold)
        highlight_penalty = new_highlights - orig_highlights
        highlight_penalty = (
            highlight_penalty if highlight_penalty < 0 else highlight_penalty * 100
        )

        # Fitness finale
        fitness_value = sigma_score - shadow_penalty - highlight_penalty
        # print(
        #     f"solution: {solution} - fitness_value: {fitness_value} = {sigma_score} - {shadow_penalty} - {highlight_penalty}"
        # )

        return float(fitness_value)
