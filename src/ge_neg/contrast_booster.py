import os

import cv2
import numpy as np
import pygad

from ge_neg.utils import apply_s_curve, downsample_for_optimizer


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
            [0.3, 0.6],  # x0
            [1.0, 8.0],  # k
            [0.7, 1.3],  # h
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
            parent_selection_type="tournament",
            mutation_probability=self.mutation_rate,
            # on_generation=self._on_gen,
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
        # print("s curve applied")
        sigma = np.std(image)

        # 3. Penalità relativa per Clipping (Confronto con l'immagine originale)
        # Penalizziamo solo se AUMENTIAMO i pixel bruciati rispetto al file di partenza
        shadows_threshold: float = 0.05
        orig_shadows = np.mean(self.img < shadows_threshold)
        new_shadows = np.mean(image < shadows_threshold)
        shadow_penalty = max(0.0, new_shadows - orig_shadows)

        highlight_threshold: float = 0.99
        orig_highlights = np.mean(self.img > highlight_threshold)
        new_highlights = np.mean(image > highlight_threshold)
        highlight_penalty = max(0.0, new_highlights - orig_highlights)

        clipping_penalty = 70.0 * shadow_penalty + 30 * highlight_penalty

        # 4. Invece di penalizzare k alto, premiamo il delta di contrasto rispetto all'originale
        # Evita che il GA scelga k=1.0 (che lascia o appiattisce il contrasto)
        delta_sigma = sigma - np.std(self.img)

        fitness_value = delta_sigma - clipping_penalty

        return float(fitness_value)
