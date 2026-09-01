import os

import cv2
import numpy as np
import pygad

from src.ge_neg.utils import (
    apply_log_logistic_curve,
    apply_s_curve,
    compute_hue_shift,
    downsample_for_optimizer,
    image_entropy,
    zonal_system_fitness_penalty,
)


class ContrastBoosterGenetic:
    def __init__(
        self,
        img: np.ndarray,
        seed: int,
        alpha: float = 3,
        population_size: int = 50,
        num_generations: int = 50,
        num_parents_mating: int = 15,
        mutation_rate: float = 0.33,
    ) -> None:
        self.img: np.ndarray = downsample_for_optimizer(img)
        self.seed = seed
        self.alpha: float = alpha
        self.bounds: list[list[float]] = [
            [0.25, 0.65],  # x0
            [1.0, 8.0],  # k
            [0.5, 1.5],  # h
        ]
        self.normalized_bounds: list[list[float]] = [
            [0.0, 1.0],  # x0
            [0.0, 1.0],  # k
            [0.0, 1.0],  # h
        ]

        self.population_size: int = population_size
        self.num_generations: int = num_generations
        self.num_parents_mating: int = num_parents_mating
        self.mutation_rate: float = mutation_rate
        self.genetic_optimizer: pygad.GA = self._initialize_genetic_optimizer()
        self.fitness_values: list[float] = []

    def _initialize_genetic_optimizer(self) -> pygad.GA:
        return pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            sol_per_pop=self.population_size,
            num_genes=3,
            init_range_low=0,
            init_range_high=1,
            fitness_func=self.fitness_func,
            gene_type=float,
            gene_space={"low": 0.0, "high": 1.0},
            allow_duplicate_genes=True,
            on_generation=self._on_gen,
            parallel_processing=["thread", os.cpu_count()],
            stop_criteria="saturate_10",
            # random_seed=self.seed,
            random_seed=42,
            parent_selection_type="tournament",
            crossover_type="sbx",
            sbx_crossover_eta=40,
            mutation_type="random",
            mutation_probability=self.mutation_rate,
            keep_elitism=1,
            mutation_by_replacement=False,
            save_solutions=True,
            random_mutation_min_val=-1,
            random_mutation_max_val=1,
        )

    def _initialize_population(self, population_size: int) -> np.ndarray:
        bounds: np.ndarray = np.array(self.normalized_bounds)

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
        solutions = np.column_stack(
            (self.genetic_optimizer.solutions, self.genetic_optimizer.solutions_fitness)
        )
        # print("========== ALL SOLUTIONS =========")
        # for i in range(len(solutions)):
        #     if solutions[i, -1] == self.genetic_optimizer.best_solution()[1]:
        #         print(
        #             f"{solutions[i]} - Best solution - generation: {(i // self.population_size) + 1}"
        #         )
        #     else:
        #         print(solutions[i])

    def _get_image(self) -> np.ndarray: ...

    def _get_borders(self, merge: bool = False) -> list[np.ndarray]: ...

    def fitness_func(
        self, genetic_optimizer: pygad.GA, solution: list[float], solution_idx: int
    ) -> float:
        """Calcola la fitness: massimizza la Deviazione Standard e penalizza il Clipping."""
        # 1. Obiettivo principale: Massimizzare il contrasto (Deviazione Standard)
        # print(f"solution: {solution}")
        norm_x0, norm_k, norm_h = solution

        x0 = self.bounds[0][0] + norm_x0 * (self.bounds[0][1] - self.bounds[0][0])
        k = self.bounds[1][0] + norm_k * (self.bounds[1][1] - self.bounds[1][0])
        h = self.bounds[2][0] + norm_h * (self.bounds[2][1] - self.bounds[2][0])

        # image = apply_s_curve(self.img, x0, k, h)
        image = apply_log_logistic_curve(self.img, x0, k, h)

        # 1. Target di contrasto a 0.21 (Premia la vicinanza a 0.21 con la radice)
        sigma = np.std(image)
        sigma_score = np.exp(-15 * abs(sigma - 0.21))

        target_median = 0.48
        current_median = np.median(image)
        # Scala esponenziale: cresce velocemente se ci si allontana da 0.46
        median_score = np.exp(-15 * abs(current_median - target_median))

        # Parametro di severità per la crescita esponenziale delle penalità
        # Più è alto, più la barriera contro il clipping è rigida e precoce
        k_penalty = 3.0

        # 2. Penalità Ombre
        shadows_threshold: float = 0.05
        orig_shadows = np.mean(self.img < shadows_threshold)
        new_shadows = np.mean(image < shadows_threshold)
        shadow_diff = new_shadows - orig_shadows
        shadow_penalty = np.exp(k_penalty * max(0, shadow_diff)) - 1

        # 3. Penalità Luci
        highlight_threshold: float = 0.98
        orig_highlights = np.mean(self.img > highlight_threshold)
        new_highlights = np.mean(image > highlight_threshold)
        highlight_diff = new_highlights - orig_highlights
        highlight_penalty = np.exp(k_penalty * max(0, highlight_diff)) - 1

        # pentaly on loss of information
        original_entropy = image_entropy(self.img, normalize=True)
        new_entropy = image_entropy(image, normalize=True)
        entropy_diff: float = (
            original_entropy - new_entropy
        )  # if new entropy is higher, then it's a bonus, not a penalty
        entropy_penalty = np.exp(k_penalty * max(0, entropy_diff)) - 1

        zonal_system_penalty = zonal_system_fitness_penalty(
            image, alpha=1.0, beta=1.2, gamma=2.5
        )

        hue_shift_penalty = compute_hue_shift(self.img, image)

        # Fitness finale
        fitness_value: float = float(
            sigma_score
            + median_score
            - shadow_penalty
            - highlight_penalty
            - entropy_penalty
            - zonal_system_penalty
            - hue_shift_penalty
        )
        if fitness_value > 1.4:
            print("=" * 100)
            print(f"solution: {np.round([x0, k, h], 3)}")
            print(
                f"median: {np.round(current_median, 3)} - std dev: {np.round(sigma, 3)} - entropy: {np.round(new_entropy, 3)}"
            )
            print(
                f"fitness_value: {np.round(fitness_value, 3)} = {np.round(sigma_score, 3)} + {np.round(median_score, 3)} - ({np.round(shadow_penalty, 3)}) - ({np.round(highlight_penalty, 3)}) - ({np.round(entropy_penalty, 3)}) - ({np.round(zonal_system_penalty, 3)}) - ({np.round(hue_shift_penalty, 3)})"
            )
            print("=" * 100)

        return float(fitness_value)
