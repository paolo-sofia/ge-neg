import typing

import numpy as np
import pygad

"""
Given the following function:
    y = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=44
What are the best values for the 6 weights (w1 to w6)? We are going to use the genetic algorithm to optimize this function.
"""


class GeneticOptimizer:
    def __init__(
        self,
        initial_population: list[list[float | int]],
        num_generations: int,
        fitness_fn: typing.Callable[[float], float],
        num_genes: int,
        num_parents_mating: int = 4,
        crossover_rate: float = 0.5,
        mutation_rate: float = 0.01,
    ) -> None:
        self.initial_population: list[list[float | int]] = initial_population
        self.num_generations: int = num_generations
        self.fitness_fn: typing.Callable[[float], float] = fitness_fn
        self.num_parents_mating: int = num_parents_mating
        self.crossover_rate: float = crossover_rate
        self.mutation_rate: float = mutation_rate
        self.genetic_optimizer: pygad.GA | None = None
        self.fitness_values: list[float] = []

    def run(self) -> None:
        self._init_genetic_optimizer()
        self.genetic_optimizer.run()

    def _init_genetic_optimizer(self) -> None:
        self.genetic_optimizer = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            sol_per_pop=self.pop_size,
            num_genes=self.num_genes,
            fitness_func=self.fitness_fn,
            on_generation=self.on_generation_end,
        )

    def on_generation_end(self):
        print(f"Generation = {self.genetic_optimizer.generations_completed}")
        print(
            f"Fitness    = {self.genetic_optimizer.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}"
        )
        print(
            f"Change     = {self.genetic_optimizer.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}"
        )
        self.fitness_values.append(
            self.genetic_optimizer.best_solution(
                pop_fitness=self.genetic_optimizer.last_generation_fitness[-1]
            )[1]
        )

    def save(self): ...

    def load(self): ...
