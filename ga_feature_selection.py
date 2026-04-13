"""
Topic 2.1: Genetic Algorithm for Feature Selection in Classification Problems

This module extends the existing fuzzy-logic project with a second soft-computing
topic (different vertical) by applying a Genetic Algorithm (GA) to select the best
input features for classifying student performance categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


@dataclass
class GAConfig:
    population_size: int = 10
    generations: int = 8
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    tournament_size: int = 3
    random_seed: int = 42


class GAFeatureSelector:
    """Binary genetic algorithm for feature selection."""

    def __init__(self, x: pd.DataFrame, y: np.ndarray, config: GAConfig):
        self.x = x
        self.y = y
        self.config = config
        self.rng = random.Random(config.random_seed)
        self.n_features = x.shape[1]

    def _fix_chromosome(self, chromosome: List[int]) -> List[int]:
        if sum(chromosome) == 0:
            chromosome[self.rng.randrange(self.n_features)] = 1
        return chromosome

    def _random_chromosome(self) -> List[int]:
        chromosome = [self.rng.randint(0, 1) for _ in range(self.n_features)]
        return self._fix_chromosome(chromosome)

    def _fitness(self, chromosome: List[int]) -> float:
        selected_indices = [idx for idx, bit in enumerate(chromosome) if bit == 1]
        if not selected_indices:
            return 0.0

        x_selected = self.x.iloc[:, selected_indices]
        model = RandomForestClassifier(n_estimators=40, random_state=self.config.random_seed)
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.config.random_seed)
        scores = cross_val_score(model, x_selected, self.y, cv=cv, scoring="accuracy")
        return float(np.mean(scores))

    def _tournament_selection(self, population: List[List[int]], fitness_values: List[float]) -> List[int]:
        candidate_ids = [self.rng.randrange(len(population)) for _ in range(self.config.tournament_size)]
        best_id = max(candidate_ids, key=lambda idx: fitness_values[idx])
        return population[best_id][:]

    def _crossover(self, parent_a: List[int], parent_b: List[int]) -> Tuple[List[int], List[int]]:
        if self.rng.random() >= self.config.crossover_rate or self.n_features < 2:
            return parent_a[:], parent_b[:]

        point = self.rng.randint(1, self.n_features - 1)
        child_a = parent_a[:point] + parent_b[point:]
        child_b = parent_b[:point] + parent_a[point:]
        return self._fix_chromosome(child_a), self._fix_chromosome(child_b)

    def _mutate(self, chromosome: List[int]) -> List[int]:
        for idx in range(self.n_features):
            if self.rng.random() < self.config.mutation_rate:
                chromosome[idx] = 1 - chromosome[idx]
        return self._fix_chromosome(chromosome)

    def run(self) -> Tuple[List[int], float, List[dict]]:
        population = [self._random_chromosome() for _ in range(self.config.population_size)]
        history: List[dict] = []

        best_chromosome: List[int] = []
        best_fitness = -1.0

        for generation in range(1, self.config.generations + 1):
            fitness_values = [self._fitness(chromosome) for chromosome in population]

            generation_best_idx = int(np.argmax(fitness_values))
            generation_best_fitness = fitness_values[generation_best_idx]
            generation_best_chromosome = population[generation_best_idx][:]

            if generation_best_fitness > best_fitness:
                best_fitness = generation_best_fitness
                best_chromosome = generation_best_chromosome[:]

            history.append(
                {
                    "generation": generation,
                    "best_fitness": generation_best_fitness,
                    "avg_fitness": float(np.mean(fitness_values)),
                    "active_features": int(sum(generation_best_chromosome)),
                }
            )

            print(
                f"Generation {generation}/{self.config.generations} | "
                f"Best={generation_best_fitness:.4f} | Avg={np.mean(fitness_values):.4f}",
                flush=True,
            )

            new_population: List[List[int]] = [generation_best_chromosome[:]]
            while len(new_population) < self.config.population_size:
                parent_a = self._tournament_selection(population, fitness_values)
                parent_b = self._tournament_selection(population, fitness_values)

                child_a, child_b = self._crossover(parent_a, parent_b)
                child_a = self._mutate(child_a)
                child_b = self._mutate(child_b)

                new_population.append(child_a)
                if len(new_population) < self.config.population_size:
                    new_population.append(child_b)

            population = new_population

        return best_chromosome, best_fitness, history


def discover_dataset(explicit_path: str | None = None) -> Path:
    if explicit_path:
        candidate = Path(explicit_path)
        if not candidate.exists():
            raise FileNotFoundError(f"Dataset not found: {explicit_path}")
        return candidate

    candidates = sorted(Path(".").glob("student_fuzzy_dataset_*.csv"))
    if candidates:
        return candidates[-1]

    fallback = Path("test_student_dataset.csv")
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "No dataset found. Expected one of: student_fuzzy_dataset_*.csv or test_student_dataset.csv"
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: column.strip().lower() for column in df.columns}
    return df.rename(columns=renamed)


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    expected_features = ["attendance", "internal_marks", "assignment_marks"]

    missing = [name for name in expected_features if name not in df.columns]
    if missing:
        raise ValueError(f"Required feature columns missing: {missing}")

    target_column = "category" if "category" in df.columns else "final_grade_band"
    if target_column not in df.columns:
        raise ValueError("Target column missing. Expected 'category'.")

    x = df[expected_features].copy()
    y_raw = df[target_column].astype(str).values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    return x, y, encoder


def main(dataset_path: str | None = None) -> None:
    path = discover_dataset(dataset_path)
    df = pd.read_csv(path)
    df = normalize_columns(df)

    x, y, encoder = prepare_data(df)
    if len(x) > 220:
        x = x.iloc[:220].copy()
        y = y[:220]
    config = GAConfig()

    selector = GAFeatureSelector(x, y, config)
    best_chromosome, best_score, history = selector.run()

    selected_features = [name for bit, name in zip(best_chromosome, x.columns) if bit == 1]

    summary = pd.DataFrame(
        {
            "feature": x.columns,
            "selected": best_chromosome,
        }
    )
    summary["selected"] = summary["selected"].map({1: "YES", 0: "NO"})

    history_df = pd.DataFrame(history)

    summary.to_csv("ga_feature_selection_results.csv", index=False)
    history_df.to_csv("ga_convergence_history.csv", index=False)

    print("=" * 72)
    print("TOPIC 2.1: GENETIC ALGORITHM FOR FEATURE SELECTION")
    print("=" * 72)
    print(f"Dataset: {path.name}")
    print(f"Samples: {len(df)}")
    print(f"Classes: {list(encoder.classes_)}")
    print("-" * 72)
    print(f"Best cross-validated accuracy: {best_score:.4f}")
    print(f"Selected feature subset: {selected_features}")
    print("-" * 72)
    print("Feature Selection Summary")
    print(summary.to_string(index=False))
    print("-" * 72)
    print("Convergence (first 5 generations)")
    print(history_df.head().to_string(index=False))
    print("-" * 72)
    print("Saved: ga_feature_selection_results.csv")
    print("Saved: ga_convergence_history.csv")


if __name__ == "__main__":
    main()