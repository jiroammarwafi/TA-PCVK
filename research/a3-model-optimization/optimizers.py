"""
Implementasi Algoritma Optimasi Metaheuristik untuk Tuning LS-SVM:
1. Jaya Optimisation (Rao, 2016; Kar et al., 2021)
2. Particle Swarm Optimization (PSO)
3. Genetic Algorithm (GA)

Referensi:
- Kar, N.B., Nayak, D.R., Babu, K.S., & Zhang, Y.-D. (2021).
  A hybrid feature descriptor with Jaya optimised least squares SVM for facial expression recognition.
  IET Image Processing, 15(7), 1471-1483. Persamaan (5) & (6), Tabel 12.
- Rao, R. (2016). Jaya: A simple and new optimization algorithm for solving constrained
  and unconstrained optimization problems. Int. J. Ind. Eng. Comput. 7(1), 19-34.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class OptimizationResult:
    """Wadah hasil eksekusi algoritma optimasi."""

    def __init__(
        self,
        best_params: np.ndarray,
        best_fitness: float,
        history_fitness: List[float],
        algorithm_name: str,
    ):
        self.best_params = best_params
        self.best_fitness = best_fitness
        self.history_fitness = history_fitness
        self.algorithm_name = algorithm_name

    def __repr__(self) -> str:
        return (
            f"<{self.algorithm_name}: best_fitness={self.best_fitness:.4f}, "
            f"best_params={np.round(self.best_params, 4)}>"
        )


class JayaOptimizer:
    """
    Algoritma Optimasi Jaya (Rao, 2016).
    Tidak memerlukan parameter khusus algoritma (parameterless).

    Pembaruan solusi:
        P'_{x,i,c} = P_{x,i,c} + r1 * (P_{x,i,Best} - |P_{x,i,c}|)
                              - r2 * (P_{x,i,Worst} - |P_{x,i,c}|)

    Parameters
    ----------
    bounds : Sequence[Tuple[float, float]]
        Batas bawah dan atas untuk setiap variabel [(L_1, U_1), (L_2, U_2), ...].
    pop_size : int, default=10
        Jumlah populasi kandidat solusi (Tabel 12 jurnal = 10).
    max_iter : int, default=100
        Jumlah iterasi maksimum (Tabel 12 jurnal = 400).
    random_state : int, default=42
        Seed untuk generator bilangan acak agar reprodusibel.
    maximize : bool, default=True
        True bila fungsi objektif dimaksimalkan (misalnya akurasi klasifikasi).
    """

    def __init__(
        self,
        bounds: Sequence[Tuple[float, float]],
        pop_size: int = 10,
        max_iter: int = 100,
        random_state: int = 42,
        maximize: bool = True,
    ):
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.maximize = maximize

    def optimize(
        self, objective_func: Callable[[np.ndarray], float]
    ) -> OptimizationResult:
        """
        Menjalankan optimasi Jaya pada fungsi objektif.

        Parameters
        ----------
        objective_func : callable
            Fungsi yang menerima vektor 1D parameter dan mengembalikan nilai skalar fitness.
        """
        rng = np.random.RandomState(self.random_state)
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]

        # 1. Inisialisasi populasi acak dalam batas
        pop = rng.uniform(lows, highs, size=(self.pop_size, self.dim))
        fitness = np.array([objective_func(ind) for ind in pop], dtype=np.float64)

        history: List[float] = []

        for it in range(self.max_iter):
            # Identifikasi solusi terbaik dan terburuk
            if self.maximize:
                best_idx = int(np.argmax(fitness))
                worst_idx = int(np.argmin(fitness))
            else:
                best_idx = int(np.argmin(fitness))
                worst_idx = int(np.argmax(fitness))

            best_sol = pop[best_idx].copy()
            worst_sol = pop[worst_idx].copy()
            history.append(float(fitness[best_idx]))

            # Update setiap individu (Persamaan 5 & 6)
            for i in range(self.pop_size):
                r1 = rng.uniform(0.0, 1.0, size=self.dim)
                r2 = rng.uniform(0.0, 1.0, size=self.dim)

                # P'_{x,i,c} = P + r1*(Best - |P|) - r2*(Worst - |P|)
                cand = (
                    pop[i]
                    + r1 * (best_sol - np.abs(pop[i]))
                    - r2 * (worst_sol - np.abs(pop[i]))
                )
                # Proyeksikan kembali ke dalam batas yang valid
                cand = np.clip(cand, lows, highs)

                cand_fit = objective_func(cand)

                # Seleksi Greedy: terima kandidat baru jika lebih baik
                if self.maximize:
                    if cand_fit >= fitness[i]:
                        pop[i] = cand
                        fitness[i] = cand_fit
                else:
                    if cand_fit <= fitness[i]:
                        pop[i] = cand
                        fitness[i] = cand_fit

        if self.maximize:
            best_idx = int(np.argmax(fitness))
        else:
            best_idx = int(np.argmin(fitness))
        history.append(float(fitness[best_idx]))

        return OptimizationResult(
            best_params=pop[best_idx],
            best_fitness=float(fitness[best_idx]),
            history_fitness=history,
            algorithm_name="Jaya",
        )


class PSOOptimizer:
    """
    Particle Swarm Optimization (PSO).
    Sesuai Tabel 12 jurnal: pop_size=10, c1=0.8, c2=0.2, momentum w=1.0.
    """

    def __init__(
        self,
        bounds: Sequence[Tuple[float, float]],
        pop_size: int = 10,
        max_iter: int = 100,
        c1: float = 0.8,
        c2: float = 0.2,
        w: float = 0.8,
        random_state: int = 42,
        maximize: bool = True,
    ):
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.random_state = random_state
        self.maximize = maximize

    def optimize(
        self, objective_func: Callable[[np.ndarray], float]
    ) -> OptimizationResult:
        rng = np.random.RandomState(self.random_state)
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        v_max = (highs - lows) * 0.2

        pos = rng.uniform(lows, highs, size=(self.pop_size, self.dim))
        vel = rng.uniform(-v_max, v_max, size=(self.pop_size, self.dim))

        pbest_pos = pos.copy()
        pbest_fit = np.array([objective_func(p) for p in pos], dtype=np.float64)

        if self.maximize:
            gbest_idx = int(np.argmax(pbest_fit))
        else:
            gbest_idx = int(np.argmin(pbest_fit))

        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = float(pbest_fit[gbest_idx])

        history: List[float] = []

        for it in range(self.max_iter):
            history.append(gbest_fit)

            for i in range(self.pop_size):
                r1 = rng.uniform(0.0, 1.0, size=self.dim)
                r2 = rng.uniform(0.0, 1.0, size=self.dim)

                # Pembaruan kecepatan
                vel[i] = (
                    self.w * vel[i]
                    + self.c1 * r1 * (pbest_pos[i] - pos[i])
                    + self.c2 * r2 * (gbest_pos - pos[i])
                )
                vel[i] = np.clip(vel[i], -v_max, v_max)

                # Pembaruan posisi
                pos[i] = np.clip(pos[i] + vel[i], lows, highs)

                fit = objective_func(pos[i])

                # Update personal best
                is_better = (fit > pbest_fit[i]) if self.maximize else (fit < pbest_fit[i])
                if is_better:
                    pbest_fit[i] = fit
                    pbest_pos[i] = pos[i].copy()

                    # Update global best
                    is_gbest = (fit > gbest_fit) if self.maximize else (fit < gbest_fit)
                    if is_gbest:
                        gbest_fit = float(fit)
                        gbest_pos = pos[i].copy()

        history.append(gbest_fit)

        return OptimizationResult(
            best_params=gbest_pos,
            best_fitness=gbest_fit,
            history_fitness=history,
            algorithm_name="PSO",
        )


class GAOptimizer:
    """
    Genetic Algorithm (GA).
    Sesuai Tabel 12 jurnal: pop_size=10, crossover=0.8, mutation=0.2, elitism.
    """

    def __init__(
        self,
        bounds: Sequence[Tuple[float, float]],
        pop_size: int = 10,
        max_iter: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        random_state: int = 42,
        maximize: bool = True,
    ):
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.maximize = maximize

    def optimize(
        self, objective_func: Callable[[np.ndarray], float]
    ) -> OptimizationResult:
        rng = np.random.RandomState(self.random_state)
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        sigma_m = (highs - lows) * 0.1

        pop = rng.uniform(lows, highs, size=(self.pop_size, self.dim))
        fitness = np.array([objective_func(p) for p in pop], dtype=np.float64)

        history: List[float] = []

        for it in range(self.max_iter):
            if self.maximize:
                best_idx = int(np.argmax(fitness))
            else:
                best_idx = int(np.argmin(fitness))

            history.append(float(fitness[best_idx]))
            best_ind = pop[best_idx].copy()
            best_fit_val = float(fitness[best_idx])

            # Seleksi Turnamen untuk membentuk generasi baru
            new_pop = [best_ind]  # Elitisme: selalu pertahankan individu terbaik

            while len(new_pop) < self.pop_size:
                # Pilih dua induk via turnamen biner
                cand1 = rng.choice(self.pop_size, size=2, replace=False)
                p1_idx = cand1[0] if (fitness[cand1[0]] > fitness[cand1[1]] if self.maximize else fitness[cand1[0]] < fitness[cand1[1]]) else cand1[1]

                cand2 = rng.choice(self.pop_size, size=2, replace=False)
                p2_idx = cand2[0] if (fitness[cand2[0]] > fitness[cand2[1]] if self.maximize else fitness[cand2[0]] < fitness[cand2[1]]) else cand2[1]

                p1 = pop[p1_idx].copy()
                p2 = pop[p2_idx].copy()

                # Crossover Aritmetik
                if rng.uniform(0.0, 1.0) < self.crossover_rate:
                    alpha = rng.uniform(0.0, 1.0)
                    child1 = alpha * p1 + (1.0 - alpha) * p2
                    child2 = alpha * p2 + (1.0 - alpha) * p1
                else:
                    child1, child2 = p1, p2

                # Mutasi Gaussian
                for child in (child1, child2):
                    if len(new_pop) >= self.pop_size:
                        break
                    if rng.uniform(0.0, 1.0) < self.mutation_rate:
                        child += rng.normal(0.0, sigma_m, size=self.dim)
                    child = np.clip(child, lows, highs)
                    new_pop.append(child)

            pop = np.array(new_pop[: self.pop_size])
            fitness = np.array([objective_func(p) for p in pop], dtype=np.float64)
            # Pastikan elitisme tidak hilang
            if self.maximize:
                if np.max(fitness) < best_fit_val:
                    pop[0] = best_ind
                    fitness[0] = best_fit_val
            else:
                if np.min(fitness) > best_fit_val:
                    pop[0] = best_ind
                    fitness[0] = best_fit_val

        if self.maximize:
            best_idx = int(np.argmax(fitness))
        else:
            best_idx = int(np.argmin(fitness))
        history.append(float(fitness[best_idx]))

        return OptimizationResult(
            best_params=pop[best_idx],
            best_fitness=float(fitness[best_idx]),
            history_fitness=history,
            algorithm_name="GA",
        )
