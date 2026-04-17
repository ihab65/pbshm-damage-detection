"""
optimization.py — The Math

Objective functions that guide the Genetic Algorithm for
Sensor Network Placement Optimization (SNPO).
"""

import numpy as np


def MAC(phi_A, phi_B):
    """
    Compute the Modal Assurance Criterion (MAC) matrix.

    The MAC compares mode shapes and evaluates their similarity.

    MAC_ij = |phi_i^T phi_j|^2 / (phi_i^T phi_i)(phi_j^T phi_j)

    Result is a value between 0 (completely different) and 1 (identical).

    Parameters
    ----------
    phi_A : array-like, shape (n_modes_A, n_points)
        First set of mode shapes.
    phi_B : array-like, shape (n_modes_B, n_points)
        Second set of mode shapes.

    Returns
    -------
    ndarray, shape (n_modes_A, n_modes_B)
        The MAC matrix.
    """
    phi_A = np.array(phi_A)
    phi_B = np.array(phi_B)

    n_modes_A, n_points_A = phi_A.shape
    n_modes_B, n_points_B = phi_B.shape

    MAC_matrix = np.zeros((n_modes_A, n_modes_B))

    for i in range(n_modes_A):
        for j in range(n_modes_B):
            numerator = np.abs(np.dot(phi_A[i, :], phi_B[j, :])) ** 2
            denominator = (np.dot(phi_A[i, :], phi_A[i, :])
                           * np.dot(phi_B[j, :], phi_B[j, :]))
            MAC_matrix[i, j] = numerator / denominator

    return MAC_matrix


def fitness_func(ga_instance, solution, solution_idx,
                 PHI, n_modes, num_genes):
    """
    PyGAD fitness function for sensor placement optimization.

    Minimizes off-diagonal MAC values while penalizing the use of
    more sensors (sparsity penalty).

    Parameters
    ----------
    ga_instance : pygad.GA
        The GA instance (required by PyGAD signature).
    solution : ndarray
        Current solution (gene indices). -1 means inactive.
    solution_idx : int
        Index of this solution in the population.
    PHI : ndarray, shape (n_modes, 3, n_sensors)
        Mode shape tensor for all candidate sensors.
    n_modes : int
        Number of modes considered.
    num_genes : int
        Maximum number of genes (sensors).

    Returns
    -------
    float
        Fitness value (higher is better).
    """
    solution = np.delete(solution, solution == -1)
    solution = np.unique(solution)
    phi = PHI[:, :, solution]
    mac = MAC(phi.reshape((n_modes, -1)), phi.reshape((n_modes, -1)))
    np.fill_diagonal(mac, 0)

    # Calculate the sum of non-diagonal elements
    sum_non_diagonal = np.sum(mac)
    num_parameters = len(solution)

    # Penalty for using more parameters
    penalty = num_parameters / num_genes
    fitness = 1 / np.sqrt(sum_non_diagonal) / (1 + penalty)

    return fitness


def custom_initialization(num_solutions, num_genes,
                          init_range_low, init_range_high):
    """
    Create a custom initialization for fixed-length GA solutions
    with -1 (inactive) for unused parameters.

    Parameters
    ----------
    num_solutions : int
        Number of solutions (population size).
    num_genes : int
        Number of genes per solution.
    init_range_low : int
        Lower bound for gene values.
    init_range_high : int
        Upper bound (exclusive) for gene values.

    Returns
    -------
    ndarray, shape (num_solutions, num_genes)
        Initial population array.
    """
    population = []
    for _ in range(num_solutions):
        solution = np.ones(num_genes, dtype=int) * -1
        num_active_genes = np.random.randint(1, num_genes + 1)
        active_positions = np.random.choice(
            range(num_genes), num_active_genes, replace=False
        )
        active_values = np.random.choice(
            range(init_range_low, init_range_high),
            num_active_genes, replace=False
        )
        solution[active_positions] = active_values
        population.append(solution)
    return np.array(population)
