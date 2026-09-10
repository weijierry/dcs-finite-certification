"""Cross-model checks for selective causal compression.

Family SCC-F1 is a six-state metastable two-basin Markov chain. Family
SCC-RG-F0 is an eight-state hierarchical bit-flip chain with one retained
macro bit and two fast nuisance bits. Standard finite Markov aggregation,
spectral decomposition, response geometry, and principal-angle comparisons
are reused; only the joint DCS certificate is project-specific.
"""

from itertools import product
import json
from pathlib import Path

import numpy as np


def _stochastic_check(matrix):
    return bool(np.all(matrix >= -1e-14) and np.allclose(matrix.sum(axis=1), 1.0))


def _tv(left, right):
    return 0.5 * float(np.abs(np.asarray(left) - np.asarray(right)).sum())


def _js(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    if np.allclose(left, right, rtol=0.0, atol=1e-14):
        return 0.0
    mid = 0.5 * (left + right)
    terms = []
    for first in (left, right):
        mask = first > 0
        terms.append(float(np.sum(first[mask] * np.log(first[mask] / mid[mask]))))
    return max(0.0, 0.5 * sum(terms))


def _partition_metrics(matrix, partition, outcomes, horizons=(1, 2, 4, 8)):
    n = len(matrix)
    block_of = {state: i for i, block in enumerate(partition) for state in block}
    response = {}
    for state in range(n):
        values = []
        for horizon in horizons:
            row = np.linalg.matrix_power(matrix, horizon)[state]
            values.extend(sum(row[j] for j in group) for group in outcomes)
        response[state] = np.asarray(values)

    response_intra = 0.0
    closure = 0.0
    coarse = np.zeros((len(partition), len(partition)))
    for bi, block in enumerate(partition):
        pushes = []
        for state in block:
            push = np.array([sum(matrix[state, j] for j in target) for target in partition])
            pushes.append(push)
        coarse[bi] = np.mean(pushes, axis=0)
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                closure = max(closure, float(np.max(np.abs(pushes[i] - pushes[j]))))
                chunks_i = response[block[i]].reshape(len(horizons), len(outcomes))
                chunks_j = response[block[j]].reshape(len(horizons), len(outcomes))
                distance = np.sqrt(np.mean([_js(a, b) for a, b in zip(chunks_i, chunks_j)]))
                response_intra = max(response_intra, float(distance))

    memory = 0.0
    micro_two = matrix @ matrix
    coarse_two = coarse @ coarse
    for state in range(n):
        actual = [sum(micro_two[state, j] for j in target) for target in partition]
        memory = max(memory, _tv(actual, coarse_two[block_of[state]]))

    return {
        "response_intra_distortion": response_intra,
        "closure_defect": closure,
        "memory_residual": memory,
        "coarse_matrix": coarse.tolist(),
    }


def _contrast_vector(labels):
    vector = np.asarray([1.0 if value == labels[0] else -1.0 for value in labels], float)
    vector -= vector.mean()
    return vector / np.linalg.norm(vector)


def _slow_vector(matrix):
    values, vectors = np.linalg.eig(matrix.T)
    order = sorted(range(len(values)), key=lambda i: abs(values[i]), reverse=True)
    vector = np.real(vectors[:, order[1]])
    vector -= vector.mean()
    return vector / np.linalg.norm(vector), float(np.real(values[order[1]]))


def _slow_observable(matrix):
    """Leading nonconstant right eigenmode, appropriate for observables."""
    values, vectors = np.linalg.eig(matrix)
    order = sorted(range(len(values)), key=lambda i: abs(values[i]), reverse=True)
    vector = np.real(vectors[:, order[1]])
    vector -= vector.mean()
    return vector / np.linalg.norm(vector), complex(values[order[1]])


def stationary_distribution(matrix):
    values, vectors = np.linalg.eig(matrix.T)
    index = min(range(len(values)), key=lambda i: abs(values[i] - 1.0))
    vector = np.real(vectors[:, index])
    if vector.sum() < 0:
        vector = -vector
    vector = np.maximum(vector, 0.0)
    return vector / vector.sum()


def detailed_balance_defect(matrix):
    stationary = stationary_distribution(matrix)
    flow = stationary[:, None] * matrix
    return float(np.max(np.abs(flow - flow.T)))


def _alignment(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return float(abs(np.dot(left, right)) / (np.linalg.norm(left) * np.linalg.norm(right)))


def metastable_matrix(cross_rate=0.01, internal_rate=0.32, asymmetry=0.0):
    """Symmetric two-basin chain, with optional within-basin exit asymmetry."""
    matrix = np.zeros((6, 6))
    for state in range(6):
        basin = 0 if state < 3 else 1
        local = [x for x in range(3 * basin, 3 * basin + 3) if x != state]
        for target in local:
            matrix[state, target] = internal_rate / 2.0
        sign = (state % 3) - 1
        exit_rate = cross_rate * (1.0 + asymmetry * sign)
        matrix[state, 3 + state % 3 if basin == 0 else state % 3] = exit_rate
        matrix[state, state] = 1.0 - internal_rate - exit_rate
    return matrix


def run_metastable(cross_rate=0.01, asymmetry=0.0):
    matrix = metastable_matrix(cross_rate, asymmetry=asymmetry)
    partition = ((0, 1, 2), (3, 4, 5))
    metrics = _partition_metrics(matrix, partition, partition)
    slow, eigenvalue = _slow_vector(matrix)
    response_direction = _contrast_vector((0, 0, 0, 1, 1, 1))
    return {
        "family": "SCC-F1-metastable",
        "parameters": {"cross_rate": cross_rate, "asymmetry": asymmetry},
        "stochastic": _stochastic_check(matrix),
        "second_eigenvalue": eigenvalue,
        "dynamical_response_alignment": _alignment(slow, response_direction),
        **metrics,
    }


BITS = tuple(product((0, 1), repeat=3))


def hierarchical_matrix(macro_flip=0.03, nuisance_flip=0.25):
    """Independent one-step bit flips; only the first bit is retained."""
    matrix = np.zeros((8, 8))
    for i, state in enumerate(BITS):
        for j, target in enumerate(BITS):
            probability = 1.0
            for bit, rate in enumerate((macro_flip, nuisance_flip, nuisance_flip)):
                probability *= rate if state[bit] != target[bit] else 1.0 - rate
            matrix[i, j] = probability
    return matrix


def run_hierarchical(macro_flip=0.03, nuisance_flip=0.25):
    matrix = hierarchical_matrix(macro_flip, nuisance_flip)
    blocks = (
        tuple(i for i, bits in enumerate(BITS) if bits[0] == 0),
        tuple(i for i, bits in enumerate(BITS) if bits[0] == 1),
    )
    metrics = _partition_metrics(matrix, blocks, blocks)
    slow, eigenvalue = _slow_vector(matrix)
    macro_direction = _contrast_vector(tuple(bits[0] for bits in BITS))
    return {
        "family": "SCC-RG-F0-hierarchical",
        "parameters": {"macro_flip": macro_flip, "nuisance_flip": nuisance_flip},
        "stochastic": _stochastic_check(matrix),
        "second_eigenvalue": eigenvalue,
        "dynamical_response_alignment": _alignment(slow, macro_direction),
        "response_rg_alignment": 1.0,
        **metrics,
    }


def circulating_matrix(cross_rate=0.02, heterogeneity=0.0, forward=0.34, backward=0.04):
    """Nonproduct nonreversible two-basin chain with phase-dependent exits."""
    if not 0.0 <= heterogeneity <= 0.9:
        raise ValueError("heterogeneity must lie in [0, 0.9]")
    matrix = np.zeros((8, 8))
    phase_profile = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)
    for state in range(8):
        basin, phase = divmod(state, 4)
        matrix[state, 4 * basin + (phase + 1) % 4] = forward
        matrix[state, 4 * basin + (phase - 1) % 4] = backward
        exit_rate = cross_rate * (1.0 + heterogeneity * phase_profile[phase])
        matrix[state, 4 * (1 - basin) + phase] = exit_rate
        matrix[state, state] = 1.0 - forward - backward - exit_rate
    return matrix


def run_circulating(heterogeneity=0.0):
    matrix = circulating_matrix(heterogeneity=heterogeneity)
    blocks = (tuple(range(4)), tuple(range(4, 8)))
    metrics = _partition_metrics(matrix, blocks, blocks)
    slow, eigenvalue = _slow_observable(matrix)
    macro_direction = _contrast_vector((0, 0, 0, 0, 1, 1, 1, 1))
    return {
        "family": "SCC-F2-circulating-nonreversible",
        "parameters": {"heterogeneity": heterogeneity, "forward": 0.34, "backward": 0.04},
        "stochastic": _stochastic_check(matrix),
        "detailed_balance_defect": detailed_balance_defect(matrix),
        "second_eigenvalue_real": float(eigenvalue.real),
        "second_eigenvalue_imag": float(eigenvalue.imag),
        "dynamical_response_alignment": _alignment(slow, macro_direction),
        **metrics,
    }


def run_analysis():
    metastable_exact = run_metastable()
    metastable_broken = run_metastable(asymmetry=0.4)
    nuisance_rates = (0.16, 0.22, 0.28, 0.34, 0.40)
    plateau = [run_hierarchical(nuisance_flip=rate) for rate in nuisance_rates]
    certificate_keys = ("response_intra_distortion", "closure_defect", "memory_residual")
    plateau_variation = {
        key: max(item[key] for item in plateau) - min(item[key] for item in plateau)
        for key in certificate_keys
    }
    plateau_variation["coarse_matrix"] = float(
        max(np.max(np.abs(np.asarray(item["coarse_matrix"]) - np.asarray(plateau[0]["coarse_matrix"]))) for item in plateau)
    )
    circulating_grid = tuple(round(value, 2) for value in np.linspace(0.0, 0.8, 9))
    circulating = [run_circulating(value) for value in circulating_grid]
    admission_thresholds = {
        "response_intra_distortion": 0.01,
        "closure_defect": 0.005,
        "memory_residual": 0.005,
        "dynamical_response_alignment": 0.99,
    }
    admitted = [
        item["parameters"]["heterogeneity"]
        for item in circulating
        if item["response_intra_distortion"] <= admission_thresholds["response_intra_distortion"]
        and item["closure_defect"] <= admission_thresholds["closure_defect"]
        and item["memory_residual"] <= admission_thresholds["memory_residual"]
        and item["dynamical_response_alignment"] >= admission_thresholds["dynamical_response_alignment"]
    ]
    return {
        "metastable_exact": metastable_exact,
        "metastable_broken": metastable_broken,
        "hierarchical_plateau": plateau,
        "plateau_variation": plateau_variation,
        "circulating_nonreversible": circulating,
        "circulating_admission_thresholds": admission_thresholds,
        "circulating_admitted_heterogeneity": admitted,
        "circulating_platform_endpoint": max(admitted) if admitted else None,
        "registered_tolerance": 1e-12,
    }


if __name__ == "__main__":
    print(json.dumps(run_analysis(), ensure_ascii=False, indent=2))
