"""Finite controlled-Markov prototype for DCS cross-scale certification.

The implementation deliberately reuses standard finite-state constructions:
controlled transition kernels, Kemeny--Snell lumpability, multi-step kernel
comparison, deterministic intervention commutation, Jensen--Shannon response
distance, and Pareto dominance.  It does not assume that response compression,
dynamical closure, persistence, and intervention consistency coincide.
"""

from itertools import combinations
from math import log, sqrt


STATES = ("A0", "A1", "B0", "B1", "D")
LIVE_STATES = STATES[:-1]
OUTCOME_OF = {"A0": "A", "A1": "A", "B0": "B", "B1": "B", "D": "D"}
OUTCOMES = ("A", "B", "D")
ACTIONS = ("baseline", "protect_a", "protect_b")
HORIZONS = (1, 2, 3)


def _empty_matrix():
    return [[0.0] * len(STATES) for _ in STATES]


def controlled_transition_matrices(heterogeneity=0.0):
    """Return three registered row-stochastic controlled kernels.

    At heterogeneity zero, states with the same A/B label have equal rates to
    every A/B/D block.  A signed internal-index perturbation breaks that exact
    specialization without changing the state or action type.
    """
    if not 0.0 <= abs(heterogeneity) <= 0.5:
        raise ValueError("heterogeneity must have absolute value at most 0.5")

    matrices = {}
    for action in ACTIONS:
        matrix = _empty_matrix()
        for state in LIVE_STATES:
            family = state[0]
            index = int(state[1])
            internal_target = family + str(1 - index)
            other_family = "B" if family == "A" else "A"
            cross_target = other_family + str(index)

            internal = 0.24
            cross = 0.18 * (1.0 + heterogeneity * (1.0 if index == 0 else -1.0))
            death = 0.06 if family == "A" else 0.10
            if action == "protect_a" and family == "A":
                death *= 0.35
                cross *= 0.80
            if action == "protect_b" and family == "B":
                death *= 0.35
                cross *= 0.80

            row = STATES.index(state)
            matrix[row][STATES.index(internal_target)] = internal
            matrix[row][STATES.index(cross_target)] = cross
            matrix[row][STATES.index("D")] = death
            matrix[row][row] = 1.0 - internal - cross - death
        matrix[STATES.index("D")][STATES.index("D")] = 1.0
        matrices[action] = matrix
    return matrices


def matrix_multiply(left, right):
    return [
        [sum(left[i][k] * right[k][j] for k in range(len(right))) for j in range(len(right[0]))]
        for i in range(len(left))
    ]


def matrix_power(matrix, exponent):
    result = [[1.0 if i == j else 0.0 for j in range(len(matrix))] for i in range(len(matrix))]
    base = matrix
    power = exponent
    while power:
        if power % 2:
            result = matrix_multiply(result, base)
        base = matrix_multiply(base, base)
        power //= 2
    return result


def outcome_distribution(row):
    return tuple(sum(row[j] for j, state in enumerate(STATES) if OUTCOME_OF[state] == outcome) for outcome in OUTCOMES)


def response_vectors(matrices=None):
    """Flatten all registered action-horizon future outcome kernels."""
    matrices = matrices or controlled_transition_matrices()
    powers = {(action, horizon): matrix_power(matrices[action], horizon) for action in ACTIONS for horizon in HORIZONS}
    responses = {}
    for state in STATES:
        values = []
        row = STATES.index(state)
        for action in ACTIONS:
            for horizon in HORIZONS:
                values.extend(outcome_distribution(powers[(action, horizon)][row]))
        responses[state] = tuple(values)
    return responses


def jensen_shannon(left, right):
    midpoint = tuple((x + y) / 2.0 for x, y in zip(left, right))

    def kl(first, second):
        return sum(x * log(x / y) for x, y in zip(first, second) if x > 0.0)

    return 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)


def response_distance(left, right):
    """Root-mean Jensen--Shannon distance over action-horizon responses."""
    total = 0.0
    count = 0
    width = len(OUTCOMES)
    for offset in range(0, len(left), width):
        total += jensen_shannon(left[offset : offset + width], right[offset : offset + width])
        count += 1
    return sqrt(total / count)


def set_partitions(items=LIVE_STATES):
    """Enumerate canonical set partitions; four live states give Bell(4)=15."""
    items = tuple(items)
    if not items:
        return [tuple()]
    partitions = [((items[0],),)]
    for item in items[1:]:
        expanded = []
        for partition in partitions:
            for index in range(len(partition)):
                blocks = [tuple(block) for block in partition]
                blocks[index] = blocks[index] + (item,)
                expanded.append(tuple(blocks))
            expanded.append(partition + ((item,),))
        partitions = expanded
    return partitions


def with_failure_block(live_partition):
    return tuple(live_partition) + (("D",),)


def block_map(partition):
    return {state: index for index, block in enumerate(partition) for state in block}


def _push_row_to_blocks(row, partition):
    return tuple(sum(row[STATES.index(state)] for state in block) for block in partition)


def total_variation(left, right):
    return 0.5 * sum(abs(x - y) for x, y in zip(left, right))


def lumpability_defect(matrices, partition):
    defect = 0.0
    for matrix in matrices.values():
        for block in partition:
            for left, right in combinations(block, 2):
                left_push = _push_row_to_blocks(matrix[STATES.index(left)], partition)
                right_push = _push_row_to_blocks(matrix[STATES.index(right)], partition)
                defect = max(defect, max(abs(x - y) for x, y in zip(left_push, right_push)))
    return defect


def coarse_matrix(matrix, partition):
    coarse = [[0.0] * len(partition) for _ in partition]
    for i, block in enumerate(partition):
        rows = [_push_row_to_blocks(matrix[STATES.index(state)], partition) for state in block]
        for j in range(len(partition)):
            coarse[i][j] = sum(row[j] for row in rows) / len(rows)
    return coarse


def memory_residual(matrices, partition):
    """Two-step macro error relative to the squared uniform coarse kernel."""
    mapping = block_map(partition)
    residual = 0.0
    for matrix in matrices.values():
        micro_two = matrix_power(matrix, 2)
        coarse = coarse_matrix(matrix, partition)
        coarse_two = matrix_power(coarse, 2)
        for state in STATES:
            actual = _push_row_to_blocks(micro_two[STATES.index(state)], partition)
            predicted = coarse_two[mapping[state]]
            residual = max(residual, total_variation(actual, predicted))
    return residual


def registered_interventions():
    identity = {state: state for state in STATES}
    flip = {"A0": "A1", "A1": "A0", "B0": "B1", "B1": "B0", "D": "D"}
    swap = {"A0": "B0", "A1": "B1", "B0": "A0", "B1": "A1", "D": "D"}
    reset_zero = {"A0": "A0", "A1": "A0", "B0": "B0", "B1": "B0", "D": "D"}
    return {"identity": identity, "flip_internal": flip, "swap_family": swap, "reset_zero": reset_zero}


def intervention_exchange_error(partition):
    """Worst deterministic micro-intervention disagreement within a macro block."""
    mapping = block_map(partition)
    error = 0.0
    for intervention in registered_interventions().values():
        for block in partition:
            images = [mapping[intervention[state]] for state in block]
            counts = [images.count(index) / len(images) for index in range(len(partition))]
            for image in images:
                point = [1.0 if index == image else 0.0 for index in range(len(partition))]
                error = max(error, total_variation(point, counts))
    return error


def survival_probabilities(matrices, horizon=max(HORIZONS)):
    powered = matrix_power(matrices["baseline"], horizon)
    death = STATES.index("D")
    return {state: 1.0 - powered[STATES.index(state)][death] for state in LIVE_STATES}


def persistence_distortion(partition, survival):
    distortion = 0.0
    for block in partition:
        values = [survival[state] for state in block if state != "D"]
        if values:
            distortion = max(distortion, max(values) - min(values))
    return distortion


def _centroid(states, responses):
    return tuple(sum(responses[state][i] for state in states) / len(states) for i in range(len(next(iter(responses.values())))))


def response_separation(partition, responses):
    live_blocks = [block for block in partition if block != ("D",)]
    intra = 0.0
    for block in live_blocks:
        for left, right in combinations(block, 2):
            intra = max(intra, response_distance(responses[left], responses[right]))
    centroids = [_centroid(block, responses) for block in live_blocks]
    if len(centroids) < 2:
        inter = 0.0
    else:
        inter = min(response_distance(left, right) for left, right in combinations(centroids, 2))
    return intra, inter, centroids


def _jacobi_eigenvalues(matrix, tolerance=1e-13, max_iterations=200):
    values = [row[:] for row in matrix]
    n = len(values)
    if n == 0:
        return []
    for _ in range(max_iterations):
        p, q, largest = 0, 0, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(values[i][j]) > largest:
                    p, q, largest = i, j, abs(values[i][j])
        if largest < tolerance:
            break
        tau = (values[q][q] - values[p][p]) / (2.0 * values[p][q])
        tangent = (1.0 if tau >= 0.0 else -1.0) / (abs(tau) + sqrt(1.0 + tau * tau))
        cosine = 1.0 / sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine
        app, aqq, apq = values[p][p], values[q][q], values[p][q]
        values[p][p] = app - tangent * apq
        values[q][q] = aqq + tangent * apq
        values[p][q] = values[q][p] = 0.0
        for k in range(n):
            if k in (p, q):
                continue
            akp, akq = values[k][p], values[k][q]
            values[k][p] = values[p][k] = cosine * akp - sine * akq
            values[k][q] = values[q][k] = sine * akp + cosine * akq
    return sorted((max(0.0, values[i][i]) for i in range(n)), reverse=True)


def response_spectrum(centroids):
    if len(centroids) <= 1:
        return [], 0, 0.0
    mean = [sum(vector[i] for vector in centroids) / len(centroids) for i in range(len(centroids[0]))]
    centered = [[value - mean[i] for i, value in enumerate(vector)] for vector in centroids]
    gram = [[sum(x * y for x, y in zip(left, right)) / len(centroids) for right in centered] for left in centered]
    spectrum = _jacobi_eigenvalues(gram)
    largest = spectrum[0] if spectrum else 0.0
    numerical_rank = sum(value > largest * 1e-10 for value in spectrum) if largest else 0
    total = sum(spectrum)
    square_total = sum(value * value for value in spectrum)
    effective_rank = total * total / square_total if square_total else 0.0
    return spectrum, numerical_rank, effective_rank


def partition_label(partition):
    return "|".join("".join(block) for block in partition)


def partition_certificate(live_partition, matrices=None):
    matrices = matrices or controlled_transition_matrices()
    partition = with_failure_block(live_partition)
    responses = response_vectors(matrices)
    intra, inter, centroids = response_separation(partition, responses)
    spectrum, rank, effective_rank = response_spectrum(centroids)
    survival = survival_probabilities(matrices)
    return {
        "partition": [list(block) for block in partition],
        "label": partition_label(partition),
        "live_block_count": len(live_partition),
        "interface_cost": len(live_partition) / len(LIVE_STATES),
        "causal_response_spectrum": spectrum,
        "causal_response_rank": rank,
        "causal_response_effective_rank": effective_rank,
        "response_intra_distortion": intra,
        "response_inter_separation": inter,
        "closure_defect": lumpability_defect(matrices, partition),
        "memory_residual": memory_residual(matrices, partition),
        "intervention_exchange_error": intervention_exchange_error(partition),
        "persistence_score": sum(survival.values()) / len(survival),
        "persistence_distortion": persistence_distortion(partition, survival),
    }


def _objective_vector(certificate):
    return (
        certificate["interface_cost"],
        certificate["response_intra_distortion"],
        -certificate["response_inter_separation"],
        certificate["closure_defect"],
        certificate["memory_residual"],
        certificate["intervention_exchange_error"],
        certificate["persistence_distortion"],
    )


def dominates(left, right, tolerance=1e-12):
    left_values = _objective_vector(left)
    right_values = _objective_vector(right)
    weak = all(x <= y + tolerance for x, y in zip(left_values, right_values))
    strict = any(x < y - tolerance for x, y in zip(left_values, right_values))
    return weak and strict


def pareto_frontier(certificates):
    return [candidate for candidate in certificates if not any(dominates(other, candidate) for other in certificates if other is not candidate)]


def exact_interface_frontier(certificates, tolerance=1e-10, min_separation=1e-6):
    """Apply scientific admission gates before Pareto comparison.

    This excludes both the one-block collapse (no between-block contrast) and
    the identity map (no compression) even when either is Pareto-optimal under
    a raw cost-versus-error tradeoff.
    """
    qualified = [
        item
        for item in certificates
        if item["interface_cost"] < 1.0 - tolerance
        and item["response_inter_separation"] > min_separation
        and item["response_intra_distortion"] <= tolerance
        and item["closure_defect"] <= tolerance
        and item["memory_residual"] <= tolerance
        and item["intervention_exchange_error"] <= tolerance
        and item["persistence_distortion"] <= tolerance
    ]
    return pareto_frontier(qualified)


def registered_partition():
    return (("A0", "A1"), ("B0", "B1"))


def run_analysis(heterogeneity=0.0):
    matrices = controlled_transition_matrices(heterogeneity)
    certificates = [partition_certificate(partition, matrices) for partition in set_partitions()]
    frontier = pareto_frontier(certificates)
    exact_frontier = exact_interface_frontier(certificates)
    target_label = partition_label(with_failure_block(registered_partition()))
    registered = next(item for item in certificates if item["label"] == target_label)
    return {
        "model_type": "F0 finite controlled Markov chain",
        "heterogeneity": heterogeneity,
        "partition_count": len(certificates),
        "registered_partition": registered,
        "pareto_labels": sorted(item["label"] for item in frontier),
        "pareto_count": len(frontier),
        "exact_interface_labels": sorted(item["label"] for item in exact_frontier),
        "exact_interface_count": len(exact_frontier),
        "certificates": certificates,
    }
