"""Finite exact CPIR positive model and X1--X8 regression counterexamples.

Only finite sets, deterministic maps and row-stochastic kernels are used.
The module tests logical non-implications; it does not simulate a physical
system or claim empirical confirmation.
"""

from collections import defaultdict


def quotient_edges(edges, partition):
    block_of = {x: i for i, block in enumerate(partition) for x in block}
    return {(block_of[a], block_of[b]) for a, b in edges if block_of[a] != block_of[b]}


def has_directed_cycle(nodes, edges):
    adjacency = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for source, target in edges:
        adjacency[source].append(target)
        indegree[target] += 1
    queue = [node for node in nodes if indegree[node] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited != len(nodes)


def strongly_connected_components(nodes, edges):
    adjacency = defaultdict(list)
    reverse = defaultdict(list)
    for source, target in edges:
        adjacency[source].append(target)
        reverse[target].append(source)
    seen, order = set(), []

    def visit(node):
        seen.add(node)
        for target in adjacency[node]:
            if target not in seen:
                visit(target)
        order.append(node)

    for node in nodes:
        if node not in seen:
            visit(node)
    seen.clear()
    components = []

    def collect(node, component):
        seen.add(node)
        component.append(node)
        for target in reverse[node]:
            if target not in seen:
                collect(target, component)

    for node in reversed(order):
        if node not in seen:
            component = []
            collect(node, component)
            components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def push_row(row, partition):
    return tuple(sum(row[state] for state in block) for block in partition)


def lumpability_defect(kernel, partition):
    defect = 0.0
    for block in partition:
        signatures = [push_row(kernel[state], partition) for state in block]
        for left in signatures:
            for right in signatures:
                defect = max(defect, max(abs(a - b) for a, b in zip(left, right)))
    return defect


def block_measurability_defect(values, partition):
    return max(max(values[state] for state in block) - min(values[state] for state in block) for block in partition)


def x1_order_quotient_cycle():
    edges = {("a0", "b0"), ("b1", "a1")}
    partition = (("a0", "a1"), ("b0", "b1"))
    macro_edges = quotient_edges(edges, partition)
    return {"macro_edges": sorted(macro_edges), "macro_cycle": has_directed_cycle(range(2), macro_edges)}


def x2_scc_repair_changes_partition():
    x1 = x1_order_quotient_cycle()
    components = strongly_connected_components(range(2), set(map(tuple, x1["macro_edges"])))
    return {"original_block_count": 2, "repaired_block_count": len(components), "components": components, "requires_recertification": len(components) != 2}


def x3_non_lumpable_partition():
    kernel = ((0.9, 0.0, 0.1), (0.0, 0.1, 0.9), (0.0, 0.0, 1.0))
    partition = ((0, 1), (2,))
    return {"lumpability_defect": lumpability_defect(kernel, partition)}


def x4_implementation_noninvariance():
    partition = ((0, 1), (2, 3))
    implementation_a = {0: 2, 1: 2, 2: 2, 3: 3}
    implementation_b = {0: 0, 1: 0, 2: 2, 3: 3}
    block_of = {state: i for i, block in enumerate(partition) for state in block}
    effects = {"a": tuple(block_of[implementation_a[state]] for state in partition[0]), "b": tuple(block_of[implementation_b[state]] for state in partition[0])}
    return {"macro_effects": effects, "implementation_invariant": effects["a"] == effects["b"]}


def x5_target_not_measurable():
    kernel = ((1.0, 0.0), (0.0, 1.0))
    partition = ((0, 1),)
    target = (0.0, 1.0)
    return {"lumpability_defect": lumpability_defect(kernel, partition), "target_measurability_defect": block_measurability_defect(target, partition)}


def x6_one_step_fit_hides_memory():
    kernel = ((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    partition = ((0, 1), (2, 3))
    one_step = [push_row(kernel[state], partition) for state in partition[0]]
    two_step = []
    for state in partition[0]:
        row = tuple(sum(kernel[state][middle] * kernel[middle][target] for middle in range(4)) for target in range(4))
        two_step.append(push_row(row, partition))
    return {"one_step_equal": one_step[0] == one_step[1], "two_step_equal": two_step[0] == two_step[1], "two_step_macro_rows": two_step}


def x7_persistence_not_measurable():
    partition = ((0, 1), (2,))
    persistence_identity = (1.0, 0.0, 0.0)
    return {"persistence_measurability_defect": block_measurability_defect(persistence_identity, partition)}


def x8_compression_without_capability_gain():
    micro_states = (0, 1, 2, 3)
    partition = ((0, 1), (2, 3))
    micro_reachable_outputs = {0, 1}
    macro_reachable_outputs = {0, 1}
    return {"micro_state_count": len(micro_states), "macro_state_count": len(partition), "compression_ratio": len(micro_states) / len(partition), "micro_reachable_outputs": sorted(micro_reachable_outputs), "macro_reachable_outputs": sorted(macro_reachable_outputs), "new_reachable_capability": bool(macro_reachable_outputs - micro_reachable_outputs)}


def positive_exact_quotient():
    partition = ((0, 1), (2, 3))
    kernel = ((0.6, 0.2, 0.1, 0.1), (0.2, 0.6, 0.1, 0.1), (0.15, 0.15, 0.5, 0.2), (0.15, 0.15, 0.2, 0.5))
    target = (1.0, 1.0, 0.0, 0.0)
    persistence = (0.9, 0.9, 0.4, 0.4)
    swap_internal = {0: 1, 1: 0, 2: 3, 3: 2}
    block_of = {state: i for i, block in enumerate(partition) for state in block}
    intervention_effects = {block_of[swap_internal[state]] for state in partition[0]}
    return {"lumpability_defect": lumpability_defect(kernel, partition), "target_measurability_defect": block_measurability_defect(target, partition), "persistence_measurability_defect": block_measurability_defect(persistence, partition), "implementation_invariant": len(intervention_effects) == 1, "macro_kernel": [list(push_row(kernel[block[0]], partition)) for block in partition]}


def run_all():
    return {"positive": positive_exact_quotient(), "CPIR-X1": x1_order_quotient_cycle(), "CPIR-X2": x2_scc_repair_changes_partition(), "CPIR-X3": x3_non_lumpable_partition(), "CPIR-X4": x4_implementation_noninvariance(), "CPIR-X5": x5_target_not_measurable(), "CPIR-X6": x6_one_step_fit_hides_memory(), "CPIR-X7": x7_persistence_not_measurable(), "CPIR-X8": x8_compression_without_capability_gain()}

