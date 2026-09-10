"""Finite approximate CPIR error certificate.

This module instantiates a standard total-variation contraction/perturbation
bound.  It is a regression artifact, not empirical evidence for DCS.
"""


def tv(left, right):
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def row_times_kernel(row, kernel):
    return tuple(sum(row[i] * kernel[i][j] for i in range(len(row))) for j in range(len(kernel[0])))


def evolve(row, kernel, steps):
    for _ in range(steps):
        row = row_times_kernel(row, kernel)
    return row


def pushforward(row, partition):
    return tuple(sum(row[state] for state in block) for block in partition)


def pushforward_kernel_row(kernel, state, partition):
    return pushforward(kernel[state], partition)


def dobrushin(kernel):
    return max(tv(left, right) for left in kernel for right in kernel)


def one_step_defect(micro_kernel, macro_kernel, partition):
    block_of = {state: block_id for block_id, block in enumerate(partition) for state in block}
    return max(
        tv(pushforward_kernel_row(micro_kernel, state, partition), macro_kernel[block_of[state]])
        for state in range(len(micro_kernel))
    )


def geometric_bound(initial_error, defect, alpha, horizon):
    if alpha == 1.0:
        return initial_error + horizon * defect
    return alpha**horizon * initial_error + defect * (1.0 - alpha**horizon) / (1.0 - alpha)


def expectation(row, observable):
    return sum(probability * value for probability, value in zip(row, observable))


def contractive_example(max_horizon=12):
    partition = ((0, 1), (2, 3))
    macro_kernel = ((0.8, 0.2), (0.3, 0.7))
    micro_kernel = (
        (0.41, 0.41, 0.09, 0.09),
        (0.39, 0.39, 0.11, 0.11),
        (0.14, 0.14, 0.36, 0.36),
        (0.16, 0.16, 0.34, 0.34),
    )
    micro_initial = (1.0, 0.0, 0.0, 0.0)
    macro_initial = pushforward(micro_initial, partition)
    alpha = dobrushin(macro_kernel)
    defect = one_step_defect(micro_kernel, macro_kernel, partition)
    horizons = []
    for horizon in range(max_horizon + 1):
        micro_macro = pushforward(evolve(micro_initial, micro_kernel, horizon), partition)
        macro = evolve(macro_initial, macro_kernel, horizon)
        error = tv(micro_macro, macro)
        bound = geometric_bound(0.0, defect, alpha, horizon)
        horizons.append({"horizon": horizon, "error": round(error, 12), "bound": round(bound, 12)})
    target = (1.0, 0.0)
    last = horizons[-1]
    micro_macro = pushforward(evolve(micro_initial, micro_kernel, max_horizon), partition)
    macro = evolve(macro_initial, macro_kernel, max_horizon)
    response_error = abs(expectation(micro_macro, target) - expectation(macro, target))
    return {
        "alpha": round(alpha, 12),
        "one_step_defect": round(defect, 12),
        "horizons": horizons,
        "bounded_target_error": round(response_error, 12),
        "target_bound": last["bound"],
    }


def noncontractive_example(max_horizon=8):
    macro_kernel = ((1.0, 0.0), (0.0, 1.0))
    defect = 0.02
    return {
        "alpha": dobrushin(macro_kernel),
        "defect": defect,
        "horizon": max_horizon,
        "linear_bound": geometric_bound(0.0, defect, 1.0, max_horizon),
    }


def hidden_target_counterexample():
    return {
        "macro_distribution_error": 0.0,
        "hidden_target_error": 1.0,
        "lesson": "distribution closure does not certify a non-macro-measurable target",
    }


def run_all():
    return {
        "contractive": contractive_example(),
        "noncontractive": noncontractive_example(),
        "hidden_target": hidden_target_counterexample(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_all(), ensure_ascii=False, indent=2))
