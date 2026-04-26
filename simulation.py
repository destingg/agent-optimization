"""
simulation.py
─────────────────────────────────────────────────────────────────────────────
Generic simulate-and-plot utility for Gurobi optimisation models.


Works with any problem whose structure follows:
    • A built Gurobi model with flow/assignment/covering/knapsack constraints
    • A flat dict of {key: gp.Var} decision variables
    • A matching dict of {key: float} original objective coefficients
    • A known objective sense (GRB.MINIMIZE or GRB.MAXIMIZE)


The function randomises the objective n_simulations times, solves each
perturbation, evaluates the *original* objective on every feasible solution
found, then overlays the true optimum on the resulting histogram.


Each problem file (assignment, knapsack, network_routing, set_covering)
exposes a build_model() function that returns:
    (model, variables, obj_coeffs, obj_sense, problem_name, cost_unit)


Run all four problems:
    python simulation.py


Run a single problem file directly (no simulation):
    python assignment.py
─────────────────────────────────────────────────────────────────────────────
"""


import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB


# ─────────────────────────────────────────────────────────────────────────────
# Core function
# ─────────────────────────────────────────────────────────────────────────────


def simulate_and_plot(
    model,
    variables,
    obj_coeffs,
    obj_sense=GRB.MINIMIZE,
    n_simulations=500,
    problem_name="Optimisation Problem",
    cost_unit="$",
    figsize=(10, 5),
    save_dir="plots",
):
    """
    Simulate random feasible solutions and compare against the true optimum.


    Parameters
    ----------
    model        : gp.Model
        A fully-constrained Gurobi model (objective may be unset — it will be
        overwritten). All variables and constraints must already be added.
    variables    : dict  {key: gp.Var}
        Decision variables keyed however is natural for the problem
        (e.g. (agent, project) tuples, integer indices, arc tuples, strings).
        A Gurobi tupledict returned by model.addVars() works directly.
    obj_coeffs   : dict  {key: float}
        Original objective coefficients, using the same keys as `variables`.
    obj_sense    : GRB.MINIMIZE | GRB.MAXIMIZE
        Direction of the true optimisation.
    n_simulations : int
        Number of random perturbations to run.
    problem_name  : str
        Descriptive label shown in the plot title.
    cost_unit     : str
        Unit appended to axis labels and printed summary
        (e.g. "$", "tokens", "score", "k$").
    figsize       : tuple
        Matplotlib figure dimensions.
    save_dir      : str
        Directory where the plot image will be saved.


    Returns
    -------
    sim_costs : list[float]
        Original-objective values recorded from each feasible simulation.
    opt_cost  : float
        True optimal objective value.
    """
    keys = list(variables.keys())

    model.setParam("OutputFlag", 0)

    # ── simulate random feasible solutions ────────────────────────────────────
    sim_costs = []
    for _ in range(n_simulations):
        rand_coeffs = {k: np.random.uniform(-1, 1) for k in keys}

        # Always minimise the random surrogate so the solver explores both
        # ends of the feasible region regardless of the true sense.
        model.setObjective(
            gp.quicksum(rand_coeffs[k] * variables[k] for k in keys),
            GRB.MINIMIZE,
        )
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            orig_val = sum(obj_coeffs[k] * variables[k].X for k in keys)
            sim_costs.append(orig_val)

    # ── solve for the true optimum ────────────────────────────────────────────
    model.setObjective(
        gp.quicksum(obj_coeffs[k] * variables[k] for k in keys),
        obj_sense,
    )
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        print("Warning: true optimum could not be found.")
        return sim_costs, None

    opt_cost = model.ObjVal

    if not sim_costs:
        print("No feasible solutions were found during simulation.")
        return sim_costs, opt_cost

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(sim_costs, bins=40, kde=True, ax=ax,
                 color="#ABD1DC", label="Simulated")

    opt_label = (
        # f"Optimal = ${opt_cost:,.0f}"
        # if cost_unit == "$"
        # else 
        f"Optimal = {opt_cost:,.2f} {cost_unit}"
    )
    ax.axvline(opt_cost, color="#C6A477", linestyle="--",
               linewidth=2, label=opt_label)

    sim_mean = np.mean(sim_costs)
    ax.axvline(sim_mean, color="#7FAEAF", linestyle=":",
               linewidth=2, label=f"Sim Mean = {sim_mean:,.2f} {cost_unit}")

    ax.set_title(
        f"Random Feasible vs Optimal — {problem_name}",
        fontsize=14, fontweight="bold",
    )
    x_label = f"Objective Value ({cost_unit})" if cost_unit else "Objective Value"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.legend()
    sns.despine()
    plt.tight_layout()

    # ── save figure ───────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    safe_name = problem_name.lower().replace(" ", "_").replace("/", "-")
    save_path = os.path.join(save_dir, f"{safe_name}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {save_path}")

    plt.show()
    plt.close(fig)

    # ── printed summary ───────────────────────────────────────────────────────
    direction = "Maximum" if obj_sense == GRB.MAXIMIZE else "Minimum"
    fmt = lambda v: f"${v:,.2f}" if cost_unit == "$" else f"{v:,.2f} {cost_unit}"
    print(f"\n{'─'*50}")
    print(f"  {problem_name}")
    print(f"{'─'*50}")
    print(f"  {direction} (Optimal) : {fmt(opt_cost)}")
    print(f"  Sim mean           : {fmt(np.mean(sim_costs))}")
    print(f"  Sim min            : {fmt(min(sim_costs))}")
    print(f"  Sim max            : {fmt(max(sim_costs))}")
    print(f"  Feasible solutions : {len(sim_costs)}/{n_simulations}")
    print(f"{'─'*50}\n")

    return sim_costs, opt_cost


# ─────────────────────────────────────────────────────────────────────────────
# Run all four problems — imports model + scenario directly from each file
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    from assignment      import build_model as assignment_model
    from knapsack        import build_model as knapsack_model
    from network_routing import build_model as network_routing_model
    from set_covering    import build_model as set_covering_model

    problems = [
        assignment_model,
        knapsack_model,
        network_routing_model,
        set_covering_model,
    ]

    for build_fn in problems:
        model, variables, obj_coeffs, obj_sense, problem_name, cost_unit = build_fn()
        simulate_and_plot(
            model=model,
            variables=variables,
            obj_coeffs=obj_coeffs,
            obj_sense=obj_sense,
            n_simulations=500,
            problem_name=problem_name,
            cost_unit=cost_unit,
            save_dir="plots",
        )