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

Usage examples for each of the four problem files are shown at the bottom
of this module (guarded by `if __name__ == "__main__"`).
─────────────────────────────────────────────────────────────────────────────
"""

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
        f"Optimal = ${opt_cost:,.0f}"
        if cost_unit == "$"
        else f"Optimal = {opt_cost:,.2f} {cost_unit}"
    )
    ax.axvline(opt_cost, color="#C6A477", linestyle="--",
               linewidth=2, label=opt_label)

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
    plt.show()

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
# Usage examples — one block per problem file
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Assignment Problem (assignment.py) ─────────────────────────────────
    import gurobipy as gp
    from gurobipy import GRB

    agents_a = [
        "Analytics & Insights Agent",
        "Knowledge Management Agent",
        "Marketing Campaign Orchestration Agent",
        "Customer / Support Automation Agent",
        "Data Quality & Monitoring Agent",
        "Workflow Orchestration Agent",
        "Governance & Compliance Agent",
    ]
    projects_a = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
    values_a = [
        [9, 6, 8, 5, 9, 7, 8, 6, 7],
        [6, 9, 5, 8, 6, 7, 5, 9, 6],
        [5, 6, 9, 4, 7, 9, 8, 5, 5],
        [4, 9, 6, 7, 5, 6, 7, 8, 9],
        [7, 5, 7, 6, 8, 5, 9, 5, 8],
        [6, 6, 7, 6, 7, 8, 8, 7, 9],
        [5, 6, 4, 9, 6, 5, 7, 6, 9],
    ]
    value_map_a = {
        (agents_a[i], projects_a[j]): values_a[i][j]
        for i in range(len(agents_a))
        for j in range(len(projects_a))
    }

    m_a = gp.Model("Assignment")
    X_a = m_a.addVars(agents_a, projects_a, lb=0, ub=1, vtype=GRB.BINARY, name="x")
    for j in projects_a:
        m_a.addConstr(gp.quicksum(X_a[(i, j)] for i in agents_a) == 1)
    for i in agents_a:
        m_a.addConstr(gp.quicksum(X_a[(i, j)] for j in projects_a) == 1)

    simulate_and_plot(
        model=m_a,
        variables=X_a,
        obj_coeffs=value_map_a,
        obj_sense=GRB.MAXIMIZE,
        n_simulations=500,
        problem_name="Assignment — Agent-to-Project Allocation",
        cost_unit="score",
    )

    # ── 2. Knapsack Problem (knapsack.py) ─────────────────────────────────────
    agents_k = [
        "Intent Classifier", "Inventory Checker", "Fraud Detector",
        "Order Processor", "Shipping Optimizer", "Returns Handler",
        "Customer Support Bot", "Review Analyzer", "Upsell Recommender",
        "Analytics Reporter",
    ]
    C_k = [800, 1200, 2500, 3000, 2200, 1800, 2000, 900, 1500, 1100]
    P_k = [50, 70, 90, 95, 80, 65, 75, 45, 60, 50]
    K_k = 4000
    N_k = range(len(agents_k))

    m_k = gp.Model("Knapsack")
    x_k = m_k.addVars(N_k, lb=0, ub=1, vtype=GRB.BINARY, name="x")
    m_k.addConstr(gp.quicksum(C_k[i] * x_k[i] for i in N_k) <= K_k)

    simulate_and_plot(
        model=m_k,
        variables=x_k,
        obj_coeffs={i: P_k[i] for i in N_k},
        obj_sense=GRB.MAXIMIZE,
        n_simulations=500,
        problem_name="Knapsack — Agent Budget Selection",
        cost_unit="tokens (M)",
    )

    # ── 3. Network Routing / Shortest Path (network_routing.py) ──────────────
    nodes_n = [
        "Coding Agent", "Writing Agent", "Marketing Dept",
        "IT Dept", "Operations Dept", "Relay Hub A", "Relay Hub B",
    ]
    arcs_data_n = [
        ("Coding Agent",  "Marketing Dept",  800),
        ("Coding Agent",  "IT Dept",         600),
        ("Coding Agent",  "Operations Dept", 300),
        ("Coding Agent",  "Relay Hub A",     300),
        ("Coding Agent",  "Relay Hub B",     200),
        ("Writing Agent", "Marketing Dept",  500),
        ("Writing Agent", "IT Dept",         200),
        ("Writing Agent", "Operations Dept", 200),
        ("Writing Agent", "Relay Hub A",     400),
        ("Writing Agent", "Relay Hub B",     200),
        ("Relay Hub A",   "Marketing Dept",  300),
        ("Relay Hub A",   "IT Dept",         800),
        ("Relay Hub A",   "Operations Dept", 500),
        ("Relay Hub A",   "Relay Hub B",     500),
        ("Relay Hub B",   "Marketing Dept",  800),
        ("Relay Hub B",   "IT Dept",         100),
        ("Relay Hub B",   "Operations Dept", 100),
        ("Relay Hub B",   "Relay Hub A",     500),
    ]
    arcs_n   = [(a, b) for a, b, _ in arcs_data_n]
    costs_n  = {(a, b): c for a, b, c in arcs_data_n}
    caps_n   = {arc: 2000 for arc in arcs_n}
    supplies_n = {"Coding Agent": 6500, "Writing Agent": 5500,
                  "Marketing Dept": 0, "IT Dept": 0, "Operations Dept": 0,
                  "Relay Hub A": 0, "Relay Hub B": 0}
    demands_n  = {"Coding Agent": 0, "Writing Agent": 0,
                  "Marketing Dept": 5000, "IT Dept": 4000, "Operations Dept": 3000,
                  "Relay Hub A": 0, "Relay Hub B": 0}

    m_n = gp.Model("Network Routing")
    X_n = m_n.addVars(arcs_n, lb=0, ub=caps_n, vtype=GRB.CONTINUOUS, name="x")
    for i in nodes_n:
        m_n.addConstr(
            supplies_n[i] + gp.quicksum(X_n[(j, i)] for j in nodes_n if (j, i) in arcs_n)
            == demands_n[i] + gp.quicksum(X_n[(i, j)] for j in nodes_n if (i, j) in arcs_n)
        )

    simulate_and_plot(
        model=m_n,
        variables=X_n,
        obj_coeffs=costs_n,
        obj_sense=GRB.MINIMIZE,
        n_simulations=500,
        problem_name="Network Routing — Shortest Path",
        cost_unit="$",
    )

    # ── 4. Set Covering Problem (set_covering.py) ─────────────────────────────
    agents_s = [
        "General Support Agent", "Access & Permissions Agent",
        "CRM / Relationship Management Agent", "Campaign Orchestration Agent",
        "Analytics & Insights Agent", "Billing & Transactions Agent",
        "Onboarding & Setup Agent", "Knowledge Management Agent",
        "Incident & Alert Handling Agent", "Procurement & Resources Agent",
        "Governance & Compliance Agent", "Cost & Performance Optimization Agent",
    ]
    skills_s = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    agent_skills_s = {
        "General Support Agent":                    ["F", "C", "H"],
        "Access & Permissions Agent":               ["F", "E", "G"],
        "CRM / Relationship Management Agent":      ["A", "B", "C"],
        "Campaign Orchestration Agent":             ["A", "B", "H"],
        "Analytics & Insights Agent":               ["C", "D", "I"],
        "Billing & Transactions Agent":             ["B", "D", "E"],
        "Onboarding & Setup Agent":                 ["E", "F", "H"],
        "Knowledge Management Agent":               ["C", "A"],
        "Incident & Alert Handling Agent":          ["A", "I", "G"],
        "Procurement & Resources Agent":            ["B", "E", "J"],
        "Governance & Compliance Agent":            ["G", "C", "I"],
        "Cost & Performance Optimization Agent":    ["D", "I", "J"],
    }
    agent_costs_s = {i: 20 for i in agents_s}

    m_s = gp.Model("Set Covering")
    X_s = m_s.addVars(agents_s, lb=0, ub=1, vtype=GRB.BINARY, name="x")
    for j in skills_s:
        m_s.addConstr(
            gp.quicksum(X_s[i] for i in agents_s if j in agent_skills_s[i]) >= 1
        )

    simulate_and_plot(
        model=m_s,
        variables=X_s,
        obj_coeffs=agent_costs_s,
        obj_sense=GRB.MINIMIZE,
        n_simulations=500,
        problem_name="Set Covering — Skill Coverage",
        cost_unit="k$",
    )
