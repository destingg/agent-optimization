import gurobipy as gp
from gurobipy import GRB

## data
# Nodes: supply (+), demand (-), relay (0)
nodes = [
    "Coding Agent",      # supply 6500
    "Writing Agent",     # supply 5500
    "Marketing Dept",    # demand 5000
    "IT Dept",           # demand 4000
    "Operations Dept",   # demand 3000
    "Relay Hub A",       # relay (no net supply/demand)
    "Relay Hub B",       # relay (no net supply/demand)
]

# Net balance per node: supply > 0, demand < 0, relay = 0
supplies = {
    "Coding Agent":      6500,
    "Writing Agent":     5500,
    "Marketing Dept":    0,
    "IT Dept":           0,
    "Operations Dept":   0,
    "Relay Hub A":       0,
    "Relay Hub B":       0,
}
demands = {
    "Coding Agent":      0,
    "Writing Agent":     0,
    "Marketing Dept":    5000,
    "IT Dept":           4000,
    "Operations Dept":   3000,
    "Relay Hub A":       0,
    "Relay Hub B":       0,
}

# Arcs: (from, to) with routing cost per request
# Cluster A = Coding Agent, Cluster B = Writing Agent
arcs_data = [
    # from Coding Agent (Cluster A)
    ("Coding Agent",  "Marketing Dept",  0.80),
    ("Coding Agent",  "IT Dept",         0.60),
    ("Coding Agent",  "Operations Dept", 0.30),
    ("Coding Agent",  "Relay Hub A",     0.30),
    ("Coding Agent",  "Relay Hub B",     0.20),
    # from Writing Agent (Cluster B)
    ("Writing Agent", "Marketing Dept",  0.50),
    ("Writing Agent", "IT Dept",         0.20),
    ("Writing Agent", "Operations Dept", 0.20),
    ("Writing Agent", "Relay Hub A",     0.40),
    ("Writing Agent", "Relay Hub B",     0.20),
    # from Relay Hub A
    ("Relay Hub A",   "Marketing Dept",  0.30),
    ("Relay Hub A",   "IT Dept",         0.80),
    ("Relay Hub A",   "Operations Dept", 0.50),
    ("Relay Hub A",   "Relay Hub B",     0.50),
    # from Relay Hub B
    ("Relay Hub B",   "Marketing Dept",  0.80),
    ("Relay Hub B",   "IT Dept",         0.10),
    ("Relay Hub B",   "Operations Dept", 0.10),
    ("Relay Hub B",   "Relay Hub A",     0.50),
]

arcs = [(a, b) for a, b, _ in arcs_data]
costs = {(a, b): c for a, b, c in arcs_data}
capacities = {
    ("Coding Agent",  "Marketing Dept"):  2000,
    ("Coding Agent",  "IT Dept"):         2000,
    ("Coding Agent",  "Operations Dept"): 1200,
    ("Coding Agent",  "Relay Hub A"):     1500,
    ("Coding Agent",  "Relay Hub B"):     1200,
    ("Writing Agent", "Marketing Dept"):  2000,
    ("Writing Agent", "IT Dept"):         1000,
    ("Writing Agent", "Operations Dept"): 1000,
    ("Writing Agent", "Relay Hub A"):     2000,
    ("Writing Agent", "Relay Hub B"):     1200,
    ("Relay Hub A",   "Marketing Dept"):  1500,
    ("Relay Hub A",   "IT Dept"):         2000,
    ("Relay Hub A",   "Operations Dept"): 2000,
    ("Relay Hub A",   "Relay Hub B"):     2000,
    ("Relay Hub B",   "Marketing Dept"):  2000,
    ("Relay Hub B",   "IT Dept"):         1000,
    ("Relay Hub B",   "Operations Dept"): 1000,
    ("Relay Hub B",   "Relay Hub A"):     2000,
}

def build_model():
    """Return (model, variables, obj_coeffs, obj_sense, problem_name, cost_unit)."""
    model = gp.Model("Network Routing")
    model.Params.OutputFlag = 0

    ## variables: flow on each arc (continuous, bounded by capacity)
    X = model.addVars(arcs, lb=0, ub=capacities, vtype=GRB.CONTINUOUS, name="x")

    ## constraints: flow conservation at each node
    # supply[i] + inflow[i] == demand[i] + outflow[i]
    for i in nodes:
        model.addConstr(
            supplies[i] + gp.quicksum(X[(j, i)] for j in nodes if (j, i) in arcs)
            == demands[i] + gp.quicksum(X[(i, j)] for j in nodes if (i, j) in arcs),
            name=f"{i}_balance_constr",
        )

    return model, X, costs, GRB.MINIMIZE, "Network Routing — Agent Routing", "$"


if __name__ == "__main__":
    model, X, obj_coeffs, obj_sense, _, _ = build_model()

    ## objective: minimize total routing cost
    model.setObjective(
        gp.quicksum(obj_coeffs[(i, j)] * X[(i, j)] for i, j in arcs),
        obj_sense,
    )
    model.optimize()

    print("================ Network Routing Problem: Agent Routing ================")
    if model.Status == GRB.OPTIMAL:
        print(f"Minimum total routing cost: ${model.ObjVal:,.2f}\n")
        print(f"{'Arc':<45} {'Flow':>8} {'Cap':>6} {'Cost/req':>10} {'Total Cost':>12}")
        print("-" * 85)
        for i, j in arcs:
            flow = X[(i, j)].X
            if flow > 0.5:
                total_cost = obj_coeffs[(i, j)] * flow
                print(
                    f"{i} -> {j:<30} {flow:>8.0f} {capacities[(i,j)]:>6} {obj_coeffs[(i,j)]:>10.2f} {total_cost:>12,.2f}"
                )
        print("\nDemand fulfillment:")
        demand_nodes = ["Marketing Dept", "IT Dept", "Operations Dept"]
        for d in demand_nodes:
            inflow = sum(X[(j, d)].X for j in nodes if (j, d) in arcs)
            print(f"  {d}: received {inflow:.0f} / needed {demands[d]}")
    else:
        print("No optimal solution found.")
