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
    ("Coding Agent",  "Marketing Dept",  800),
    ("Coding Agent",  "IT Dept",         600),
    ("Coding Agent",  "Operations Dept", 300),
    ("Coding Agent",  "Relay Hub A",     300),
    ("Coding Agent",  "Relay Hub B",     200),
    # from Writing Agent (Cluster B)
    ("Writing Agent", "Marketing Dept",  500),
    ("Writing Agent", "IT Dept",         200),
    ("Writing Agent", "Operations Dept", 200),
    ("Writing Agent", "Relay Hub A",     400),
    ("Writing Agent", "Relay Hub B",     200),
    # from Relay Hub A
    ("Relay Hub A",   "Marketing Dept",  300),
    ("Relay Hub A",   "IT Dept",         800),
    ("Relay Hub A",   "Operations Dept", 500),
    ("Relay Hub A",   "Relay Hub B",     500),
    # from Relay Hub B
    ("Relay Hub B",   "Marketing Dept",  800),
    ("Relay Hub B",   "IT Dept",         100),
    ("Relay Hub B",   "Operations Dept", 100),
    ("Relay Hub B",   "Relay Hub A",     500),
]

arcs = [(a, b) for a, b, _ in arcs_data]
costs = {(a, b): c for a, b, c in arcs_data}
capacities = {arc: 2000 for arc in arcs}  # max 2,000 requests per link

## model
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

## objective: minimize total routing cost
model.setObjective(
    gp.quicksum(costs[(i, j)] * X[(i, j)] for i, j in arcs),
    GRB.MINIMIZE,
)

model.optimize()

print("================ Network / Shortest Path: Routing ================")
if model.Status == GRB.OPTIMAL:
    print(f"Minimum total routing cost: ${model.ObjVal:,.0f}\n")
    print(f"{'Arc':<45} {'Flow':>8} {'Cap':>6} {'Cost/req':>10} {'Total Cost':>12}")
    print("-" * 85)
    for i, j in arcs:
        flow = X[(i, j)].X
        if flow > 0.5:
            total_cost = costs[(i, j)] * flow
            print(
                f"{i} -> {j:<30} {flow:>8.0f} {capacities[(i,j)]:>6} {costs[(i,j)]:>10} {total_cost:>12,.0f}"
            )
    print("\nDemand fulfillment:")
    demand_nodes = ["Marketing Dept", "IT Dept", "Operations Dept"]
    for d in demand_nodes:
        inflow = sum(X[(j, d)].X for j in nodes if (j, d) in arcs)
        print(f"  {d}: received {inflow:.0f} / needed {demands[d]}")
else:
    print("No optimal solution found.")
