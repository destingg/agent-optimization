import gurobipy as gp
from gurobipy import GRB

## data
data = {
    "agents": [
        "Analytics & Insights Agent",
        "Knowledge Management Agent",
        "Marketing Campaign Orchestration Agent",
        "Customer / Support Automation Agent",
        "Data Quality & Monitoring Agent",
        "Workflow Orchestration Agent",
        "Governance & Compliance Agent",
    ],
    "projects": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"],
    # rows = agents, cols = projects
    "values": [
        [9, 6, 8, 5, 9, 7, 8, 6, 7],  # Analytics & Insights Agent
        [6, 9, 5, 8, 6, 7, 5, 9, 6],  # Knowledge Management Agent
        [5, 6, 9, 4, 7, 9, 8, 5, 5],  # Marketing Campaign Orchestration Agent
        [4, 9, 6, 7, 5, 6, 7, 8, 9],  # Customer / Support Automation Agent
        [7, 5, 7, 6, 8, 5, 9, 5, 8],  # Data Quality & Monitoring Agent
        [6, 6, 7, 6, 7, 8, 8, 7, 9],  # Workflow Orchestration Agent
        [5, 6, 4, 9, 6, 5, 7, 6, 9],  # Governance & Compliance Agent
    ],
}

project_labels = {
    "P1": "Exec Metrics Dashboard",
    "P2": "CX Knowledge Bot",
    "P3": "Lead-Scoring Model",
    "P4": "Policy QA Assistant",
    "P5": "Experiment Analysis",
    "P6": "Feature Launch Brief",
    "P7": "Churn Risk Monitor",
    "P8": "Onboarding Guide Bot",
    "P9": "SLA Incident Triage",
}

agents = data["agents"]
projects = data["projects"]
values = data["values"]

# Build value lookup: (agent, project) -> score
value_map = {
    (agents[i], projects[j]): values[i][j]
    for i in range(len(agents))
    for j in range(len(projects))
}

## model
model = gp.Model("Assignment")
model.Params.OutputFlag = 0

## variables: X[agent, project] = 1 if agent is assigned to project
X = model.addVars(agents, projects, lb=0, ub=1, vtype=GRB.BINARY, name="x")

## constraints: each project is assigned exactly one agent
for j in projects:
    model.addConstr(
        gp.quicksum(X[(i, j)] for i in agents) == 1,
        name=f"{j}_project_constr",
    )

## constraints: each agent is assigned exactly one project
for i in agents:
    model.addConstr(
        gp.quicksum(X[(i, j)] for j in projects) == 1,
        name=f"{i}_agent_constr",
    )

## objective: maximize total suitability score
model.setObjective(
    gp.quicksum(value_map[(i, j)] * X[(i, j)] for i in agents for j in projects),
    GRB.MAXIMIZE,
)

model.optimize()

print("================ Assignment Problem: Resource Allocation ================")
if model.Status == GRB.OPTIMAL:
    print(f"Maximum total suitability score: {model.ObjVal:.0f}\n")
    print(f"{'Agent':<45} {'Project':<6} {'Description':<30} {'Score'}")
    print("-" * 95)
    for i in agents:
        for j in projects:
            if X[(i, j)].X > 0.5:
                print(
                    f"{i:<45} {j:<6} {project_labels[j]:<30} {value_map[(i, j)]}"
                )
else:
    print("No optimal solution found.")
