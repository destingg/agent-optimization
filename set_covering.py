import gurobipy as gp
from gurobipy import GRB

## data
data = {
    "skills": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    "agents": [
        "General Support Agent",
        "Access & Permissions Agent",
        "CRM / Relationship Management Agent",
        "Campaign Orchestration Agent",
        "Analytics & Insights Agent",
        "Billing & Transactions Agent",
        "Onboarding & Setup Agent",
        "Knowledge Management Agent",
        "Incident & Alert Handling Agent",
        "Procurement & Resources Agent",
        "Governance & Compliance Agent",
        "Cost & Performance Optimization Agent",
    ],
    "agent_skills": {
        "General Support Agent":                     ["F", "C", "H"],
        "Access & Permissions Agent":                ["F", "E", "G"],
        "CRM / Relationship Management Agent":       ["A", "B", "C"],
        "Campaign Orchestration Agent":              ["A", "B", "H"],
        "Analytics & Insights Agent":                ["C", "D", "I"],
        "Billing & Transactions Agent":              ["B", "D", "E"],
        "Onboarding & Setup Agent":                  ["E", "F", "H"],
        "Knowledge Management Agent":                ["C", "A"],
        "Incident & Alert Handling Agent":           ["A", "I", "G"],
        "Procurement & Resources Agent":             ["B", "E", "J"],
        "Governance & Compliance Agent":             ["G", "C", "I"],
        "Cost & Performance Optimization Agent":     ["D", "I", "J"],
    },
}

skill_labels = {
    "A": "Planning",
    "B": "Tool Use",
    "C": "Knowledge",
    "D": "Analytics",
    "E": "Automation",
    "F": "Support",
    "G": "Governance",
    "H": "Collaboration",
    "I": "Monitoring",
    "J": "Optimization",
}

skills = data["skills"]
agents = data["agents"]
agent_skills = data["agent_skills"]
agent_costs = {i: 20 for i in agents}  # $20k per agent

## model
model = gp.Model("Set Covering")
model.Params.OutputFlag = 0

## variables
X = model.addVars(agents, lb=0, ub=1, vtype=GRB.BINARY, name="x")

## constraints: every skill must be covered by at least one selected agent
for j in skills:
    model.addConstr(
        gp.quicksum(X[i] for i in agents if j in agent_skills[i]) >= 1,
        name=f"{j}_area_constr",
    )

## objective: minimize total cost of selected agents
model.setObjective(
    gp.quicksum(agent_costs[i] * X[i] for i in agents),
    GRB.MINIMIZE,
)

model.optimize()

print("================ Set-Covering: Skill Coverage ================")
if model.Status == GRB.OPTIMAL:
    selected = [i for i in agents if X[i].X > 0.5]
    print(f"Minimum agents needed : {len(selected)}")
    print(f"Total cost            : ${model.ObjVal:.0f}k")
    print("\nSelected agents:")
    for agent in selected:
        covered = ", ".join(
            f"{s}({skill_labels[s]})" for s in agent_skills[agent]
        )
        print(f"  - {agent}  [{covered}]")

    print("\nSkill coverage verification:")
    for s in skills:
        covering = [a for a in selected if s in agent_skills[a]]
        print(f"  {s} ({skill_labels[s]}): covered by {covering}")
else:
    print("No optimal solution found.")
