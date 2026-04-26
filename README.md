# Agent Cost Optimization

Four classic combinatorial optimization problems modelled in [Gurobi](https://www.gurobi.com/), each framed around AI-agent deployment scenarios. A generic Monte-Carlo simulation utility compares the true optimal solution against a distribution of random feasible solutions.

---

## Problems

### 1. Assignment — Agent-to-Project Allocation (`assignment.py`)
Assigns **7 specialist AI agents** to **9 projects** to maximize total suitability score.

| Constraint | Rule |
|---|---|
| Each project | assigned to exactly one agent |
| Each agent | assigned to at least one project |

*Objective:* **Maximize** total score across all assignments.

---

### 2. Knapsack — Agent Budget Selection (`knapsack.py`)
Selects the best subset of **10 AI agents** within a **$4,000 monthly budget** to maximize total token throughput.

| Agent | Monthly Cost | Tokens (M/month) |
|---|---|---|
| Intent Classifier | $800 | 50 |
| Inventory Checker | $1,200 | 70 |
| Fraud Detector | $2,500 | 90 |
| … | … | … |

*Objective:* **Maximize** total monthly tokens generated.

---

### 3. Network Routing — Agent Request Routing (`network_routing.py`)
Routes requests from two supply agents (**Coding Agent**, **Writing Agent**) through relay hubs to three demand departments (**Marketing**, **IT**, **Operations**), respecting arc capacities of 2,000 requests per link.

*Objective:* **Minimize** total routing cost.

---

### 4. Set Covering — Skill Coverage (`set_covering.py`)
Selects the minimum-cost subset of **12 AI agents** (at $20k each) that collectively cover all **10 required skills** (Planning, Tool Use, Knowledge, Analytics, Automation, Support, Governance, Collaboration, Monitoring, Optimization).

*Objective:* **Minimize** total cost while ensuring every skill is covered by at least one selected agent.

---

## Simulation (`simulation.py`)

`simulate_and_plot` is a generic utility that works with any of the four problem models. For each problem it:

1. Randomizes the objective coefficients 500 times and solves each perturbation to sample the feasible region.
2. Evaluates the *original* objective on every feasible solution found.
3. Solves for the true optimum.
4. Plots a histogram of simulated objective values with the optimal value overlaid.
5. Saves the plot to the `plots/` directory.

### Example output plots

| Problem | Plot |
|---|---|
| Assignment | `plots/assignment_—_agent-to-project_allocation.png` |
| Knapsack | `plots/knapsack_—_agent_budget_selection.png` |
| Network Routing | `plots/network_routing_—_agent_routing.png` |
| Set Covering | `plots/set_covering_—_skill_coverage.png` |

---

## Requirements

- Python 3.10+
- A valid [Gurobi licence](https://www.gurobi.com/academia/academic-program-and-licenses/) (free academic licences available)

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
seaborn==0.13.2
gurobipy==13.0.1
numpy==2.4.4
```

---

## Usage

**Run all four problems with simulation:**
```bash
python simulation.py
```

**Run a single problem (no simulation, prints solution table):**
```bash
python assignment.py
python knapsack.py
python network_routing.py
python set_covering.py
```

---

## Project Structure

```
.
├── assignment.py        # Agent-to-project allocation (maximize score)
├── knapsack.py          # Agent budget selection (maximize tokens)
├── network_routing.py   # Request routing through relay hubs (minimize cost)
├── set_covering.py      # Minimum-cost skill coverage (minimize cost)
├── simulation.py        # Generic Monte-Carlo simulation & plotting utility
├── requirements.txt
└── plots/               # Auto-generated output plots
```

Each problem file exposes a `build_model()` function that returns:
```python
(model, variables, obj_coeffs, obj_sense, problem_name, cost_unit)
```
This uniform interface lets `simulation.py` drive any of the four problems without modification.
