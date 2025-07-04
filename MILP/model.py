import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple, Optional
import numpy as np
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
from itertools import permutations  # for disjunctive constraints

class MILP:
    """
    Continuous-time disjunctive MILP model for Flexible Job Shop Problem using Gurobi.
    
    Decision Variables:
    - s[i]: Continuous start time of operation i
    - x[i,k]: Binary variable = 1 if operation i is assigned to machine k
    - z[i,j,k]: Binary variable = 1 if operation i precedes operation j on machine k (for unordered pairs sharing a compatible machine)
    - C_max: Makespan (continuous)
    - T[j]: Tardiness of job j (continuous)
    
    Constraints:
    1. Routing: Each operation assigned to exactly one machine (sum_k x[i,k] = 1)
    2. Disjunctive capacity: For each unordered pair (i,j) and machine k both can use, s[i] + p[i,k] <= s[j] + M*(1-z[i,j,k]), s[j] + p[j,k] <= s[i] + M*z[i,j,k], and z[i,j,k] + z[j,i,k] = x[i,k] + x[j,k] - 1
    3. Operation precedence: s[current] >= s[prev] + sum_k p[prev,k]*x[prev,k]
    4. Makespan: C_max >= s[i] + sum_k p[i,k]*x[i,k]
    5. Tardiness: T[j] >= C[j] - d[j], T[j] >= 0 (where C[j] is completion time of job j)
    
    Objective: Minimize alpha * C_max + (1-alpha) * sum(w[j] * T[j])
    where alpha is the weight for makespan vs weighted tardiness
    """
    
    def __init__(self, data_handler: FlexibleJobShopDataHandler, twt_weight: float = 0.5):
        """
        Initialize the MILP model.
        
        Args:
            data_handler: FlexibleJobShopDataHandler instance containing problem data
            twt_weight: Weight for total weighted tardiness (0.0 = makespan only, 1.0 = TWT only, 0.5 = balanced)
        """
        self.data_handler = data_handler
        self.twt_weight = twt_weight
        
        # Problem dimensions
        self.num_jobs = data_handler.num_jobs
        self.num_machines = data_handler.num_machines
        self.num_operations = data_handler.num_operations
        self.processing_times = data_handler.processing_time_matrix
        self.job_operations = data_handler.get_operations_each_jobs()
        self.compatible_machines_list = data_handler.get_compatible_machines_each_jobs()
        
        # Due dates and weights
        self.due_dates = data_handler.get_jobs_due_date()
        self.weights = data_handler.get_jobs_weight()
        
        # Gurobi model
        self.model = gp.Model("FlexibleJobShop")
        self.pair_indices = []  # (i, j, k) for unordered pairs sharing a machine
        self.heuristic_makespan = int(np.sum(self.processing_times)) # upper bound of big-M

    def build_model(self, time_limit: Optional[int] = None, MIPFocus: int = 1, verbose: int = 0):
        """Build the MILP model with all variables and constraints.
        
        Args:
            time_limit: Time limit for the solver in seconds
            MIPFocus: MIPFocus parameter for Gurobi (1: balance optimality and feasibility, 2: balance optimality and bound tightening, 3: balance feasibility and bound tightening)
        """
        
        # Set model parameters for better performance
        if time_limit is not None: self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('LogToConsole', verbose)
        self.model.setParam('MIPFocus', MIPFocus)

        # Create decision variables
        self._create_variables()
        
        # Create constraints
        self._create_constraints()
        
        # Set objective function
        self._set_objective()
        
        self.model.update()
    
    def _create_variables(self):
        """Create decision variables and objective variable"""
        
        # Start times for each operation (non-negative)
        self.s = self.model.addVars(self.num_operations, vtype=GRB.CONTINUOUS, lb=0, name="s")
        
        # Routing: x[i,k] = 1 if operation i assigned to machine k
        self.x = self.model.addVars(self.num_operations, self.num_machines, vtype=GRB.BINARY, name="x")
        
        # Disjunctive sequencing: z[i,j,k] = 1 if i precedes j on k, bi-directional
        for k in range(self.num_machines):
            operation_on_k = self.data_handler.get_machine_operations(k)
            operation_id_list = [operation.operation_id for operation in operation_on_k]
            for i, j in permutations(operation_id_list, 2):
                self.pair_indices.append((i, j, k))
        self.z = self.model.addVars(self.pair_indices, vtype=GRB.BINARY, name="z")
        
        # Makespan
        self.C_max = self.model.addVar(vtype=GRB.CONTINUOUS, name="C_max")
        
        # Tardiness variables
        self.T = self.model.addVars(self.num_jobs, vtype=GRB.CONTINUOUS, name="T")
    
    def _create_constraints(self):
        
        # Routing: Each operation assigned to exactly one machine
        for i in range(self.num_operations):
            job_id, op_position = self.data_handler.get_operation_info(i)
            compatible_machines = self.compatible_machines_list[job_id][op_position]
            self.model.addConstr(gp.quicksum(self.x[i, k] for k in compatible_machines) == 1, name=f"routing_{i}")
            # exclude incompatible machines
            for k in range(self.num_machines):
                if k not in compatible_machines:
                    self.model.addConstr(self.x[i, k] == 0, name=f"exclude_incompatible_machines_{i}_{k}")
        
        # Disjunctive constraints
        for (i, j, k) in self.pair_indices:
            # At least one of z[i,j,k] or z[j,i,k] must be 1
            self.model.addConstr(self.z[i, j, k] + self.z[j, i, k] >= self.x[i, k] + self.x[j, k] - 1, name=f"activate_{i}_{j}_{k}")
            # At most one of z[i,j,k] or z[j,i,k] can be 1
            self.model.addConstr(self.z[i, j, k] + self.z[j, i, k] <= 1, name=f"either_or_{i}_{j}_{k}")

            # If z[i,j,k] = 1, then s[i] + p[i,k] <= s[j]
            self.model.addGenConstrIndicator(self.z[i, j, k],  True, self.s[i] + self.processing_times[i][k] <= self.s[j])
            self.model.addGenConstrIndicator(self.z[j, i, k],  True, self.s[j] + self.processing_times[j][k] <= self.s[i])
    
        # Operation precedence constraints
        for job_id in range(self.num_jobs):
            for op_position in range(1, len(self.job_operations[job_id])):
                current_op = self.job_operations[job_id][op_position]
                prev_op = self.job_operations[job_id][op_position - 1]
                # s[current] >= s[prev] + sum_k p[prev,k]*x[prev,k]
                self.model.addConstr(
                    self.s[current_op] >= self.s[prev_op] + gp.quicksum(self.data_handler.get_processing_time(prev_op, k) * self.x[prev_op, k]
                                                                        for k in self.compatible_machines_list[job_id][op_position - 1]),
                    name=f"precedence_{current_op}_{prev_op}"
                )
        
        # Objective constraints
        for job_id in range(self.num_jobs):
            # Get the last operation of this job
            last_op_id = self.job_operations[job_id][-1]
            job_id_check, op_position = self.data_handler.get_operation_info(last_op_id)
            compatible_machines = self.compatible_machines_list[job_id][op_position]


            self.model.addConstr(
                self.C_max >= self.s[last_op_id] + gp.quicksum(self.data_handler.get_processing_time(last_op_id, k) * self.x[last_op_id, k] for k in compatible_machines),
                name=f"makespan_{last_op_id}"
            )

            # T[j] >= C[j] - d[j] (tardiness = completion_time - due_date)
            self.model.addConstr(
                self.T[job_id] >= self.s[last_op_id] + gp.quicksum(self.data_handler.get_processing_time(last_op_id, k) * self.x[last_op_id, k] for k in compatible_machines) - self.due_dates[job_id],
                name=f"tardiness_{job_id}"
            )
            
            # T[j] >= 0 (tardiness is non-negative)
            self.model.addConstr(
                self.T[job_id] >= 0,
                name=f"tardiness_nonneg_{job_id}"
            )
        
    def _set_objective(self):
        """Set objective function: weighted combination of makespan and total weighted tardiness."""
        # Weighted objective: α * C_max + (1-α) * sum(w[j] * T[j])
        makespan_weight = 1.0 - self.twt_weight
        twt_component = gp.quicksum(self.weights[job_id] * self.T[job_id] for job_id in range(self.num_jobs))
        self.model.setObjective(makespan_weight * self.C_max + self.twt_weight * twt_component, GRB.MINIMIZE)
    
    def solve(self) -> Dict:
        """
        Solve the MILP model.
        
        Returns:
            Dictionary with clean format:
            {
                "performance": {
                    "status": str,
                    "objective": float,
                    "solve_time": float,
                    "makespan": float,
                    "total_weighted_tardiness": float
                },
                "schedule_result": Dict[int, List[Tuple[int, float]]]
            }
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Solve the model
        self.model.optimize()
        
        # Check if we have a valid solution
        has_solution = (self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT] 
                       and self.C_max.X is not None)
        
        # Initialize results
        total_weighted_tardiness = None
        schedule_result = {}
        
        if has_solution:
            # Calculate TWT directly from job completion times
            total_weighted_tardiness = self._calculate_twt()
            # Extract machine schedule
            schedule_result = self._extract_machine_schedule()
        
        # Performance metrics
        performance = {
            "status": self._get_status_string(self.model.status),
            "objective": self.model.objVal if self.model.objVal is not None else None,
            "solve_time": self.model.Runtime,
            "makespan": self.C_max.X if has_solution else None,
            "total_weighted_tardiness": total_weighted_tardiness
        }
        
        return {
            "performance": performance,
            "schedule_result": schedule_result
        }
    
    def _calculate_twt(self):
        """Calculate total weighted tardiness directly from tardiness variables."""
        total_twt = 0
        for job_id in range(self.num_jobs):
            tardiness = self.T[job_id].X
            if tardiness is not None:
                weight = self.data_handler.get_job_weight(job_id)
                total_twt += weight * tardiness
        
        return total_twt
    
    def _extract_machine_schedule(self):
        """Extract machine schedule from solution."""
        schedule_result = {}
        
        # Extract machine schedule
        for i in range(self.num_operations):
            assigned_machine = None
            for k in range(self.num_machines):
                if (i, k) in self.x and self.x[i, k].X > 0.5:
                    assigned_machine = k
                    break
            if assigned_machine is not None:
                if assigned_machine not in schedule_result:
                    schedule_result[assigned_machine] = []
                schedule_result[assigned_machine].append((i, self.s[i].X))
        
        # Sort operations by start time for each machine
        for machine_id in schedule_result:
            schedule_result[machine_id].sort(key=lambda x: x[1])
        
        return schedule_result
    
    def _get_status_string(self, status: int) -> str:
        """Convert Gurobi status to readable string."""
        status_map = {
            GRB.OPTIMAL: "Optimal",
            GRB.TIME_LIMIT: "Time Limit",
            GRB.SOLUTION_LIMIT: "Solution Limit", 
            GRB.INFEASIBLE: "Infeasible",
            GRB.UNBOUNDED: "Unbounded",
            GRB.INF_OR_UNBD: "Infeasible or Unbounded"
        }
        return status_map.get(status, f"Unknown Status ({status})")
