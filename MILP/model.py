import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple, Optional
import numpy as np
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler


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
    3. Job precedence: s[current] >= s[prev] + sum_k p[prev,k]*x[prev,k]
    4. Makespan: C_max >= s[i] + sum_k p[i,k]*x[i,k]
    5. Tardiness: T[j] >= C[j] - d[j], T[j] >= 0 (where C[j] is completion time of job j)
    
    Objective: Minimize α * C_max + (1-α) * sum(w[j] * T[j])
    where α is the weight for makespan vs weighted tardiness
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
        self.job_operations = data_handler.get_job_operations_list()
        self.operation_machines = data_handler.get_operation_machines_list()
        
        # Due dates and weights
        self.due_dates = data_handler.get_job_due_dates()
        self.weights = data_handler.get_job_weights()
        
        # Gurobi model
        self.model = gp.Model("FlexibleJobShop")
        self.pair_indices = []  # (i, j, k) for unordered pairs sharing a machine
        self.heuristic_makespan = int(np.sum(self.processing_times))
        self.min_processing_time = int(np.min(self.processing_times[np.nonzero(self.processing_times)]))
        
    def get_operation_info(self, operation_id: int):
        """
        Get job_id and operation_index for a given operation_id.
        Args:
            operation_id: Operation ID to look up
        Returns:
            Tuple of (job_id, operation_index)
        """
        return self.data_handler.get_operation_info(operation_id)

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
        
        # Start times for each operation
        self.s = self.model.addVars(self.num_operations, vtype=GRB.CONTINUOUS, name="s")
        
        # Routing: x[i,k] = 1 if operation i assigned to machine k
        self.x = self.model.addVars(self.num_operations, self.num_machines, vtype=GRB.BINARY, name="x")
        
        # Build pair_indices: unordered pairs (i, j, k) where i < j and both can use machine k
        self.pair_indices = []
        for k in range(self.num_machines):
            # Use data_handler's get_machine_operations to get all operations compatible with machine k
            ops_on_k = [op.operation_id for op in self.data_handler.get_machine_operations(k)]
            for idx1 in range(len(ops_on_k)):
                for idx2 in range(idx1 + 1, len(ops_on_k)):
                    i, j = ops_on_k[idx1], ops_on_k[idx2]
                    self.pair_indices.append((i, j, k))
                    
        # Disjunctive sequencing: z[i,j,k] = 1 if i precedes j on k
        self.z = self.model.addVars(self.pair_indices, vtype=GRB.BINARY, name="z")
        
        # Makespan
        self.C_max = self.model.addVar(vtype=GRB.CONTINUOUS, name="C_max")
        
        # Tardiness variables
        self.T = self.model.addVars(self.num_jobs, vtype=GRB.CONTINUOUS, name="T")
    
    def _create_constraints(self):
        
        # Routing: Each operation assigned to exactly one machine
        for i in range(self.num_operations):
            job_id, op_index = self.get_operation_info(i)
            compatible_machines = self.operation_machines[job_id][op_index]
            self.model.addConstr(gp.quicksum(self.x[i, k] for k in compatible_machines) == 1, name=f"routing_{i}")
    
        M = self.heuristic_makespan - self.min_processing_time
        for (i, j, k) in self.pair_indices:
            # Only for pairs where both can use machine k
            p_i_k = self.processing_times[i][k]
            p_j_k = self.processing_times[j][k]
            # s[i] + p[i,k] <= s[j] + M*(1-z[i,j,k])
            self.model.addConstr(
                self.s[i] + p_i_k <= self.s[j] + M * (1 - self.z[i, j, k]),
                name=f"disj1_{i}_{j}_{k}"
            )
            # s[j] + p[j,k] <= s[i] + M*z[i,j,k]
            self.model.addConstr(
                self.s[j] + p_j_k <= self.s[i] + M * self.z[i, j, k],
                name=f"disj2_{i}_{j}_{k}"
            )
            # z[i,j,k] + z[j,i,k] = x[i,k] + x[j,k] - 1
            if (j, i, k) in self.pair_indices:
                self.model.addConstr(
                    self.z[i, j, k] + self.z[j, i, k] == self.x[i, k] + self.x[j, k] - 1,
                    name=f"activate_{i}_{j}_{k}"
                )
    
        # Job precedence constraints
        for job_id in range(self.num_jobs):
            for op_index in range(1, len(self.job_operations[job_id])):
                current_op = self.job_operations[job_id][op_index]
                prev_op = self.job_operations[job_id][op_index - 1]
                # s[current] >= s[prev] + sum_k p[prev,k]*x[prev,k]
                self.model.addConstr(
                    self.s[current_op] >= self.s[prev_op] + gp.quicksum(self.processing_times[prev_op][k] * self.x[prev_op, k]
                                                                        for k in self.operation_machines[job_id][op_index - 1]),
                    name=f"precedence_{current_op}_{prev_op}"
                )
    
        # Makespan constraints
        for i in range(self.num_operations):
            job_id, op_index = self.get_operation_info(i)
            compatible_machines = self.operation_machines[job_id][op_index]
            self.model.addConstr(
                self.C_max >= self.s[i] + gp.quicksum(self.processing_times[i][k] * self.x[i, k] for k in compatible_machines),
                name=f"makespan_{i}"
            )
        
        # Tardiness constraints
        for job_id in range(self.num_jobs):
            # Get the last operation of this job
            last_op = self.job_operations[job_id][-1]
            job_id_check, op_index = self.get_operation_info(last_op)
            compatible_machines = self.operation_machines[job_id][op_index]
            
            # T[j] >= C[j] - d[j] (tardiness = completion_time - due_date)
            self.model.addConstr(
                self.T[job_id] >= self.s[last_op] + gp.quicksum(self.processing_times[last_op][k] * self.x[last_op, k] for k in compatible_machines) - self.due_dates[job_id],
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
        
        # Calculate makespan and TWT from solution
        makespan = self.C_max.X if self.C_max.X is not None else None
        total_weighted_tardiness = None
        
        if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]:
            # Calculate completion times and TWT
            completion_times = {}
            for job_id in range(self.num_jobs):
                last_op = self.job_operations[job_id][-1]
                job_id_check, op_index = self.get_operation_info(last_op)
                compatible_machines = self.operation_machines[job_id][op_index]
                
                # Find assigned machine and completion time
                for k in compatible_machines:
                    if (last_op, k) in self.x and self.x[last_op, k].X > 0.5:
                        completion_time = self.s[last_op].X + self.processing_times[last_op][k]
                        completion_times[job_id] = completion_time
                        break
            
            total_weighted_tardiness = self.data_handler.get_total_weighted_tardiness(completion_times)
        
        # Performance metrics
        performance = {
            "status": self._get_status_string(self.model.status),
            "objective": self.model.objVal if self.model.objVal is not None else None,
            "solve_time": self.model.Runtime,
            "makespan": makespan,
            "total_weighted_tardiness": total_weighted_tardiness
        }
        
        # Schedule results
        schedule_result = {}
        
        # Extract schedule if we have a solution (optimal or feasible)
        if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT] and self.C_max.X is not None:
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
        
        return {
            "performance": performance,
            "schedule_result": schedule_result
        }
    
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
