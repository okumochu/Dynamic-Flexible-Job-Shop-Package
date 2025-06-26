import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Tuple
import numpy as np
from benchmark.data_handler import FlexibleJobShopDataHandler


class MILP:
    """
    MILP model for Flexible Job Shop Problem using Gurobi.
    
    Decision Variables:
    - x[i,k,t]: Binary variable = 1 if operation i starts on machine k at time t
    
    Constraints:
    1. Each operation must be assigned to exactly one machine at one time
    2. Machine capacity: No overlapping operations on the same machine
    3. Job precedence: Operations within a job must follow sequence
    4. Makespan definition: C_max >= completion time of all operations
    """
    
    def __init__(self, data_handler: FlexibleJobShopDataHandler):
        """
        Initialize the MILP model.
        
        Args:
            data_handler: FlexibleJobShopDataHandler instance containing problem data
        """
        self.data_handler = data_handler
        
        # Problem dimensions
        self.num_jobs = data_handler.num_jobs
        self.num_machines = data_handler.num_machines
        self.num_operations = data_handler.num_operations
        self.processing_times = data_handler.processing_time_matrix
        self.time_horizon = int(np.sum(self.processing_times))

        # Get data from data handler
        self.job_operations = data_handler.get_job_operations_list()
        self.operation_machines = data_handler.get_operation_machines_list()
        
        # Gurobi model
        self.model = None
        self.x = {}  # Decision variables x[i,k,t]
        self.C_max = None  # Makespan variable
        
    def get_operation_info(self, operation_id: int):
        """
        Get job_id and operation_index for a given operation_id.
        Args:
            operation_id: Operation ID to look up
        Returns:
            Tuple of (job_id, operation_index)
        """
        return self.data_handler.get_operation_info(operation_id)

    def build_model(self, time_limit: int = None, MIPFocus: int = 1):
        """Build the MILP model with all variables and constraints.
        
        Args:
            time_limit: Time limit for the solver in seconds
            MIPFocus: MIPFocus parameter for Gurobi (1: balance optimality and feasibility, 2: balance optimality and bound tightening, 3: balance feasibility and bound tightening)
        """
        # Create Gurobi model
        self.model = gp.Model("FlexibleJobShop")
        
        # Set model parameters for better performance
        if time_limit is not None:
            self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('MIPFocus', MIPFocus)
        
        # Create decision variables
        self._create_variables()
        
        # Create constraints
        self._create_constraints()
        
        # Set objective function
        self._set_objective()
        
        print(f"Model built with {self.model.NumVars} variables and {self.model.NumConstrs} constraints")
    
    def _create_variables(self):
        """Create decision variables and objective variable"""
        
        # Decision variables x[i,k,t]: operation i starts on machine k at time t
        for i in range(self.num_operations):
            for k in range(self.num_machines):
                # Only create variables for compatible machines
                job_id, op_index = self.get_operation_info(i)
                if k in self.operation_machines[job_id][op_index]:
                    for t in range(self.time_horizon):
                        self.x[i, k, t] = self.model.addVar(
                            vtype=GRB.BINARY,
                            name=f"x_{i}_{k}_{t}"
                        )
        
        # Objective variable
        self.C_max = self.model.addVar(
            vtype=GRB.CONTINUOUS,
            name="C_max"
        )
        
        self.model.update()
    
    def _create_constraints(self):
        print("Creating assignment constraints...")
        # Each operation must be assigned to exactly one machine at one time.
        for i in range(self.num_operations):
            # Get compatible machines for operation i
            job_id, op_index = self.get_operation_info(i)
            compatible_machines = self.operation_machines[job_id][op_index]
            
            # Sum over all compatible machines and all start times = 1
            lhs = gp.quicksum(
                self.x[i, k, t]
                for k in compatible_machines
                for t in range(self.time_horizon)
                if (i, k, t) in self.x
            )
            self.model.addConstr(lhs == 1, name=f"assignment_{i}")
    
        print("Creating capacity constraints...")
        # No overlapping operations on the same machine.
        for k in range(self.num_machines):
            for t in range(self.time_horizon):
                # For each machine k and time t, ensure no overlap
                overlapping_operations = []
                
                for i in range(self.num_operations):
                    # Check if operation i can be processed on machine k
                    job_id, op_index = self.get_operation_info(i)
                    compatible_machines = self.operation_machines[job_id][op_index]
                    
                    if k in compatible_machines:
                        # Check all possible start times that could overlap with time t
                        for start_time in range(max(0, t - self.processing_times[i][k] + 1), t + 1):
                            if (i, k, start_time) in self.x:
                                overlapping_operations.append(self.x[i, k, start_time])
                
                if overlapping_operations:
                    self.model.addConstr(
                        gp.quicksum(overlapping_operations) <= 1,
                        name=f"capacity_{k}_{t}"
                    )
    
        print("Creating precedence constraints...")
        # Operations within a job must follow sequence.
        M = self.time_horizon + 1  # Big-M value, larger than any possible start time
        constraint_count = 0
        for job_id in range(self.num_jobs):
            for op_index in range(1, len(self.job_operations[job_id])):
                current_op = self.job_operations[job_id][op_index]
                prev_op = self.job_operations[job_id][op_index - 1]
                
                # Only add constraints for compatible machines
                current_compatible = self.operation_machines[job_id][op_index]
                prev_compatible = self.operation_machines[job_id][op_index - 1]
                
                for k1 in current_compatible:
                    for k2 in prev_compatible:
                        # Only add constraint if both variables exist
                        if (current_op, k1, 0) in self.x and (prev_op, k2, 0) in self.x:
                            # Simplified precedence constraint: if both operations are scheduled, current must start after prev ends
                            self.model.addConstr(
                                gp.quicksum(t * self.x[current_op, k1, t] for t in range(self.time_horizon) if (current_op, k1, t) in self.x) >= 
                                gp.quicksum((t + self.processing_times[prev_op][k2]) * self.x[prev_op, k2, t] for t in range(self.time_horizon) if (prev_op, k2, t) in self.x),
                                name=f"precedence_{current_op}_{prev_op}_{k1}_{k2}"
                            )
                            constraint_count += 1
                            if constraint_count % 1000 == 0:
                                print(f"  Created {constraint_count} precedence constraints...")
    
        print(f"Creating makespan constraints...")
        # C_max >= completion time of all operations.
        for i in range(self.num_operations):
            job_id, op_index = self.get_operation_info(i)
            compatible_machines = self.operation_machines[job_id][op_index]
            
            for k in compatible_machines:
                for t in range(self.time_horizon):
                    if (i, k, t) in self.x:
                        # C_max >= t + processing_time[i][k]
                        self.model.addConstr(
                            self.C_max >= t + self.processing_times[i][k] - (1 - self.x[i, k, t]) * self.time_horizon,
                            name=f"makespan_{i}_{k}_{t}"
                        )
        
        self.model.update()
        print(f"Constraints created successfully. Total constraints: {self.model.NumConstrs}")
    
    def _set_objective(self):
        """Set objective function: minimize makespan."""
        self.model.setObjective(self.C_max, GRB.MINIMIZE)
    
    def solve(self) -> Dict:
        """
        Solve the MILP model.
        
        Returns:
            Dictionary with clean format:
            {
                "performance": {
                    "status": str,
                    "objective": float,
                    "solve_time": float
                },
                "schedule_result": Dict[int, List[Tuple[int, int]]]
            }
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Solve the model
        self.model.optimize()
        
        # Performance metrics
        performance = {
            "status": self._get_status_string(self.model.status),
            "objective": self.C_max.X,
            "solve_time": self.model.Runtime,
        }
        
        # Schedule results
        schedule_result = {}
        
        # Extract schedule if we have a solution (optimal or feasible)
        if self.model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT] and self.C_max.X is not None:
            # Extract machine schedule
            for i in range(self.num_operations):
                for k in range(self.num_machines):
                    for t in range(self.time_horizon):
                        if (i, k, t) in self.x and self.x[i, k, t].X > 0.5:
                            if k not in schedule_result:
                                schedule_result[k] = []
                            schedule_result[k].append((i, t))
            
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
