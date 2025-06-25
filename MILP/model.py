import gurobipy as gp
from gurobipy import GRB
from typing import Dict
import numpy as np
from .env import prepare_milp_data, get_operation_info


class MILP:
    """
    MILP model for Flexible Job Shop Problem using Gurobi.
    
    Decision Variables:
    - x[i,k,t]: Binary variable = 1 if operation i starts on machine k at time t
    - C_max: Continuous variable representing the makespan (maximum completion time)
    
    Constraints:
    1. Each operation must be assigned to exactly one machine at one time
    2. Machine capacity: No overlapping operations on the same machine
    3. Job precedence: Operations within a job must follow sequence
    4. Makespan definition: C_max >= completion time of all operations
    """
    
    def __init__(self, data_handler):
        """
        Initialize the MILP model.
        
        Args:
            data_handler: FlexibleJobShopDataHandler instance containing problem data
        """
        # Prepare data using env functions
        self.data = prepare_milp_data(data_handler)
        
        # Problem dimensions
        self.num_jobs = self.data["num_jobs"]
        self.num_machines = self.data["num_machines"]
        self.num_operations = self.data["num_operations"]
        self.processing_times = self.data["processing_times"]
        self.job_operations = self.data["job_operations"]
        self.operation_machines = self.data["operation_machines"]
        self.time_horizon = self.data["time_horizon"]
        
        # Gurobi model
        self.model = None
        self.x = {}  # Decision variables x[i,k,t]
        self.C_max = None  # Makespan variable
        
    def build_model(self):
        """Build the MILP model with all variables and constraints."""
        # Create Gurobi model
        self.model = gp.Model("FlexibleJobShop")
        
        # Set model parameters for better performance
        self.model.setParam('TimeLimit', 300)  # 5 minutes time limit
        self.model.setParam('MIPGap', 0.01)    # 1% optimality gap
        
        # Create decision variables
        self._create_variables()
        
        # Create constraints
        self._create_constraints()
        
        # Set objective function
        self._set_objective()
        
        print(f"Model built with {self.model.NumVars} variables and {self.model.NumConstrs} constraints")
    
    def _create_variables(self):
        """Create decision variables."""
        # Binary variable x[i,k,t]: operation i starts on machine k at time t
        for i in range(self.num_operations):
            for k in range(self.num_machines):
                # Only create variables for compatible machines
                job_id, op_index = get_operation_info(i, self.job_operations)
                if k + 1 in self.operation_machines[job_id][op_index]:
                    for t in range(self.time_horizon):
                        self.x[i, k, t] = self.model.addVar(
                            vtype=GRB.BINARY,
                            name=f"x_{i}_{k}_{t}"
                        )
        
        # Continuous variable for makespan
        self.C_max = self.model.addVar(
            vtype=GRB.CONTINUOUS,
            name="C_max"
        )
        
        self.model.update()
    
    def _create_constraints(self):
        """Create all constraints."""
        self._create_operation_assignment_constraints()
        self._create_machine_capacity_constraints()
        self._create_job_precedence_constraints()
        self._create_makespan_constraints()
        
        self.model.update()
    
    def _create_operation_assignment_constraints(self):
        """Constraint 1: Each operation must be assigned to exactly one machine at one time."""
        for i in range(self.num_operations):
            # Get compatible machines for operation i
            job_id, op_index = get_operation_info(i, self.job_operations)
            compatible_machines = self.operation_machines[job_id][op_index]
            
            # Sum over all compatible machines and all start times = 1
            lhs = gp.quicksum(
                self.x[i, k-1, t]
                for k in compatible_machines
                for t in range(self.time_horizon)
                if (i, k-1, t) in self.x
            )
            self.model.addConstr(lhs == 1, name=f"assignment_{i}")
    
    def _create_machine_capacity_constraints(self):
        """Constraint 2: No overlapping operations on the same machine."""
        for k in range(self.num_machines):
            for t in range(self.time_horizon):
                # For each machine k and time t, ensure no overlap
                overlapping_operations = []
                
                for i in range(self.num_operations):
                    # Check if operation i can be processed on machine k
                    job_id, op_index = get_operation_info(i, self.job_operations)
                    compatible_machines = self.operation_machines[job_id][op_index]
                    
                    if k + 1 in compatible_machines:
                        # Check all possible start times that could overlap with time t
                        for start_time in range(max(0, t - self.processing_times[i][k] + 1), t + 1):
                            if (i, k, start_time) in self.x:
                                overlapping_operations.append(self.x[i, k, start_time])
                
                if overlapping_operations:
                    self.model.addConstr(
                        gp.quicksum(overlapping_operations) <= 1,
                        name=f"capacity_{k}_{t}"
                    )
    
    def _create_job_precedence_constraints(self):
        """Constraint 3: Operations within a job must follow sequence."""
        for job_id in range(self.num_jobs):
            for op_index in range(1, len(self.job_operations[job_id])):
                current_op = self.job_operations[job_id][op_index]
                prev_op = self.job_operations[job_id][op_index - 1]
                
                # Current operation must start after previous operation completes
                for k1 in range(self.num_machines):
                    for k2 in range(self.num_machines):
                        # Check if machines are compatible
                        current_compatible = self.operation_machines[job_id][op_index]
                        prev_compatible = self.operation_machines[job_id][op_index - 1]
                        
                        if k1 + 1 in current_compatible and k2 + 1 in prev_compatible:
                            for t1 in range(self.time_horizon):
                                for t2 in range(self.time_horizon):
                                    if (current_op, k1, t1) in self.x and (prev_op, k2, t2) in self.x:
                                        # t1 >= t2 + processing_time[prev_op][k2]
                                        self.model.addConstr(
                                            t1 >= t2 + self.processing_times[prev_op][k2],
                                            name=f"precedence_{current_op}_{prev_op}_{k1}_{k2}_{t1}_{t2}"
                                        )
    
    def _create_makespan_constraints(self):
        """Constraint 4: C_max >= completion time of all operations."""
        for i in range(self.num_operations):
            job_id, op_index = get_operation_info(i, self.job_operations)
            compatible_machines = self.operation_machines[job_id][op_index]
            
            for k in compatible_machines:
                k_idx = k - 1
                for t in range(self.time_horizon):
                    if (i, k_idx, t) in self.x:
                        # C_max >= t + processing_time[i][k]
                        self.model.addConstr(
                            self.C_max >= t + self.processing_times[i][k_idx],
                            name=f"makespan_{i}_{k}_{t}"
                        )
    
    def _set_objective(self):
        """Set objective function: minimize makespan."""
        self.model.setObjective(self.C_max, GRB.MINIMIZE)
    
    def solve(self) -> Dict:
        """
        Solve the MILP model.
        
        Returns:
            Dictionary containing solution information
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Solve the model
        self.model.optimize()
        
        # Collect solution
        solution = {
            "status": self.model.status,
            "makespan": None,
            "schedule": [],
            "solve_time": self.model.Runtime,
            "gap": self.model.MIPGap if self.model.status == GRB.OPTIMAL else None
        }
        
        if self.model.status == GRB.OPTIMAL:
            solution["makespan"] = self.C_max.X
            
            # Extract schedule
            for i in range(self.num_operations):
                for k in range(self.num_machines):
                    for t in range(self.time_horizon):
                        if (i, k, t) in self.x and self.x[i, k, t].X > 0.5:
                            job_id, op_index = get_operation_info(i, self.job_operations)
                            solution["schedule"].append({
                                "operation_id": i,
                                "job_id": job_id,
                                "machine_id": k + 1,
                                "start_time": t,
                                "end_time": t + self.processing_times[i][k],
                                "processing_time": self.processing_times[i][k]
                            })
        
        return solution
    
    def get_solution_summary(self, solution: Dict) -> str:
        """Get a formatted summary of the solution."""
        if solution["status"] == GRB.OPTIMAL:
            return f"""Solution Summary:
Status: Optimal
Makespan: {solution['makespan']:.2f}
Solve Time: {solution['solve_time']:.2f} seconds
Gap: {solution['gap']:.4f}
Number of Operations Scheduled: {len(solution['schedule'])}"""
        else:
            return f"""Solution Summary:
Status: {solution['status']}
Solve Time: {solution['solve_time']:.2f} seconds""" 