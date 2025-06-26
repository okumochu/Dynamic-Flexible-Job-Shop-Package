from typing import List, Tuple, Dict, Optional
import plotly.graph_objects as go
from benchmark.data_handler import FlexibleJobShopDataHandler
import matplotlib.pyplot as plt
import random

class SolutionUtils:
    """
    Utility class for validating and visualizing Flexible Job Shop solutions.
    
    Args:
        data_handler: FlexibleJobShopDataHandler instance containing problem data
        machine_schedule: Dictionary mapping machine_id to list of (operation_id, start_time) tuples
    """
    
    def __init__(self, data_handler: FlexibleJobShopDataHandler, machine_schedule: Dict[int, List[Tuple[int, int]]]):
        self.data_handler = data_handler
        self.machine_schedule = machine_schedule
        self.num_jobs = data_handler.num_jobs
        self.num_machines = data_handler.num_machines
        self.num_operations = data_handler.num_operations
        
        # Generate consistent colors for jobs
        self.job_colors = self._generate_job_colors()
        
        # Validate and process the schedule
        self.validation_result = self._validate_solution()
    
    def _generate_job_colors(self) -> Dict[int, str]:
        """Generate exactly num_jobs unique colors by random sampling."""
        colors = []
        random.seed(42)  # For reproducibility
        def random_color():
            return f"#{random.randint(0, 0xFFFFFF):06x}"
        for i in range(self.num_jobs):
            color = random_color()
            while color in colors:
                color = random_color()
            colors.append(color)
        return {job_id: colors[job_id] for job_id in range(self.num_jobs)}
    
    def _validate_solution(self) -> Dict:
        """
        Validate the solution for:
        - No overlapping operations on same machine
        - Job precedence constraints
        - All operations scheduled
        - Valid machine assignments
        """
        violations = []
        scheduled_operations = set()
        operation_end_times = {}  # operation_id -> end_time
        
        # Check machine overlaps and collect scheduled operations
        for machine_id, operations in self.machine_schedule.items():
            if machine_id < 0 or machine_id >= self.num_machines:
                violations.append(f"Invalid machine_id: {machine_id}")
                continue
                
            # Sort operations by start time
            sorted_ops = sorted(operations, key=lambda x: x[1])
            
            for i, (op_id, start_time) in enumerate(sorted_ops):
                if op_id < 0 or op_id >= self.num_operations:
                    violations.append(f"Invalid operation_id: {op_id}")
                    continue
                
                # Get processing time for this operation on this machine
                processing_time = self.data_handler.get_processing_time(op_id, machine_id)
                if processing_time == 0:
                    violations.append(f"Operation {op_id} cannot be processed on machine {machine_id}")
                    continue
                
                end_time = start_time + processing_time
                operation_end_times[op_id] = end_time
                scheduled_operations.add(op_id)
                
                # Check for overlaps with previous operations on same machine
                for j in range(i):
                    prev_op_id, prev_start = sorted_ops[j]
                    prev_processing_time = self.data_handler.get_processing_time(prev_op_id, machine_id)
                    prev_end = prev_start + prev_processing_time
                    
                    if start_time < prev_end and end_time > prev_start:
                        violations.append(f"Overlap on machine {machine_id}: operations {prev_op_id} and {op_id}")
        
        # Check job precedence constraints
        for job_id in range(self.num_jobs):
            job_operations = self.data_handler.get_job_operations(job_id)
            for i in range(1, len(job_operations)):
                current_op = job_operations[i]
                prev_op = job_operations[i-1]
                
                if current_op.operation_id in operation_end_times and prev_op.operation_id in operation_end_times:
                    if operation_end_times[prev_op.operation_id] > operation_end_times[current_op.operation_id]:
                        violations.append(f"Job precedence violation in job {job_id}: operation {prev_op.operation_id} ends after {current_op.operation_id}")
        
        # Check if all operations are scheduled
        missing_operations = set(range(self.num_operations)) - scheduled_operations
        if missing_operations:
            violations.append(f"Missing operations: {missing_operations}")
        
        # Calculate makespan
        makespan = max(operation_end_times.values()) if operation_end_times else 0
        
        return {
            "is_valid": len(violations) == 0,
            "makespan": makespan,
            "violations": violations,
            "scheduled_operations": len(scheduled_operations),
            "total_operations": self.num_operations,
            "operation_end_times": operation_end_times
        }
    
    def validate_solution(self) -> Dict:
        """Get the validation result."""
        return self.validation_result
    
    def draw_gantt(self, show_validation: bool = True):
        """
        Draw a numeric Gantt chart for the schedule using Plotly (minutes, not datetime).
        Each operation is a bar on its assigned machine, colored by job.
        """
        if not self.validation_result["is_valid"] and show_validation:
            print("Warning: Solution has validation violations:")
            for violation in self.validation_result["violations"]:
                print(f"  - {violation}")
        
        # Prepare data for plotly
        bars = []
        yticks = []
        machine_ids = sorted(self.machine_schedule.keys())
        for machine_id in machine_ids:
            yticks.append(f"Machine {machine_id}")
            for op_id, start_time in self.machine_schedule[machine_id]:
                job_id, op_index = self.data_handler.get_operation_info(op_id)
                processing_time = self.data_handler.get_processing_time(op_id, machine_id)
                start_time = float(start_time)
                end_time = start_time + float(processing_time)
                color = self.job_colors[job_id]
                bars.append(go.Bar(
                    x=[processing_time],
                    y=[f"Machine {machine_id}"],
                    base=[start_time],
                    orientation='h',
                    marker_color=color,
                    customdata=[[job_id, op_index, start_time, end_time, processing_time, op_id]],
                    hovertemplate=(
                        "Job %{customdata[0]}-O%{customdata[1]}<br>" +
                        "Machine: %{y}<br>" +
                        "Start: %{customdata[2]:.2f} min<br>" +
                        "End: %{customdata[3]:.2f} min<br>" +
                        "Duration: %{customdata[4]} min<br>" +
                        "Operation ID: %{customdata[5]}<br>"
                    ),
                    showlegend=False
                ))
        fig = go.Figure(data=bars)
        fig.update_layout(
            title=f"{self.num_jobs} Jobs, {self.num_operations} Operations, Makespan: {self.validation_result['makespan']}",
            xaxis_title="Time (minutes)",
            yaxis_title="Machine",
            font=dict(size=12),
            yaxis=dict(categoryorder='array', categoryarray=yticks),
            xaxis=dict(rangemode='tozero'),
            hovermode='closest',
            barmode='stack',
            legend_title_text='Job'
        )
        fig.show()
        return fig