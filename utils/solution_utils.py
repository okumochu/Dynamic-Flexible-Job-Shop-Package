from typing import List, Tuple, Dict, Optional
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from benchmark.data_handler import FlexibleJobShopDataHandler

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
        """Generate consistent colors for each job."""
        # Use a color palette that works well with plotly
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
        ]
        return {job_id: colors[job_id % len(colors)] for job_id in range(self.num_jobs)}
    
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
    
    def draw_gantt(self, figsize: Tuple[int, int] = (1200, 600), show_validation: bool = True):
        """
        Draw an interactive Gantt chart for the schedule using Plotly.
        
        Args:
            figsize: Figure size (width, height) in pixels
            show_validation: Whether to show validation results in title
        """
        if not self.validation_result["is_valid"] and show_validation:
            print("Warning: Solution has validation violations:")
            for violation in self.validation_result["violations"]:
                print(f"  - {violation}")
        
        # Prepare data for plotly
        gantt_data = []
        
        for machine_id, operations in self.machine_schedule.items():
            for op_id, start_time in operations:
                # Get operation info
                job_id, op_index = self.data_handler.get_operation_info(op_id)
                processing_time = self.data_handler.get_processing_time(op_id, machine_id)
                end_time = start_time + processing_time
                
                # Create task name
                task_name = f"Machine {machine_id} - Job {job_id}-{op_index}"
                
                # Add to gantt data
                gantt_data.append(dict(
                    Task=f"Machine {machine_id}",
                    Start=start_time,
                    Finish=end_time,
                    Resource=f"Job {job_id}",
                    Operation=f"O{op_index}",
                    ProcessingTime=processing_time,
                    OperationID=op_id,
                    JobID=job_id
                ))
        
        # Create the Gantt chart
        fig = ff.create_gantt(
            gantt_data,
            colors=self.job_colors,
            index_col='Resource',
            show_colorbar=True,
            group_tasks=True,
            showgrid_x=True,
            showgrid_y=True,
            height=figsize[1]
        )
        
        # Update layout
        fig.update_layout(
            title=f"Flexible Job Shop Schedule - Makespan: {self.validation_result['makespan']}",
            xaxis_title="Time",
            yaxis_title="Machine",
            width=figsize[0],
            height=figsize[1],
            font=dict(size=12),
            hovermode='closest'
        )
        
        # Customize hover template
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>" +
                         "Machine: %{y}<br>" +
                         "Start: %{x[0]}<br>" +
                         "End: %{x[1]}<br>" +
                         "Duration: %{customdata[1]} minutes<br>" +
                         "Operation ID: %{customdata[2]}<br>" +
                         "Job ID: %{customdata[3]}<extra></extra>",
            customdata=[[f"Job {d['JobID']}-{d['Operation']}", d['ProcessingTime'], d['OperationID'], d['JobID']] for d in gantt_data]
        )
        
        # Show the plot
        fig.show()
        
        return fig
    
    def get_schedule_summary(self) -> str:
        """Get a formatted summary of the schedule."""
        result = self.validation_result
        
        summary = f"""Schedule Summary:
Makespan: {result['makespan']}
Scheduled Operations: {result['scheduled_operations']}/{result['total_operations']}
Valid: {result['is_valid']}

Machine Schedule:"""
        
        for machine_id in sorted(self.machine_schedule.keys()):
            operations = self.machine_schedule[machine_id]
            summary += f"\nMachine {machine_id}:"
            for op_id, start_time in sorted(operations, key=lambda x: x[1]):
                job_id, op_index = self.data_handler.get_operation_info(op_id)
                processing_time = self.data_handler.get_processing_time(op_id, machine_id)
                end_time = start_time + processing_time
                summary += f"\n  - Operation {op_id} (Job {job_id}-{op_index}): {start_time}-{end_time} (duration: {processing_time})"
        
        if result['violations']:
            summary += "\n\nViolations:"
            for violation in result['violations']:
                summary += f"\n  - {violation}"
        
        return summary 