from typing import List, Tuple, Dict, Optional
import plotly.graph_objects as go
import random

class SolutionUtils:
    """
    Utility class for validating and visualizing Flexible Job Shop solutions.
    
    Args:
        machine_schedule: Dictionary mapping machine_id to list of (operation_id, start_time, end_time) tuples
        num_machines: Total number of machines
        num_operations: Total number of operations
        job_assignments: Dictionary mapping operation_id to job_id (required for coloring)
        job_due_dates: Dictionary mapping job_id to due_date (required for coloring and due date lines)
        machine_assignments: Dictionary mapping operation_id to list of valid machine_ids (required for validation)
        tolerance: Tolerance for floating-point comparisons
    """
    
    def __init__(self, 
                 machine_schedule: Dict[int, List[Tuple[int, float, float]]], 
                 num_machines: int,
                 num_operations: int,
                 job_assignments: Dict[int, int],
                 job_due_dates: Dict[int, float],
                 machine_assignments: Dict[int, List[int]],
                 tolerance: float = 1e-4):
        self.machine_schedule = machine_schedule
        self.num_machines = num_machines
        self.num_operations = num_operations
        self.job_assignments = job_assignments
        self.job_due_dates = job_due_dates
        self.machine_assignments = machine_assignments
        self.tolerance = tolerance  # Tolerance for floating-point comparisons
        
        # Generate consistent colors for jobs (same color for same due date)
        self.job_colors = self._generate_job_colors()
        
        # Validate and process the schedule
        self.validation_result = self._validate_solution()
    
    def _generate_job_colors(self) -> Dict[int, str]:
        """Generate unique colors for jobs based on due dates (same due date = same color)."""
        colors = []
        random.seed(42)  # For reproducibility
        def random_color():
            return f"#{random.randint(0, 0xFFFFFF):06x}"
        
        # Get unique due dates and assign colors
        due_date_to_color = {}
        for job_id, due_date in self.job_due_dates.items():
            if due_date not in due_date_to_color:
                color = random_color()
                while color in colors:
                    color = random_color()
                colors.append(color)
                due_date_to_color[due_date] = color
        
        # Assign colors to jobs based on their due dates
        job_colors = {}
        for job_id, due_date in self.job_due_dates.items():
            job_colors[job_id] = due_date_to_color[due_date]
        
        return job_colors
    
    def _is_greater_than(self, a: float, b: float) -> bool:
        """Compare two floats with tolerance to avoid precision errors."""
        return a > b + self.tolerance
    
    def _is_less_than(self, a: float, b: float) -> bool:
        """Compare two floats with tolerance to avoid precision errors."""
        return a < b - self.tolerance
    
    def _is_equal(self, a: float, b: float) -> bool:
        """Compare two floats for equality with tolerance."""
        return abs(a - b) <= self.tolerance
    
    def _validate_solution(self) -> Dict:
        """
        Validate the solution for:
        - No overlapping operations on same machine
        - No overlapping operations within same job
        - All operations scheduled
        - Valid machine assignments (operations only scheduled on valid machines)
        - Valid start/end times
        """
        violations = []
        scheduled_operations = set()
        operation_end_times = {}  # operation_id -> end_time
        operation_start_times = {}  # operation_id -> start_time
        
        # Check machine overlaps and collect scheduled operations
        for machine_id, operations in self.machine_schedule.items():
            if machine_id < 0 or machine_id >= self.num_machines:
                violations.append(f"Invalid machine_id: {machine_id}")
                continue
                
            # Sort operations by start time
            sorted_ops = sorted(operations, key=lambda x: x[1])
            
            for i, (op_id, start_time, end_time) in enumerate(sorted_ops):
                if op_id < 0 or op_id >= self.num_operations:
                    violations.append(f"Invalid operation_id: {op_id}")
                    continue
                
                # Validate start and end times
                if start_time < 0:
                    violations.append(f"Operation {op_id} has negative start time: {start_time}")
                    continue
                
                if end_time <= start_time:
                    violations.append(f"Operation {op_id} has invalid end time: {end_time} <= start time: {start_time}")
                    continue
                
                # Validate machine assignment
                if op_id in self.machine_assignments:
                    if machine_id not in self.machine_assignments[op_id]:
                        violations.append(f"Operation {op_id} cannot be processed on machine {machine_id}. Valid machines: {self.machine_assignments[op_id]}")
                        continue
                else:
                    violations.append(f"No machine assignment information for operation {op_id}")
                    continue
                
                operation_end_times[op_id] = end_time
                operation_start_times[op_id] = start_time
                scheduled_operations.add(op_id)
                
                # Check for overlaps with previous operations on same machine
                for j in range(i):
                    prev_op_id, prev_start, prev_end = sorted_ops[j]
                    
                    # Use tolerance for floating-point comparisons to avoid precision errors
                    if (self._is_less_than(start_time, prev_end) and 
                        self._is_greater_than(end_time, prev_start)):
                        violations.append(f"Overlap on machine {machine_id}: operations {prev_op_id} and {op_id}")
        
        # Check for overlaps within same job (job precedence constraints)
        job_operations = {}  # job_id -> list of (operation_id, start_time, end_time)
        for op_id, job_id in self.job_assignments.items():
            if op_id in operation_start_times and op_id in operation_end_times:
                if job_id not in job_operations:
                    job_operations[job_id] = []
                job_operations[job_id].append((
                    op_id, 
                    operation_start_times[op_id], 
                    operation_end_times[op_id]
                ))
        
        # Check for overlaps within each job
        for job_id, ops in job_operations.items():
            # Sort operations by start time
            sorted_job_ops = sorted(ops, key=lambda x: x[1])
            
            for i, (op_id, start_time, end_time) in enumerate(sorted_job_ops):
                # Check for overlaps with other operations in the same job
                for j in range(i + 1, len(sorted_job_ops)):
                    other_op_id, other_start, other_end = sorted_job_ops[j]
                    
                    # Check if operations overlap in time
                    if (self._is_less_than(start_time, other_end) and 
                        self._is_greater_than(end_time, other_start)):
                        violations.append(f"Job {job_id} operations overlap in time: operations {op_id} and {other_op_id}")
        
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
    
    def draw_gantt(self, show_validation: bool = True, title: Optional[str] = None):
        """
        Draw a numeric Gantt chart for the schedule using Plotly (minutes, not datetime).
        Each operation is a bar on its assigned machine, colored by job.
        Due date lines are shown with the same color as the job.
        
        Args:
            show_validation: Whether to show validation warnings
            title: Optional custom title for the chart
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
            for op_id, start_time, end_time in self.machine_schedule[machine_id]:
                # Get job ID for coloring
                job_id = self.job_assignments[op_id]
                color = self.job_colors[job_id]
                
                processing_time = end_time - start_time
                due_date = self.job_due_dates[job_id]
                
                bars.append(go.Bar(
                    x=[processing_time],
                    y=[f"Machine {machine_id}"],
                    base=[start_time],
                    orientation='h',
                    marker_color=color,
                    customdata=[[job_id, start_time, end_time, processing_time, op_id, due_date]],
                    hovertemplate=(
                        "Job %{customdata[0]}<br>" +
                        "Machine: %{y}<br>" +
                        "Start: %{customdata[1]:.2f} min<br>" +
                        "End: %{customdata[2]:.2f} min<br>" +
                        "Duration: %{customdata[3]:.2f} min<br>" +
                        "Operation ID: %{customdata[4]}<br>" +
                        "Due Date: %{customdata[5]:.2f} min<br>"
                    ),
                    showlegend=False
                ))
        
        fig = go.Figure(data=bars)
        
        # Add due date lines (same color as job)
        for job_id, due_date in self.job_due_dates.items():
            color = self.job_colors[job_id]
            fig.add_vline(
                x=due_date,
                line_dash="dash",
                line_color=color,
                opacity=0.7
            )
        
        # Set title
        if title is None:
            title = f"{self.num_operations} Operations on {self.num_machines} Machines<br>Makespan: {self.validation_result['makespan']:.1f} min"
        
        fig.update_layout(
            title=title,
            xaxis_title="Time (minutes)",
            yaxis_title="Machine",
            font=dict(size=12),
            yaxis=dict(categoryorder='array', categoryarray=yticks),
            xaxis=dict(rangemode='tozero'),
            hovermode='closest',
            barmode='stack',
            legend_title_text='Job'
        )
        
        return fig