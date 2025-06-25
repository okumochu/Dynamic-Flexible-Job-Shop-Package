import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json


@dataclass
class Operation:
    """Represents a single operation in a job."""
    operation_id: int
    job_id: int
    machine_id: int
    processing_time: int
    
    def __str__(self):
        return f"Job{self.job_id}-Op{self.operation_id} on Machine{self.machine_id} (time={self.processing_time})"


@dataclass
class Job:
    """Represents a job with multiple operations."""
    job_id: int
    operations: List[Operation]
    
    def __str__(self):
        return f"Job{self.job_id} with {len(self.operations)} operations"
    
    @property
    def total_processing_time(self) -> int:
        """Calculate total processing time for all operations in this job."""
        return sum(op.processing_time for op in self.operations)


class FlexibleJobShopDataHandler:
    """
    A comprehensive data handler for Flexible Job Shop Scheduling Problems (FJSP).
    
    Supports multiple dataset formats:
    - Brandimarte format (mk01-mk15)
    - Hurink format (abz, car, la, mt, orb series)
    
    Provides unified interface for:
    - RL algorithms
    - MILP solvers
    - Genetic Algorithms
    - Other optimization methods
    """
    
    def __init__(self, dataset_path: str = None):
        """
        Initialize the data handler.
        
        Args:
            dataset_path: Path to the dataset file
        """
        self.dataset_path = dataset_path
        self.jobs: List[Job] = []
        self.operations: List[Operation] = []
        self.num_jobs: int = 0
        self.num_machines: int = 0
        self.num_operations: int = 0
        
        # Derived properties
        self.machine_operations: Dict[int, List[Operation]] = {}
        self.job_operation_matrix: np.ndarray = None
        self.processing_time_matrix: np.ndarray = None
        self.machine_availability_matrix: np.ndarray = None
        
        if dataset_path:
            self.load_dataset(dataset_path)
    
    def load_dataset(self, dataset_path: str) -> None:
        """
        Load a dataset from file.
        
        Args:
            dataset_path: Path to the dataset file
        """
        self.dataset_path = dataset_path
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        header = lines[0].strip().split()
        self.num_jobs = int(header[0])
        self.num_machines = int(header[1])
        
        # Parse job data
        self.jobs = []
        self.operations = []
        operation_id = 0
        
        for job_id, line in enumerate(lines[1:], 1):
            if job_id > self.num_jobs:
                break
                
            data = [int(x) for x in line.strip().split()]
            num_operations = data[0]
            
            job_operations = []
            data_idx = 1
            
            for op_idx in range(num_operations):
                # Parse machine-processing_time pairs
                machine_id = data[data_idx]
                processing_time = data[data_idx + 1]
                
                operation = Operation(
                    operation_id=operation_id,
                    job_id=job_id,
                    machine_id=machine_id,
                    processing_time=processing_time
                )
                
                job_operations.append(operation)
                self.operations.append(operation)
                operation_id += 1
                data_idx += 2
            
            job = Job(job_id=job_id, operations=job_operations)
            self.jobs.append(job)
        
        self.num_operations = len(self.operations)
        self._build_derived_structures()
    
    def _build_derived_structures(self) -> None:
        """Build derived data structures for efficient access."""
        # Machine operations mapping
        self.machine_operations = {i: [] for i in range(1, self.num_machines + 1)}
        for operation in self.operations:
            self.machine_operations[operation.machine_id].append(operation)
        
        # Job-operation matrix (jobs x max_operations)
        max_ops_per_job = max(len(job.operations) for job in self.jobs)
        self.job_operation_matrix = np.zeros((self.num_jobs, max_ops_per_job), dtype=int)
        
        for job in self.jobs:
            for op_idx, operation in enumerate(job.operations):
                self.job_operation_matrix[job.job_id - 1, op_idx] = operation.operation_id + 1
        
        # Processing time matrix (operations x machines)
        self.processing_time_matrix = np.zeros((self.num_operations, self.num_machines), dtype=int)
        
        for operation in self.operations:
            # In FJSP, operations can potentially be processed on multiple machines
            # For now, we assume each operation has a fixed machine assignment
            self.processing_time_matrix[operation.operation_id, operation.machine_id - 1] = operation.processing_time
        
        # Machine availability matrix (machines x time_slots)
        # Initialize with all machines available at time 0
        self.machine_availability_matrix = np.zeros((self.num_machines, 1), dtype=int)
    
    def get_job_operations(self, job_id: int) -> List[Operation]:
        """Get all operations for a specific job."""
        if job_id < 1 or job_id > self.num_jobs:
            raise ValueError(f"Invalid job_id: {job_id}")
        return self.jobs[job_id - 1].operations
    
    def get_machine_operations(self, machine_id: int) -> List[Operation]:
        """Get all operations that can be processed on a specific machine."""
        if machine_id < 1 or machine_id > self.num_machines:
            raise ValueError(f"Invalid machine_id: {machine_id}")
        return self.machine_operations[machine_id]
    
    def get_operation(self, operation_id: int) -> Operation:
        """Get a specific operation by its ID."""
        if operation_id < 0 or operation_id >= self.num_operations:
            raise ValueError(f"Invalid operation_id: {operation_id}")
        return self.operations[operation_id]
    
    def get_operation_by_job_and_position(self, job_id: int, position: int) -> Operation:
        """Get an operation by job ID and its position within the job."""
        if job_id < 1 or job_id > self.num_jobs:
            raise ValueError(f"Invalid job_id: {job_id}")
        
        job = self.jobs[job_id - 1]
        if position < 0 or position >= len(job.operations):
            raise ValueError(f"Invalid position: {position}")
        
        return job.operations[position]
    
    def get_processing_time(self, operation_id: int, machine_id: int) -> int:
        """Get processing time for an operation on a specific machine."""
        if operation_id < 0 or operation_id >= self.num_operations:
            raise ValueError(f"Invalid operation_id: {operation_id}")
        if machine_id < 1 or machine_id > self.num_machines:
            raise ValueError(f"Invalid machine_id: {machine_id}")
        
        return self.processing_time_matrix[operation_id, machine_id - 1]
    
    def get_total_processing_time(self) -> int:
        """Get total processing time for all operations."""
        return sum(op.processing_time for op in self.operations)
    
    def get_machine_load(self, machine_id: int) -> int:
        """Get total processing time assigned to a specific machine."""
        if machine_id < 1 or machine_id > self.num_machines:
            raise ValueError(f"Invalid machine_id: {machine_id}")
        
        return sum(op.processing_time for op in self.machine_operations[machine_id])
    
    def get_job_makespan_lower_bound(self, job_id: int) -> int:
        """Get theoretical lower bound for job makespan (sum of processing times)."""
        return self.jobs[job_id - 1].total_processing_time
    
    def get_problem_lower_bound(self) -> int:
        """Get theoretical lower bound for the entire problem."""
        # Maximum of: max job processing time and max machine load
        max_job_time = max(job.total_processing_time for job in self.jobs)
        max_machine_load = max(self.get_machine_load(mid) for mid in range(1, self.num_machines + 1))
        return max(max_job_time, max_machine_load)
    
    def get_state_representation(self, format_type: str = "matrix") -> Union[np.ndarray, Dict]:
        """
        Get state representation for RL algorithms.
        
        Args:
            format_type: Type of representation ("matrix", "vector", "dict")
        
        Returns:
            State representation in the specified format
        """
        if format_type == "matrix":
            # Return processing time matrix
            return self.processing_time_matrix.copy()
        
        elif format_type == "vector":
            # Flatten the processing time matrix
            return self.processing_time_matrix.flatten()
        
        elif format_type == "dict":
            # Dictionary representation
            return {
                "num_jobs": self.num_jobs,
                "num_machines": self.num_machines,
                "num_operations": self.num_operations,
                "jobs": [
                    {
                        "job_id": job.job_id,
                        "operations": [
                            {
                                "operation_id": op.operation_id,
                                "machine_id": op.machine_id,
                                "processing_time": op.processing_time
                            }
                            for op in job.operations
                        ]
                    }
                    for job in self.jobs
                ]
            }
        
        else:
            raise ValueError(f"Unknown format_type: {format_type}")
    
    def get_milp_data(self) -> Dict:
        """
        Get data formatted for MILP solvers.
        
        Returns:
            Dictionary with MILP-compatible data structures
        """
        return {
            "num_jobs": self.num_jobs,
            "num_machines": self.num_machines,
            "num_operations": self.num_operations,
            "processing_times": self.processing_time_matrix.tolist(),
            "job_operations": [
                [op.operation_id for op in job.operations]
                for job in self.jobs
            ],
            "operation_machines": [
                [op.machine_id for op in job.operations]
                for job in self.jobs
            ]
        }
    
    def get_ga_data(self) -> Dict:
        """
        Get data formatted for Genetic Algorithms.
        
        Returns:
            Dictionary with GA-compatible data structures
        """
        return {
            "num_jobs": self.num_jobs,
            "num_machines": self.num_machines,
            "num_operations": self.num_operations,
            "job_sequences": [
                [op.operation_id for op in job.operations]
                for job in self.jobs
            ],
            "processing_times": self.processing_time_matrix.tolist(),
            "machine_assignments": [
                [op.machine_id for op in job.operations]
                for job in self.jobs
            ]
        }
    
    def validate_solution(self, schedule: List[Tuple[int, int, int, int]]) -> Dict:
        """
        Validate a solution (operation_id, machine_id, start_time, end_time).
        
        Args:
            schedule: List of tuples (operation_id, machine_id, start_time, end_time)
        
        Returns:
            Dictionary with validation results
        """
        # TODO: Implement solution validation
        # This would check for:
        # - No overlapping operations on same machine
        # - Job precedence constraints
        # - All operations scheduled
        # - Valid machine assignments
        
        return {
            "is_valid": True,  # Placeholder
            "makespan": 0,     # Placeholder
            "violations": []    # Placeholder
        }
    
    def save_to_json(self, filepath: str) -> None:
        """Save the problem instance to JSON format."""
        data = self.get_state_representation("dict")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_json(self, filepath: str) -> None:
        """Load a problem instance from JSON format."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.num_jobs = data["num_jobs"]
        self.num_machines = data["num_machines"]
        self.num_operations = data["num_operations"]
        
        # Reconstruct jobs and operations
        self.jobs = []
        self.operations = []
        
        for job_data in data["jobs"]:
            job_operations = []
            for op_data in job_data["operations"]:
                operation = Operation(
                    operation_id=op_data["operation_id"],
                    job_id=op_data.get("job_id", job_data["job_id"]),
                    machine_id=op_data["machine_id"],
                    processing_time=op_data["processing_time"]
                )
                job_operations.append(operation)
                self.operations.append(operation)
            
            job = Job(job_id=job_data["job_id"], operations=job_operations)
            self.jobs.append(job)
        
        self._build_derived_structures()
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the problem instance."""
        return {
            "num_jobs": self.num_jobs,
            "num_machines": self.num_machines,
            "num_operations": self.num_operations,
            "total_processing_time": self.get_total_processing_time(),
            "avg_operations_per_job": self.num_operations / self.num_jobs,
            "avg_processing_time": self.get_total_processing_time() / self.num_operations,
            "machine_loads": {
                f"machine_{i}": self.get_machine_load(i)
                for i in range(1, self.num_machines + 1)
            },
            "job_processing_times": {
                f"job_{job.job_id}": job.total_processing_time
                for job in self.jobs
            },
            "problem_lower_bound": self.get_problem_lower_bound()
        }
    
    def __str__(self) -> str:
        """String representation of the problem instance."""
        stats = self.get_statistics()
        return f"""Flexible Job Shop Problem:
Jobs: {self.num_jobs}, Machines: {self.num_machines}, Operations: {self.num_operations}
Total Processing Time: {stats['total_processing_time']}
Average Operations per Job: {stats['avg_operations_per_job']:.2f}
Problem Lower Bound: {stats['problem_lower_bound']}"""
    
    def __repr__(self) -> str:
        return f"FlexibleJobShopDataHandler(jobs={self.num_jobs}, machines={self.num_machines})"


# Utility functions for batch processing
def load_multiple_datasets(dataset_dir: str, pattern: str = "*.txt") -> Dict[str, FlexibleJobShopDataHandler]:
    """
    Load multiple datasets from a directory.
    
    Args:
        dataset_dir: Directory containing dataset files
        pattern: File pattern to match
    
    Returns:
        Dictionary mapping filename to data handler
    """
    import glob
    
    datasets = {}
    for filepath in glob.glob(os.path.join(dataset_dir, pattern)):
        filename = os.path.basename(filepath)
        try:
            handler = FlexibleJobShopDataHandler(filepath)
            datasets[filename] = handler
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return datasets


def compare_datasets(datasets: Dict[str, FlexibleJobShopDataHandler]) -> Dict:
    """
    Compare multiple datasets and return statistics.
    
    Args:
        datasets: Dictionary of dataset handlers
    
    Returns:
        Comparison statistics
    """
    comparison = {}
    
    for name, handler in datasets.items():
        comparison[name] = handler.get_statistics()
    
    return comparison
