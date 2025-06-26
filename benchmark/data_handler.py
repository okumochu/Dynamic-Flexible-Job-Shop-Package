import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json


class Operation:
    """Represents a single operation in a job."""
    
    def __init__(self, operation_id: int, job_id: int, machine_processing_times: Dict[int, int]):
        self.operation_id = operation_id
        self.job_id = job_id
        self.machine_processing_times = machine_processing_times  # machine_id -> processing_time
    
    @property
    def compatible_machines(self) -> List[int]:
        """Get list of compatible machine IDs."""
        return list(self.machine_processing_times.keys())
    
    @property
    def min_processing_time(self) -> int:
        """Get minimum processing time across all compatible machines."""
        return min(self.machine_processing_times.values())
    
    @property
    def max_processing_time(self) -> int:
        """Get maximum processing time across all compatible machines."""
        return max(self.machine_processing_times.values())
    
    def get_processing_time(self, machine_id: int) -> int:
        """Get processing time for a specific machine."""
        if machine_id not in self.machine_processing_times:
            raise ValueError(f"Machine {machine_id} is not compatible with operation {self.operation_id}")
        return self.machine_processing_times[machine_id]
    
    def __str__(self):
        machines_str = ", ".join([f"M{m}->{t}" for m, t in self.machine_processing_times.items()])
        return f"Job{self.job_id}-Op{self.operation_id} on {machines_str}"


class Job:
    """Represents a job with multiple operations."""
    
    def __init__(self, job_id: int, operations: List[Operation]):
        self.job_id = job_id
        self.operations = operations
    
    def __str__(self):
        return f"Job{self.job_id} with {len(self.operations)} operations"
    
    @property
    def total_processing_time(self) -> int:
        """Calculate total processing time for all operations in this job."""
        return sum(op.min_processing_time for op in self.operations)


class FlexibleJobShopDataHandler:
    """
    A comprehensive data handler for Flexible Job Shop Scheduling Problems (FJSP).
    
    Supports on main dataset formats e.g. Brandimarte format (mk01-mk15) and Hurink format.
    
    Provides unified interface for:
    - RL algorithms
    - MILP solvers
    - Genetic Algorithms
    - Other optimization methods
    """
    
    def __init__(self, dataset_path: str):
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
        
        self._load_dataset(dataset_path)
    
    def _load_dataset(self, dataset_path: str) -> None:
        """
        Load a dataset from file.
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
        
        for job_id, line in enumerate(lines[1:], 0):
            if job_id > self.num_jobs:
                break
                
            data = [int(x) for x in line.strip().split()]
            num_operations = data[0]
            
            job_operations = []
            data_idx = 1
            
            for op_idx in range(num_operations):
                # Parse number of compatible machines for this operation
                num_compatible_machines = data[data_idx]
                data_idx += 1
                
                # Parse machine-processing_time pairs for this operation
                operation_machines = []
                for machine_idx in range(num_compatible_machines):
                    # machine id starts from 0 as the format in the dataset
                    machine_id = data[data_idx]
                    processing_time = data[data_idx + 1]
                    operation_machines.append((machine_id, processing_time))
                    data_idx += 2
                
                # Create operation with all compatible machines
                machine_processing_times = {machine_id: processing_time for machine_id, processing_time in operation_machines}
                operation = Operation(
                    operation_id=operation_id,
                    job_id=job_id,
                    machine_processing_times=machine_processing_times
                )
                
                job_operations.append(operation)
                self.operations.append(operation)
                operation_id += 1
            
            job = Job(job_id=job_id, operations=job_operations)
            self.jobs.append(job)
        
        self.num_operations = len(self.operations)
        self._build_derived_structures()
    
    def _build_derived_structures(self) -> None:
        """Build derived data structures for efficient access."""
        # Machine operations mapping
        self.machine_operations = {i: [] for i in range(self.num_machines)}
        for operation in self.operations:
            # Add operation to all compatible machines
            for machine_id in operation.compatible_machines:
                self.machine_operations[machine_id].append(operation)
        
        # Job-operation matrix (jobs x max_operations)
        max_ops_per_job = max(len(job.operations) for job in self.jobs)
        self.job_operation_matrix = np.zeros((self.num_jobs, max_ops_per_job), dtype=int)
        
        for job in self.jobs:
            for op_idx, operation in enumerate(job.operations):
                self.job_operation_matrix[job.job_id, op_idx] = operation.operation_id
        
        # Processing time matrix (operations x machines)
        self.processing_time_matrix = np.zeros((self.num_operations, self.num_machines), dtype=int)
        
        for operation in self.operations:
            # Set processing times for all compatible machines
            for machine_id in operation.compatible_machines:
                self.processing_time_matrix[operation.operation_id, machine_id] = operation.get_processing_time(machine_id)
        
        # Machine availability matrix (machines x time_slots)
        # Initialize with all machines available at time 0
        self.machine_availability_matrix = np.zeros((self.num_machines, 1), dtype=int)
    
    def get_job_operations(self, job_id: int) -> List[Operation]:
        """Get all operations for a specific job."""
        if job_id < 0 or job_id >= self.num_jobs:
            raise ValueError(f"Invalid job_id: {job_id}")
        return self.jobs[job_id].operations
    
    def get_machine_operations(self, machine_id: int) -> List[Operation]:
        """Get all operations that can be processed on a specific machine."""
        if machine_id < 0 or machine_id >= self.num_machines:
            raise ValueError(f"Invalid machine_id: {machine_id}")
        return self.machine_operations[machine_id]
    
    def get_operation(self, operation_id: int) -> Operation:
        """Get a specific operation by its ID."""
        if operation_id < 0 or operation_id >= self.num_operations:
            raise ValueError(f"Invalid operation_id: {operation_id}")
        return self.operations[operation_id]
    
    def get_operation_by_job_and_position(self, job_id: int, position: int) -> Operation:
        """Get an operation by job ID and its position within the job."""
        if job_id < 0 or job_id >= self.num_jobs:
            raise ValueError(f"Invalid job_id: {job_id}")
        
        job = self.jobs[job_id]
        if position < 0 or position >= len(job.operations):
            raise ValueError(f"Invalid position: {position}")
        
        return job.operations[position]
    
    def get_processing_time(self, operation_id: int, machine_id: int) -> int:
        """Get processing time for an operation on a specific machine."""
        if operation_id < 0 or operation_id >= self.num_operations:
            raise ValueError(f"Invalid operation_id: {operation_id}")
        if machine_id < 0 or machine_id >= self.num_machines:
            raise ValueError(f"Invalid machine_id: {machine_id}")
        
        return self.processing_time_matrix[operation_id, machine_id]
    
    def get_total_processing_time(self) -> int:
        """Get total processing time for all operations."""
        return sum(op.min_processing_time for op in self.operations)
    
    def get_machine_load(self, machine_id: int) -> int:
        """Get total processing time assigned to a specific machine."""
        if machine_id < 0 or machine_id >= self.num_machines:
            raise ValueError(f"Invalid machine_id: {machine_id}")
        
        total_load = 0
        for operation in self.machine_operations[machine_id]:
            if machine_id in operation.machine_processing_times:
                total_load += operation.get_processing_time(machine_id)
        return total_load
    
    
    def get_job_operations_list(self) -> List[List[int]]:
        """
        Get list of operation IDs for each job.
        Returns:
            List of lists where each inner list contains operation IDs for a job
        """
        return [
            [op.operation_id for op in job.operations]
            for job in self.jobs
        ]
    
    def get_operation_machines_list(self) -> List[List[List[int]]]:
        """
        Get list of compatible machines for each operation in each job.
        Returns:
            List of lists of lists where each innermost list contains compatible machine IDs
        """
        return [
            [op.compatible_machines for op in job.operations]
            for job in self.jobs
        ]
    
    def get_operation_info(self, operation_id: int) -> Tuple[int, int]:
        """
        Get job_id and operation_index for a given operation_id.
        Args:
            operation_id: Operation ID to look up
        Returns:
            Tuple of (job_id, operation_index)
        """
        for job_id, job in enumerate(self.jobs):
            for op_idx, operation in enumerate(job.operations):
                if operation.operation_id == operation_id:
                    return job_id, op_idx
        raise ValueError(f"Operation {operation_id} not found")
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the problem instance."""
        return {
            "num_jobs": self.num_jobs,
            "num_machines": self.num_machines,
            "num_operations": self.num_operations,
            "total_processing_time": self.get_total_processing_time(),
            "avg_operations_per_job": self.num_operations / self.num_jobs,
            "avg_processing_time": sum(op.min_processing_time for op in self.operations) / self.num_operations,
            "machine_loads": {
                f"machine_{i}": self.get_machine_load(i)
                for i in range(self.num_machines)
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
