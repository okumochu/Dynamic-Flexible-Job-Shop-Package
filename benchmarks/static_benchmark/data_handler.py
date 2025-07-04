import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import json
from .jobshop_components import Operation, Job
from .data_generator import DataGenerator

class FlexibleJobShopDataHandler:
    """
    A comprehensive data handler for Flexible Job Shop Scheduling Problems (FJSP).
    
    Supports multiple data sources through the DataGenerator interface:
    - Loading from existing datasets (Brandimarte, Hurink formats)
    - Generating synthetic data for simulation
    
    Provides unified interface for:
    - RL algorithms
    - MILP solvers
    - Genetic Algorithms
    - Other optimization methods
    """
    
    def __init__(self, data_source: Union[str, Dict], data_type: str = "dataset"):
        """
        Initialize the data handler.
        
        Args:
            data_source: Either a dataset path (str) or simulation parameters (Dict)
            data_type: Either "dataset" or "simulation"
        """
        self.data_source = data_source
        self.data_type = data_type
        self.jobs: List[Job] = []
        self.operations: List[Operation] = []
        self.num_jobs: int = 0
        self.num_machines: int = 0
        self.num_operations: int = 0
        
        # Derived properties
        self.machine_operations: Dict[int, List[Operation]] = {}
        
        # Load data using the DataGenerator interface
        self._load_data()
    
    def _load_data(self) -> None:
        """Load data using the DataGenerator interface."""
        if self.data_type == "dataset":
            if isinstance(self.data_source, str):
                self.jobs, self.operations, self.num_jobs, self.num_machines, self.num_operations = \
                    DataGenerator.load_from_dataset(self.data_source)
            else:
                raise ValueError("For dataset type, data_source must be a string (file path)")
        
        elif self.data_type == "simulation":
            if isinstance(self.data_source, dict):
                self.jobs, self.operations, self.num_jobs, self.num_machines, self.num_operations = \
                    DataGenerator.generate_synthetic_data(**self.data_source)
            else:
                raise ValueError("For simulation type, data_source must be a dictionary of parameters")
        
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}. Use 'dataset' or 'simulation'")
        
        self._build_derived_structures()
    
    def _build_derived_structures(self) -> None:
        """Build derived data structures for efficient access."""
        # Machine operations mapping
        self.machine_operations = {i: [] for i in range(self.num_machines)}
        for operation in self.operations:
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
            for machine_id in operation.compatible_machines:
                self.processing_time_matrix[operation.operation_id, machine_id] = operation.get_processing_time(machine_id)
        
    
    # ===== PUBLIC INTERFACE METHODS =====
    
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
    
    def get_machine_load(self, machine_id: int) -> int:
        """Get total processing time assigned to a specific machine."""
        if machine_id < 0 or machine_id >= self.num_machines:
            raise ValueError(f"Invalid machine_id: {machine_id}")
        
        total_load = 0
        for operation in self.machine_operations[machine_id]:
            if machine_id in operation.machine_processing_times:
                total_load += operation.get_processing_time(machine_id)
        return total_load
    
    def get_operations_each_jobs(self) -> List[List[int]]:
        """Get list of operation_id for each job."""
        return [[op.operation_id for op in job.operations] for job in self.jobs]
    
    def get_compatible_machines_each_jobs(self) -> List[List[List[int]]]:
        """Get list of compatible machine_id for each operation in each job."""
        return [[op.compatible_machines for op in job.operations] for job in self.jobs]
    
    def get_operation_info(self, operation_id: int) -> Tuple[int, int]:
        """Get job_id and operation position for a given operation_id."""
        # utilize job_operation_matrix to get job_id and operation position
        job_indices = np.where(self.job_operation_matrix == operation_id)[0]
        if len(job_indices) == 0:
            raise ValueError(f"Operation {operation_id} not found")
        
        job_id = job_indices[0]
        operation_position = np.where(self.job_operation_matrix[job_id] == operation_id)[0][0]
        return job_id, operation_position
    
    def get_job_due_date(self, job_id: int) -> int:
        """Get due date for a specific job."""
        if job_id < 0 or job_id >= self.num_jobs:
            raise ValueError(f"Invalid job_id: {job_id}")
        return self.jobs[job_id].due_date
    
    def get_job_weight(self, job_id: int) -> int:
        """Get weight for a specific job."""
        if job_id < 0 or job_id >= self.num_jobs:
            raise ValueError(f"Invalid job_id: {job_id}")
        return self.jobs[job_id].weight
    
    def get_jobs_due_date(self) -> List[int]:
        """Get list of due dates for all jobs."""
        return [job.due_date for job in self.jobs]
    
    def get_jobs_weight(self) -> List[int]:
        """Get list of weights for all jobs."""
        return [job.weight for job in self.jobs]
    
    def get_total_weighted_tardiness(self, completion_times: Dict[int, int]) -> int:
        """
        Calculate total weighted tardiness for given completion times.
        
        Args:
            completion_times: Dictionary mapping job_id to completion_time
            
        Returns:
            Total weighted tardiness
        """
        total_twt = 0
        for job in self.jobs:
            if job.job_id in completion_times:
                total_twt += job.get_weighted_tardiness(completion_times[job.job_id])
        
        return total_twt
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the problem instance."""
        due_dates = self.get_jobs_due_date()
        weights = self.get_jobs_weight()
        
        stats = {
            "num_jobs": self.num_jobs,
            "num_machines": self.num_machines,
            "num_operations": self.num_operations,
            "avg_operations_per_job": self.num_operations / self.num_jobs,
            "avg_processing_time": sum(op.min_processing_time for op in self.operations) / self.num_operations,
            "machine_loads": {f"machine_{i}": self.get_machine_load(i) for i in range(self.num_machines)},
            "job_processing_times": {f"job_{job.job_id}": job.total_processing_time for job in self.jobs},
            "data_type": self.data_type,
            "data_source": str(self.data_source) if isinstance(self.data_source, str) else "simulation",
            "due_dates": {f"job_{i}": due_dates[i] for i in range(self.num_jobs)},
            "weights": {f"job_{i}": weights[i] for i in range(self.num_jobs)},
            "avg_due_date": sum(due_dates) / len(due_dates),
            "avg_weight": sum(weights) / len(weights)
        }
        
        return stats
    
    def __str__(self) -> str:
        """String representation of the problem instance."""
        stats = self.get_statistics()
        data_source_str = f"Dataset: {self.data_source}" if self.data_type == "dataset" else "Simulation"
        return f"""Flexible Job Shop Problem ({data_source_str}):
Jobs: {self.num_jobs}, Machines: {self.num_machines}, Operations: {self.num_operations}
Total Processing Time: {stats['total_processing_time']}
Average Operations per Job: {stats['avg_operations_per_job']:.2f}
"""
    
    def __repr__(self) -> str:
        return f"FlexibleJobShopDataHandler(jobs={self.num_jobs}, machines={self.num_machines}, type={self.data_type})"


# ===== UTILITY FUNCTIONS =====

def load_multiple_datasets(dataset_dir: str, pattern: str = "*.txt") -> Dict[str, FlexibleJobShopDataHandler]:
    """Load multiple datasets from a directory."""
    import glob
    
    datasets = {}
    for filepath in glob.glob(os.path.join(dataset_dir, pattern)):
        filename = os.path.basename(filepath)
        try:
            handler = FlexibleJobShopDataHandler(data_source=filepath, data_type="dataset")
            datasets[filename] = handler
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return datasets


def generate_multiple_simulations(
    num_instances: int,
    base_params: Dict,
    seed_range: Optional[Tuple[int, int]] = None
) -> Dict[str, FlexibleJobShopDataHandler]:
    """Generate multiple simulation instances."""
    import random
    
    simulations = {}
    for i in range(num_instances):
        params = base_params.copy()
        if seed_range is not None:
            params['seed'] = random.randint(seed_range[0], seed_range[1])
        
        try:
            handler = FlexibleJobShopDataHandler(data_source=params, data_type="simulation")
            simulations[f"sim_{i+1}"] = handler
        except Exception as e:
            print(f"Error generating simulation {i+1}: {e}")
    
    return simulations
