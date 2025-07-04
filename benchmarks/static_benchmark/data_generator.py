import os
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from .jobshop_components import Operation, Job

class DataGenerator:
    """
    Data generator for Flexible Job Shop Scheduling Problems (FJSP).
    
    Supports two data generation methods:
    1. Loading from existing datasets (Brandimarte, Hurink formats)
    2. Simulating random problem instances
    
    Always generates due dates and weights following Crauwels et al. 1998.
    """
    
    @staticmethod
    def load_from_dataset(dataset_path: str, TF: float = 0.3, RDD: float = 0.6, seed: Optional[int] = None) -> Tuple[List[Job], List[Operation], int, int, int]:
        """
        Load data from an existing dataset file and generate due dates/weights.
        
        Args:
            dataset_path: Path to the dataset file
            TF: Tardiness Factor for due date generation (0.2, 0.4, 0.6, 0.8, 1.0)
            RDD: Relative Range of Due Dates (0.2, 0.4, 0.6, 0.8, 1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (jobs, operations, num_jobs, num_machines, num_operations)
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        header = lines[0].strip().split()
        num_jobs = int(header[0])
        num_machines = int(header[1])
        
        # Parse job data
        jobs = []
        operations = []
        operation_id = 0
        
        # First pass: collect job operations and calculate processing times
        job_operations_list = []
        job_processing_times = []
        
        for job_id, line in enumerate(lines[1:], 0):
            if job_id >= num_jobs:
                break
                
            data = [int(x) for x in line.strip().split()]
            num_operations = data[0]
            
            job_operations = []
            data_idx = 1
            total_processing_time = 0
            
            for op_idx in range(num_operations):
                # Parse number of compatible machines for this operation
                num_compatible_machines = data[data_idx]
                data_idx += 1
                
                # Parse machine-processing_time pairs for this operation
                operation_machines = []
                min_processing_time = float('inf')
                for machine_idx in range(num_compatible_machines):
                    machine_id = data[data_idx]
                    processing_time = data[data_idx + 1]
                    operation_machines.append((machine_id, processing_time))
                    min_processing_time = min(min_processing_time, processing_time)
                    data_idx += 2
                
                # Create operation with all compatible machines
                machine_processing_times = {machine_id: processing_time for machine_id, processing_time in operation_machines}
                operation = Operation(
                    operation_id=operation_id,
                    job_id=job_id,
                    machine_processing_times=machine_processing_times
                )
                
                job_operations.append(operation)
                operations.append(operation)
                total_processing_time += min_processing_time
                operation_id += 1
            
            job_operations_list.append(job_operations)
            job_processing_times.append(total_processing_time)
        
        # Generate due dates and weights
        due_dates, weights = DataGenerator._generate_due_dates_and_weights(job_processing_times, TF, RDD, seed)
        
        # Create jobs with due dates and weights
        for job_id in range(num_jobs):
            job = Job(
                job_id=job_id, 
                operations=job_operations_list[job_id], 
                due_date=due_dates[job_id], 
                weight=weights[job_id]
            )
            jobs.append(job)
        
        num_operations = len(operations)
        return jobs, operations, num_jobs, num_machines, num_operations

    @staticmethod
    def _generate_due_dates_and_weights(
        job_processing_times: List[int],
        TF: float,
        RDD: float,
        seed: Optional[int] = None
    ) -> Tuple[List[int], List[int]]:
        """
        Generate due dates and weights for jobs following Crauwels et al. 1998.
        
        Args:
            job_processing_times: List of total processing times for each job
            TF: Tardiness Factor (0.2, 0.4, 0.6, 0.8, 1.0)
            RDD: Relative Range of Due Dates (0.2, 0.4, 0.6, 0.8, 1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (due_dates, weights) lists
        """
        if seed is not None:
            random.seed(seed)
        
        # Calculate total processing time for all jobs
        total_processing_time = sum(job_processing_times)
        
        # Calculate parameters for due date generation
        P_sum = total_processing_time
        
        # Generate due dates and weights for each job
        due_dates = []
        weights = []
        
        for job_processing_time in job_processing_times:
            # Generate weight: U(1, 10) as integer
            weight = random.randint(1, 10)
            weights.append(weight)
            
            # Generate due date following Crauwels et al. 1998
            # d_j = r_j + P_j + U(0, RDD * P_sum) * (1 - TF)
            # where r_j = 0 (all jobs available at time 0)
            # and P_j is the total processing time of job j
            
            due_date_offset = random.uniform(0, RDD * P_sum) * (1 - TF)
            due_date = int(job_processing_time + due_date_offset)
            due_dates.append(due_date)
        
        return due_dates, weights

    @staticmethod
    def generate_synthetic_data(
        num_jobs: int,
        num_machines: int,
        operation_lb: int = 2,
        operation_ub: int = 5,
        processing_time_lb: int = 20,
        processing_time_ub: int = 50,
        compatible_machines_lb: int = 1,
        compatible_machines_ub: Optional[int] = None,
        seed: Optional[int] = None,
        TF: float = 0.4,
        RDD: float = 0.8
    ) -> Tuple[List[Job], List[Operation], int, int, int]:
        """
        Generate synthetic FJSP data with specified parameters.
        
        Args:
            num_jobs: Number of jobs
            num_machines: Number of machines
            operation_lb: Lower bound for number of operations per job
            operation_ub: Upper bound for number of operations per job
            processing_time_lb: Lower bound for processing times
            processing_time_ub: Upper bound for processing times
            compatible_machines_lb: Lower bound for number of compatible machines per operation
            compatible_machines_ub: Upper bound for number of compatible machines per operation (default: num_machines)
            seed: Random seed for reproducibility
            TF: Tardiness Factor for due date generation (0.2, 0.4, 0.6, 0.8, 1.0)
                High TF means relaxed due dates
            RDD: Relative Range of Due Dates (0.2, 0.4, 0.6, 0.8, 1.0)
                High RDD means due dates are more spread out
            
        Returns:
            Tuple of (jobs, operations, num_jobs, num_machines, num_operations)
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Set default for compatible_machines_ub
        if compatible_machines_ub is None:
            compatible_machines_ub = num_machines
        
        # Validate parameters
        if operation_lb < 1 or operation_ub < operation_lb:
            raise ValueError("Invalid operation bounds")
        if processing_time_lb < 1 or processing_time_ub < processing_time_lb:
            raise ValueError("Invalid processing time bounds")
        if compatible_machines_lb < 1 or compatible_machines_ub > num_machines:
            raise ValueError("Invalid compatible machines bounds")
        if TF < 0 or TF > 1 or RDD < 0 or RDD > 1:
            raise ValueError("TF and RDD must be between 0 and 1")
        
        # Generate jobs and operations
        jobs = []
        operations = []
        operation_id = 0
        
        # First pass: collect job operations and calculate processing times
        job_operations_list = []
        job_processing_times = []
        
        for job_id in range(num_jobs):
            # Generate random number of operations for this job
            num_operations = random.randint(operation_lb, operation_ub)
            
            job_operations = []
            total_processing_time = 0
            
            for op_idx in range(num_operations):
                # Generate random number of compatible machines for this operation
                num_compatible_machines = random.randint(compatible_machines_lb, compatible_machines_ub)
                
                # Randomly select machines for this operation
                compatible_machines = random.sample(range(num_machines), num_compatible_machines)
                
                # Generate processing times for each compatible machine
                machine_processing_times = {}
                min_processing_time = float('inf')
                for machine_id in compatible_machines:
                    processing_time = random.randint(processing_time_lb, processing_time_ub)
                    machine_processing_times[machine_id] = processing_time
                    min_processing_time = min(min_processing_time, processing_time)
                
                # Create operation
                operation = Operation(
                    operation_id=operation_id,
                    job_id=job_id,
                    machine_processing_times=machine_processing_times
                )
                
                job_operations.append(operation)
                operations.append(operation)
                total_processing_time += min_processing_time
                operation_id += 1
            
            job_operations_list.append(job_operations)
            job_processing_times.append(total_processing_time)
        
        # Generate due dates and weights
        due_dates, weights = DataGenerator._generate_due_dates_and_weights(job_processing_times, TF, RDD, seed)
        
        # Create jobs with due dates and weights
        for job_id in range(num_jobs):
            job = Job(
                job_id=job_id, 
                operations=job_operations_list[job_id], 
                due_date=due_dates[job_id], 
                weight=weights[job_id]
            )
            jobs.append(job)
        
        num_operations = len(operations)
        return jobs, operations, num_jobs, num_machines, num_operations 