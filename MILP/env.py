from benchmark.data_handler import FlexibleJobShopDataHandler
from typing import Dict, Tuple, Optional
import numpy as np

def get_milp_data(data_handler: FlexibleJobShopDataHandler) -> Dict:
    """
    Get data formatted for MILP solvers.
    Args:
        data_handler: An instance of FlexibleJobShopDataHandler
    Returns:
        Dictionary with MILP-compatible data structures
    """
    return {
        "num_jobs": data_handler.num_jobs,
        "num_machines": data_handler.num_machines,
        "num_operations": data_handler.num_operations,
        "processing_times": data_handler.processing_time_matrix.tolist(),
        "job_operations": [
            [op.operation_id for op in job.operations]
            for job in data_handler.jobs
        ],
        "operation_machines": [
            [op.compatible_machines for op in job.operations]
            for job in data_handler.jobs
        ]
    }

def calculate_time_horizon(processing_times: np.ndarray) -> int:
    """
    Calculate upper bound on makespan for time horizon.
    Args:
        processing_times: Processing time matrix
    Returns:
        Time horizon (upper bound on makespan)
    """
    # Sum of all processing times (very loose upper bound)
    total_processing_time = np.sum(processing_times)
    return int(total_processing_time)

def get_operation_info(operation_id: int, job_operations: list) -> Tuple[int, int]:
    """
    Get job_id and operation_index for a given operation_id.
    Args:
        operation_id: Operation ID to look up
        job_operations: List of job operations
    Returns:
        Tuple of (job_id, operation_index)
    """
    for job_id, operations in enumerate(job_operations):
        if operation_id in operations:
            op_index = operations.index(operation_id)
            return job_id, op_index
    raise ValueError(f"Operation {operation_id} not found")

def get_operation_predecessor(job_id: int, op_index: int, job_operations: list) -> Optional[int]:
    """
    Get the operation_id of the predecessor operation in the same job.
    Args:
        job_id: Job ID
        op_index: Operation index within the job
        job_operations: List of job operations
    Returns:
        Predecessor operation ID or None if first operation
    """
    if op_index == 0:
        return None
    return job_operations[job_id][op_index - 1]

def prepare_milp_data(data_handler: FlexibleJobShopDataHandler) -> Dict:
    """
    Prepare all data needed for MILP model construction.
    Args:
        data_handler: FlexibleJobShopDataHandler instance
    Returns:
        Dictionary with all prepared data for MILP
    """
    # Get basic MILP data
    milp_data = get_milp_data(data_handler)
    
    # Convert to numpy arrays for efficiency
    processing_times = np.array(milp_data["processing_times"])
    job_operations = milp_data["job_operations"]
    operation_machines = milp_data["operation_machines"]
    
    # Calculate time horizon
    time_horizon = calculate_time_horizon(processing_times)
    
    return {
        "num_jobs": milp_data["num_jobs"],
        "num_machines": milp_data["num_machines"],
        "num_operations": milp_data["num_operations"],
        "processing_times": processing_times,
        "job_operations": job_operations,
        "operation_machines": operation_machines,
        "time_horizon": time_horizon
    } 