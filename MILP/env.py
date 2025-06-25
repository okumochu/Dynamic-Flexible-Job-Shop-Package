from benchmark.data_handler import FlexibleJobShopDataHandler
from typing import Dict

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