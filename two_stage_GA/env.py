from benchmark.data_handler import FlexibleJobShopDataHandler
from typing import Dict

def get_ga_data(data_handler: FlexibleJobShopDataHandler) -> Dict:
    """
    Get data formatted for Genetic Algorithms.
    Args:
        data_handler: An instance of FlexibleJobShopDataHandler
    Returns:
        Dictionary with GA-compatible data structures
    """
    return {
        "num_jobs": data_handler.num_jobs,
        "num_machines": data_handler.num_machines,
        "num_operations": data_handler.num_operations,
        "job_sequences": [
            [op.operation_id for op in job.operations]
            for job in data_handler.jobs
        ],
        "processing_times": data_handler.processing_time_matrix.tolist(),
        "machine_assignments": [
            [op.compatible_machines for op in job.operations]
            for job in data_handler.jobs
        ]
    } 