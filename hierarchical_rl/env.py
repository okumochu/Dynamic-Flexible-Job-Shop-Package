from benchmark.data_handler import FlexibleJobShopDataHandler
from typing import Union, Dict
import numpy as np

def get_state_representation(data_handler: FlexibleJobShopDataHandler, format_type: str = "matrix") -> Union[np.ndarray, Dict]:
    """
    Get state representation for RL algorithms.
    Args:
        data_handler: An instance of FlexibleJobShopDataHandler
        format_type: Type of representation ("matrix", "vector", "dict")
    Returns:
        State representation in the specified format
    """
    if format_type == "matrix":
        return data_handler.processing_time_matrix.copy()
    elif format_type == "vector":
        return data_handler.processing_time_matrix.flatten()
    elif format_type == "dict":
        return {
            "num_jobs": data_handler.num_jobs,
            "num_machines": data_handler.num_machines,
            "num_operations": data_handler.num_operations,
            "jobs": [
                {
                    "job_id": job.job_id,
                    "operations": [
                        {
                            "operation_id": op.operation_id,
                            "machine_processing_times": op.machine_processing_times
                        }
                        for op in job.operations
                    ]
                }
                for job in data_handler.jobs
            ]
        }
    else:
        raise ValueError(f"Unknown format_type: {format_type}") 