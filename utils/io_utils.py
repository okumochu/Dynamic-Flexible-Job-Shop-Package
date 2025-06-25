import json
from benchmark.data_handler import FlexibleJobShopDataHandler, Job, Operation
from hierarchical_rl.env import get_state_representation

def save_to_json(data_handler: FlexibleJobShopDataHandler, filepath: str) -> None:
    """Save the problem instance to JSON format."""
    data = get_state_representation(data_handler, "dict")
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_from_json(filepath: str) -> FlexibleJobShopDataHandler:
    """Load a problem instance from JSON format and return a FlexibleJobShopDataHandler instance."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    handler = FlexibleJobShopDataHandler()
    handler.num_jobs = data["num_jobs"]
    handler.num_machines = data["num_machines"]
    handler.num_operations = data["num_operations"]
    handler.jobs = []
    handler.operations = []
    for job_data in data["jobs"]:
        job_operations = []
        for op_data in job_data["operations"]:
            operation = Operation(
                operation_id=op_data["operation_id"],
                job_id=op_data.get("job_id", job_data["job_id"]),
                machine_processing_times={int(k): int(v) for k, v in op_data["machine_processing_times"].items()}
            )
            job_operations.append(operation)
            handler.operations.append(operation)
        job = Job(job_id=job_data["job_id"], operations=job_operations)
        handler.jobs.append(job)
    handler._build_derived_structures()
    return handler 