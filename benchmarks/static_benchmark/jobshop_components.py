from typing import Dict, List

class Operation:
    """Represents a single operation in a job."""
    def __init__(self, operation_id: int, job_id: int, machine_processing_times: Dict[int, int]):
        self.operation_id = operation_id
        self.job_id = job_id
        self.machine_processing_times = machine_processing_times  # machine_id -> processing_time
    @property
    def compatible_machines(self) -> List[int]:
        return list(self.machine_processing_times.keys())
    @property
    def min_processing_time(self) -> int:
        return min(self.machine_processing_times.values())
    @property
    def max_processing_time(self) -> int:
        return max(self.machine_processing_times.values())
    def get_processing_time(self, machine_id: int) -> int:
        if machine_id not in self.machine_processing_times:
            raise ValueError(f"Machine {machine_id} is not compatible with operation {self.operation_id}")
        return self.machine_processing_times[machine_id]
    def __str__(self):
        machines_str = ", ".join([f"M{m}->{t}" for m, t in self.machine_processing_times.items()])
        return f"Job{self.job_id}-Op{self.operation_id} on {machines_str}"

class Job:
    """Represents a job with multiple operations, due date, and weight."""
    def __init__(self, job_id: int, operations: List[Operation], due_date: int, weight: int):
        self.job_id = job_id
        self.operations = operations
        self.due_date = due_date
        self.weight = weight
    def __str__(self):
        return f"Job{self.job_id} with {len(self.operations)} operations, due={self.due_date}, weight={self.weight}"
    @property
    def total_processing_time(self) -> int:
        return sum(op.min_processing_time for op in self.operations)
    def get_tardiness(self, completion_time: int) -> int:
        """Calculate tardiness: max(0, completion_time - due_date)"""
        return max(0, completion_time - self.due_date)
    def get_weighted_tardiness(self, completion_time: int) -> int:
        """Calculate weighted tardiness: weight * max(0, completion_time - due_date)"""
        return self.weight * self.get_tardiness(completion_time) 