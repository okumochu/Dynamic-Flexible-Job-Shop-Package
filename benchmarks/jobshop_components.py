"""
Job Shop Components: Core data structures for Flexible Job Shop Scheduling Problem (FJSP)

This module defines the fundamental data structures used throughout the FJSP system:
- Operation: Represents a single operation that can be processed on multiple machines
- Job: Represents a job containing multiple operations with due dates and weights
"""

from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class Operation:
    """
    Represents a single operation in the Flexible Job Shop Scheduling Problem.
    
    An operation is a unit of work that must be performed on a machine.
    Each operation can be processed on multiple compatible machines with different processing times.
    """
    
    operation_id: int
    job_id: int
    machine_processing_times: Dict[int, int]  # machine_id -> processing_time
    
    @property
    def compatible_machines(self) -> Set[int]:
        """Get set of machines that can process this operation."""
        return set(self.machine_processing_times.keys())
    
    def get_processing_time(self, machine_id: int) -> int:
        """Get processing time for this operation on a specific machine."""
        if machine_id not in self.machine_processing_times:
            raise ValueError(f"Machine {machine_id} cannot process operation {self.operation_id}")
        return self.machine_processing_times[machine_id]
    
    @property
    def min_processing_time(self) -> int:
        """Get minimum processing time across all compatible machines."""
        return min(self.machine_processing_times.values())
    
    def __str__(self) -> str:
        return f"Operation(id={self.operation_id}, job={self.job_id}, machines={len(self.compatible_machines)})"
    
    def __repr__(self) -> str:
        return f"Operation(operation_id={self.operation_id}, job_id={self.job_id}, machine_processing_times={self.machine_processing_times})"


@dataclass
class Job:
    """
    Represents a job in the Flexible Job Shop Scheduling Problem.
    
    A job contains multiple operations that must be processed in sequence.
    Each job has a due date and weight for tardiness calculations.
    """
    
    job_id: int
    operations: List[Operation]
    due_date: int
    weight: int
    
    @property
    def total_processing_time(self) -> int:
        """Get total minimum processing time for all operations in this job."""
        return sum(op.min_processing_time for op in self.operations)
    
    def get_operation(self, operation_index: int) -> Operation:
        """Get operation at specific index within this job."""
        if operation_index < 0 or operation_index >= len(self.operations):
            raise IndexError(f"Operation index {operation_index} out of range for job {self.job_id}")
        return self.operations[operation_index]
    
    def get_weighted_tardiness(self, completion_time: int) -> int:
        """
        Calculate weighted tardiness for this job.
        
        Args:
            completion_time: Time when the job is completed
            
        Returns:
            Weighted tardiness = weight * max(0, completion_time - due_date)
        """
        tardiness = max(0, completion_time - self.due_date)
        return self.weight * tardiness
    
    def __str__(self) -> str:
        return f"Job(id={self.job_id}, operations={len(self.operations)}, due_date={self.due_date}, weight={self.weight})"
    
    def __repr__(self) -> str:
        return f"Job(job_id={self.job_id}, operations={self.operations}, due_date={self.due_date}, weight={self.weight})"
