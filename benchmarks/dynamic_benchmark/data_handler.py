from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from benchmarks.static_benchmark.jobshop_components import Job, Operation
from .events import Event, NewJobArrival, MachineBreakdown, PriorityChange


class DynamicFlexibleJobShopDataHandler:
    """Dynamic data handler for FJSP with events.

    Maintains an evolving set of jobs/operations and provides matrix
    representations similar to the static handler for RL consumption.
    """

    def __init__(
        self,
        num_machines: int,
        initial_jobs: Optional[List[Job]] = None,
        events: Optional[List[Event]] = None,
    ) -> None:
        self.num_machines: int = int(num_machines)
        self.jobs: Dict[int, Job] = {}
        self.operations: List[Operation] = []
        self.num_jobs: int = 0
        self.num_operations: int = 0

        self.machine_operations: Dict[int, List[Operation]] = {i: [] for i in range(self.num_machines)}
        self.job_operation_matrix: Optional[np.ndarray] = None
        self.processing_time_matrix: Optional[np.ndarray] = None

        self.events: List[Event] = list(events) if events is not None else []

        if initial_jobs:
            for job in initial_jobs:
                self._add_job(job)

        self._rebuild_matrices()

    # ===== Internal helpers =====
    def _add_job(self, job: Job) -> None:
        """Add a job to the handler and reindex its operations globally."""
        new_job_id = int(job.job_id)
        # Ensure unique job_id: if taken, shift to next available id
        while new_job_id in self.jobs:
            new_job_id += 1

        # Reindex operations to global unique ids, and set correct job_id
        for op in job.operations:
            op.job_id = new_job_id
            op.operation_id = self.num_operations
            self.operations.append(op)
            self.num_operations += 1
            for m in op.compatible_machines:
                self.machine_operations[m].append(op)

        job.job_id = new_job_id
        self.jobs[new_job_id] = job
        self.num_jobs = len(self.jobs)

    def _rebuild_matrices(self) -> None:
        """Rebuild job-operation and processing-time matrices."""
        if self.num_jobs == 0:
            self.job_operation_matrix = np.zeros((0, 0), dtype=int)
            self.processing_time_matrix = np.zeros((0, self.num_machines), dtype=int)
            return

        max_ops = max((len(job.operations) for job in self.jobs.values()), default=0)
        self.job_operation_matrix = np.zeros((self.num_jobs, max_ops), dtype=int)
        for j_idx, job_id in enumerate(sorted(self.jobs.keys())):
            for op_idx, op in enumerate(self.jobs[job_id].operations):
                self.job_operation_matrix[j_idx, op_idx] = op.operation_id

        self.processing_time_matrix = np.zeros((self.num_operations, self.num_machines), dtype=int)
        for op in self.operations:
            for m in op.compatible_machines:
                self.processing_time_matrix[op.operation_id, m] = op.get_processing_time(m)

    # ===== Public API (subset mirroring static handler) =====
    def get_job_operations(self, job_id: int) -> List[Operation]:
        if job_id not in self.jobs:
            raise ValueError(f"Invalid job_id: {job_id}")
        return self.jobs[job_id].operations

    def get_machine_operations(self, machine_id: int) -> List[Operation]:
        if machine_id < 0 or machine_id >= self.num_machines:
            raise ValueError(f"Invalid machine_id: {machine_id}")
        return self.machine_operations[machine_id]

    def get_operation(self, operation_id: int) -> Operation:
        if operation_id < 0 or operation_id >= self.num_operations:
            raise ValueError(f"Invalid operation_id: {operation_id}")
        return self.operations[operation_id]

    def get_processing_time(self, operation_id: int, machine_id: int) -> int:
        if self.processing_time_matrix is None:
            raise RuntimeError("Processing time matrix not initialized")
        if operation_id < 0 or operation_id >= self.num_operations:
            raise ValueError(f"Invalid operation_id: {operation_id}")
        if machine_id < 0 or machine_id >= self.num_machines:
            raise ValueError(f"Invalid machine_id: {machine_id}")
        return int(self.processing_time_matrix[operation_id, machine_id])

    def get_jobs_due_date(self) -> List[int]:
        return [self.jobs[jid].due_date for jid in sorted(self.jobs.keys())]

    def get_jobs_weight(self) -> List[int]:
        return [self.jobs[jid].weight for jid in sorted(self.jobs.keys())]

    def get_max_processing_time(self) -> int:
        if self.num_operations == 0:
            return 1
        return max(
            self.get_processing_time(op.operation_id, m)
            for op in self.operations
            for m in op.compatible_machines
        )

    def get_max_weight(self) -> int:
        if self.num_jobs == 0:
            return 1
        return max(self.get_jobs_weight())

    def get_max_due_date(self) -> int:
        if self.num_jobs == 0:
            return 1
        return max(self.get_jobs_due_date())

    # ===== Event application =====
    def has_pending_events(self) -> bool:
        return len(self.events) > 0

    def peek_next_event_time(self) -> Optional[float]:
        if not self.events:
            return None
        return float(getattr(self.events[0], "time", 0.0))

    def pop_events_up_to(self, t: float) -> List[Event]:
        idx = 0
        while idx < len(self.events) and float(getattr(self.events[idx], "time", 0.0)) <= float(t) + 1e-9:
            idx += 1
        evs = self.events[:idx]
        self.events = self.events[idx:]
        return evs

    def apply_event(self, event: Event) -> Tuple[str, Dict]:
        """Apply a single event to the data model.

        Returns a tuple (event_type_name, details) for logging/info.
        """
        if isinstance(event, NewJobArrival):
            # Expect attached job object from generator
            job: Optional[Job] = getattr(event, "job", None)
            if job is None:
                raise ValueError("NewJobArrival event missing attached job instance")
            self._add_job(job)
            self._rebuild_matrices()
            return (
                "NEW_JOB",
                {"time": float(event.time), "job_id": int(job.job_id), "num_ops": len(job.operations)},
            )
        elif isinstance(event, MachineBreakdown):
            return (
                "MACHINE_BREAKDOWN",
                {
                    "time": float(event.time),
                    "machine_id": int(event.machine_id),
                    "repair_duration": float(event.repair_duration),
                },
            )
        elif isinstance(event, PriorityChange):
            if event.job_id in self.jobs:
                self.jobs[event.job_id].weight = int(event.new_weight)
            return (
                "PRIORITY_CHANGE",
                {"time": float(event.time), "job_id": int(event.job_id), "new_weight": int(event.new_weight)},
            )
        else:
            raise ValueError(f"Unsupported event: {event}")


