from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from benchmarks.static_benchmark.jobshop_components import Job, Operation
from .data_handler import DynamicFlexibleJobShopDataHandler


class DynamicState:
    """State manager for dynamic FJSP environment.

    Observation layout mirrors the static state with additional handling for
    machine downtimes and job arrivals.
    """

    def __init__(
        self,
        data_handler: DynamicFlexibleJobShopDataHandler,
        max_jobs: Optional[int] = None,
        max_operations: Optional[int] = None,
    ) -> None:
        self.data_handler = data_handler
        self.jobs: Dict[int, Job] = data_handler.jobs
        self.num_jobs: int = data_handler.num_jobs
        self.num_machines: int = data_handler.num_machines

        self.due_dates: List[int] = data_handler.get_jobs_due_date()
        self.weights: List[int] = data_handler.get_jobs_weight()

        self.job_dim: int = max_jobs if max_jobs is not None else max(1, self.num_jobs)
        self.machine_dim: int = self.num_machines
        self.operation_dim: int = max_operations if max_operations is not None else (
            max((len(job.operations) for job in self.jobs.values()), default=1)
        )

        # Global normalization statistics
        self.max_processing_time: int = max(1, data_handler.get_max_processing_time())
        self.max_weight: int = max(1, data_handler.get_max_weight())
        self.max_due_date: int = max(1, data_handler.get_max_due_date())
        self.max_machine_available_time: float = 1.0
        self.max_operation_start_time: float = 1.0

        # Downtime horizon per machine
        self.machine_down_until: List[float] = [0.0 for _ in range(self.machine_dim)]

        self.obs_dim, self.action_dim = self.reset()

    def _to_numpy(self) -> np.ndarray:
        job_states = self.readable_state["job_states"]
        obs: List[float] = []

        for job_id in range(self.job_dim):
            v = job_states[job_id]
            obs.append(v["left_ops"] / max(1, self.operation_dim))
            obs.append(v["weight"] / max(1, self.max_weight))
            obs.append(v["due_date"] / max(1, self.max_due_date))

            available_times = np.array(v["machine_available_time"], dtype=np.float32)
            # Incorporate downtime by taking the effective availability at observation time
            effective = np.maximum(available_times, np.array(self.machine_down_until, dtype=np.float32))
            obs.extend(list(effective / max(1e-6, self.max_machine_available_time)))

            for op_pos in range(self.operation_dim):
                pt = np.array(v["operations"]["process_time"][op_pos], dtype=np.float32)
                obs.extend(list(pt / max(1, self.max_processing_time)))

                st = np.array(v["operations"]["operation_start_time"][op_pos], dtype=np.float32)
                obs.extend(list(st / max(1e-6, self.max_operation_start_time)))

        return np.array(obs, dtype=np.float32)

    def reset(self) -> Tuple[int, int]:
        job_states: Dict[int, Dict] = {}
        for job_id in range(self.job_dim):
            padding = job_id not in self.jobs
            num_ops = 0 if padding else len(self.jobs[job_id].operations)
            effective_ops = min(num_ops, self.operation_dim)
            job_states[job_id] = {
                "left_ops": 0 if padding else len(self.jobs[job_id].operations),
                "current_op": 0,
                "weight": 0.0 if padding else float(self.weights[job_id]) if job_id < len(self.weights) else float(self.jobs[job_id].weight),
                "due_date": 0.0 if padding else float(self.due_dates[job_id]) if job_id < len(self.due_dates) else float(self.jobs[job_id].due_date),
                "operations": {
                    "process_time": [],
                    "operation_start_time": [[0.0] * self.machine_dim for _ in range(self.operation_dim)],
                    "finish_time": [0.0] * self.operation_dim,
                },
                "machine_available_time": [0.0] * self.machine_dim,
                "truncated": (not padding) and (num_ops > self.operation_dim),
            }

            if padding:
                job_states[job_id]["operations"]["process_time"] = [
                    [0.0] * self.machine_dim for _ in range(self.operation_dim)
                ]
            else:
                for op_pos in range(self.operation_dim):
                    if op_pos < effective_ops:
                        op = self.jobs[job_id].operations[op_pos]
                        row: List[float] = []
                        for m in range(self.machine_dim):
                            row.append(float(op.get_processing_time(m)) if m in op.compatible_machines else 0.0)
                        job_states[job_id]["operations"]["process_time"].append(row)
                    else:
                        job_states[job_id]["operations"]["process_time"].append([0.0] * self.machine_dim)
                # clamp left_ops to effective_ops to avoid index errors
                job_states[job_id]["left_ops"] = effective_ops

        self.max_machine_available_time = 1.0
        self.max_operation_start_time = 1.0
        self.machine_down_until = [0.0 for _ in range(self.machine_dim)]

        self.readable_state = {"job_states": job_states}
        obs_dim = len(self._to_numpy())
        action_dim = self.job_dim * self.machine_dim
        return obs_dim, action_dim

    # ===== Scheduling and event application =====
    def schedule_operation(self, job_id: int, machine_id: int, start_time: float, finish_time: float) -> None:
        job_states = self.readable_state["job_states"]
        js = job_states[job_id]
        op_pos = js["current_op"]

        js["operations"]["operation_start_time"][op_pos][machine_id] = float(start_time)
        self.max_operation_start_time = max(self.max_operation_start_time, float(start_time))

        js["operations"]["finish_time"][op_pos] = float(finish_time)
        js["current_op"] += 1
        js["left_ops"] -= 1

        # Update machine availability across jobs
        for j_idx in range(self.job_dim):
            job_state = job_states[j_idx]
            if job_state["left_ops"] > 0:
                if job_state["machine_available_time"][machine_id] < float(finish_time):
                    job_state["machine_available_time"][machine_id] = float(finish_time)
                    self.max_machine_available_time = max(self.max_machine_available_time, float(finish_time))

        # Update job's own machine availability to enforce precedence
        if js["left_ops"] > 0:
            for m in range(self.machine_dim):
                js["machine_available_time"][m] = max(js["machine_available_time"][m], float(finish_time))

    def add_job(self, job: Job) -> None:
        # Expand capacity if within bounds
        if job.job_id >= self.job_dim:
            # Ignore if exceeds capacity; caller can enforce max_jobs
            return
        self.jobs[job.job_id] = job
        # Update stats
        self.num_jobs = max(self.num_jobs, len(self.jobs))
        self.due_dates = [self.jobs[jid].due_date for jid in range(self.job_dim) if jid in self.jobs]
        self.weights = [self.jobs[jid].weight for jid in range(self.job_dim) if jid in self.jobs]
        self.max_processing_time = max(self.max_processing_time, max((op.max_processing_time for op in job.operations), default=1))
        self.max_weight = max(self.max_weight, int(job.weight))
        self.max_due_date = max(self.max_due_date, int(job.due_date))

        # Fill job state structure
        effective_ops = min(len(job.operations), self.operation_dim)
        js = {
            "left_ops": effective_ops,
            "current_op": 0,
            "weight": float(job.weight),
            "due_date": float(job.due_date),
            "operations": {
                "process_time": [],
                "operation_start_time": [[0.0] * self.machine_dim for _ in range(self.operation_dim)],
                "finish_time": [0.0] * self.operation_dim,
            },
            "machine_available_time": [0.0] * self.machine_dim,
            "truncated": len(job.operations) > self.operation_dim,
        }
        for op_pos in range(self.operation_dim):
            if op_pos < effective_ops:
                op = job.operations[op_pos]
                row: List[float] = []
                for m in range(self.machine_dim):
                    row.append(float(op.get_processing_time(m)) if m in op.compatible_machines else 0.0)
                js["operations"]["process_time"].append(row)
            else:
                js["operations"]["process_time"].append([0.0] * self.machine_dim)

        self.readable_state["job_states"][job.job_id] = js

    def apply_priority_change(self, job_id: int, new_weight: int) -> None:
        if job_id in self.readable_state["job_states"]:
            self.readable_state["job_states"][job_id]["weight"] = float(new_weight)
        if job_id in self.jobs:
            self.jobs[job_id].weight = int(new_weight)
        self.max_weight = max(self.max_weight, int(new_weight))

    def apply_machine_breakdown(self, machine_id: int, down_until: float) -> None:
        self.machine_down_until[machine_id] = max(self.machine_down_until[machine_id], float(down_until))
        # Push availability beyond downtime for all jobs
        for j in range(self.job_dim):
            js = self.readable_state["job_states"][j]
            if js["machine_available_time"][machine_id] < self.machine_down_until[machine_id]:
                js["machine_available_time"][machine_id] = self.machine_down_until[machine_id]
                self.max_machine_available_time = max(self.max_machine_available_time, js["machine_available_time"][machine_id])


