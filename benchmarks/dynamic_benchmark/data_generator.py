from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from benchmarks.static_benchmark.jobshop_components import Job, Operation
from benchmarks.static_benchmark.data_generator import DataGenerator as StaticGenerator
from .events import Event, EventType, NewJobArrival, MachineBreakdown, PriorityChange


class DynamicScenarioGenerator:
    """Generate dynamic FJSP scenarios with exogenous events.

    Events supported:
    - New job arrivals
    - Machine breakdowns (non-preemptive; block new starts during downtime)
    - Priority changes (job weight updates)
    """

    @staticmethod
    def _generate_single_job(
        job_id: int,
        num_machines: int,
        operation_lb: int,
        operation_ub: int,
        processing_time_lb: int,
        processing_time_ub: int,
        compatible_machines_lb: int,
        compatible_machines_ub: Optional[int] = None,
        TF: float = 0.4,
        RDD: float = 0.8,
        seed: Optional[int] = None,
    ) -> Job:
        """Create a single random job consistent with static generator semantics."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if compatible_machines_ub is None:
            compatible_machines_ub = num_machines

        if operation_lb < 1 or operation_ub < operation_lb:
            raise ValueError("Invalid operation bounds")
        if processing_time_lb < 1 or processing_time_ub < processing_time_lb:
            raise ValueError("Invalid processing time bounds")
        if compatible_machines_lb < 1 or compatible_machines_ub > num_machines:
            raise ValueError("Invalid compatible machines bounds")
        if TF < 0 or TF > 1 or RDD < 0 or RDD > 1:
            raise ValueError("TF and RDD must be between 0 and 1")

        # Build operations list and aggregate processing time for due date generation
        operations: List[Operation] = []
        total_processing_time = 0
        num_operations = random.randint(operation_lb, operation_ub)
        next_operation_global_id = 0  # local temporary; caller should reindex when integrating

        for _ in range(num_operations):
            num_compatible = random.randint(compatible_machines_lb, compatible_machines_ub)
            compatible_machines = random.sample(range(num_machines), num_compatible)
            machine_processing_times: Dict[int, int] = {}
            min_pt = float("inf")
            for m in compatible_machines:
                pt = random.randint(processing_time_lb, processing_time_ub)
                machine_processing_times[m] = pt
                min_pt = min(min_pt, pt)

            op = Operation(
                operation_id=next_operation_global_id,
                job_id=job_id,
                machine_processing_times=machine_processing_times,
            )
            next_operation_global_id += 1
            operations.append(op)
            total_processing_time += int(min_pt)

        # Use static due date/weight rule for consistency
        due_dates, weights = StaticGenerator._generate_due_dates_and_weights(
            [total_processing_time], TF, RDD, seed
        )
        job = Job(job_id=job_id, operations=operations, due_date=due_dates[0], weight=weights[0])
        return job

    @staticmethod
    def generate_scenario(
        *,
        num_machines: int,
        # initial jobs
        num_initial_jobs: int,
        operation_lb: int,
        operation_ub: int,
        processing_time_lb: int,
        processing_time_ub: int,
        compatible_machines_lb: int,
        compatible_machines_ub: Optional[int] = None,
        # dynamics
        time_horizon: float,
        arrival_rate: float = 0.0,
        breakdown_rate: float = 0.0,
        breakdown_repair_time_lb: float = 1.0,
        breakdown_repair_time_ub: float = 5.0,
        priority_change_rate: float = 0.0,
        priority_min_weight: int = 1,
        priority_max_weight: int = 10,
        TF: float = 0.4,
        RDD: float = 0.8,
        seed: Optional[int] = None,
    ) -> Tuple[List[Job], List[Event]]:
        """Generate an initial set of jobs and an event list within a time horizon.

        Returns:
            (initial_jobs, events) where events is sorted by time.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        events: List[Event] = []

        # Initial jobs at t=0
        initial_jobs: List[Job] = []
        for j in range(num_initial_jobs):
            job = DynamicScenarioGenerator._generate_single_job(
                job_id=j,
                num_machines=num_machines,
                operation_lb=operation_lb,
                operation_ub=operation_ub,
                processing_time_lb=processing_time_lb,
                processing_time_ub=processing_time_ub,
                compatible_machines_lb=compatible_machines_lb,
                compatible_machines_ub=compatible_machines_ub,
                TF=TF,
                RDD=RDD,
                seed=None if seed is None else seed + j,
            )
            # Reindex operation ids globally after integration; here just placeholders
            initial_jobs.append(job)

        # New job arrivals: Poisson process approximation with homogeneous rate
        if arrival_rate > 0:
            t = 0.0
            next_job_id = num_initial_jobs
            while True:
                # Exponential inter-arrival
                delta = np.random.exponential(1.0 / arrival_rate)
                t += float(delta)
                if t > time_horizon:
                    break

                job = DynamicScenarioGenerator._generate_single_job(
                    job_id=next_job_id,
                    num_machines=num_machines,
                    operation_lb=operation_lb,
                    operation_ub=operation_ub,
                    processing_time_lb=processing_time_lb,
                    processing_time_ub=processing_time_ub,
                    compatible_machines_lb=compatible_machines_lb,
                    compatible_machines_ub=compatible_machines_ub,
                    TF=TF,
                    RDD=RDD,
                    seed=None if seed is None else seed + next_job_id,
                )
                next_job_id += 1
                events.append(NewJobArrival(time=t, job_id=job.job_id))
                # Attach job object on the event by attribute injection to avoid cyclic typing
                # Consumers should check for `job` attribute
                setattr(events[-1], "job", job)

        # Machine breakdowns as Poisson per machine
        if breakdown_rate > 0:
            for m in range(num_machines):
                t = 0.0
                while True:
                    delta = np.random.exponential(1.0 / breakdown_rate)
                    t += float(delta)
                    if t > time_horizon:
                        break
                    repair_duration = float(
                        np.random.uniform(breakdown_repair_time_lb, breakdown_repair_time_ub)
                    )
                    events.append(
                        MachineBreakdown(time=t, machine_id=m, repair_duration=repair_duration)
                    )

        # Priority changes as Poisson per job space (apply to existing or future jobs by id)
        if priority_change_rate > 0:
            t = 0.0
            while True:
                delta = np.random.exponential(1.0 / priority_change_rate)
                t += float(delta)
                if t > time_horizon:
                    break
                # Randomly pick a job id among current + expected arrivals
                candidate_job_id = int(
                    np.random.randint(0, max(1, len(initial_jobs) + int(arrival_rate * time_horizon)))
                )
                new_weight = int(np.random.randint(priority_min_weight, priority_max_weight + 1))
                events.append(PriorityChange(time=t, job_id=candidate_job_id, new_weight=new_weight))

        # Sort events by time, and stabilize order by type to ensure deterministic application
        type_order = {EventType.NEW_JOB: 0, EventType.MACHINE_BREAKDOWN: 1, EventType.PRIORITY_CHANGE: 2}

        def sort_key(e: Event) -> Tuple[float, int]:
            if isinstance(e, NewJobArrival):
                et = EventType.NEW_JOB
            elif isinstance(e, MachineBreakdown):
                et = EventType.MACHINE_BREAKDOWN
            else:
                et = EventType.PRIORITY_CHANGE
            return (float(getattr(e, "time", 0.0)), type_order[et])

        events.sort(key=sort_key)
        return initial_jobs, events


