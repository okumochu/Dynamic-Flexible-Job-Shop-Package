from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from benchmarks.static_benchmark.jobshop_components import Job
from .data_handler import DynamicFlexibleJobShopDataHandler
from .state import DynamicState
from .events import NewJobArrival, MachineBreakdown, PriorityChange


class DynamicRLEnv(gym.Env):
    """Dynamic FJSP environment with exogenous events.

    Events: new job arrivals, machine breakdowns (non-preemptive),
    and priority (weight) changes.
    """

    def __init__(
        self,
        data_handler: DynamicFlexibleJobShopDataHandler,
        *,
        alpha: float,
        use_reward_shaping: bool = True,
        max_jobs: Optional[int] = None,
        max_operations: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.handler = data_handler
        self.alpha = float(alpha)
        self.use_reward_shaping = bool(use_reward_shaping)

        self.state = DynamicState(self.handler, max_jobs=max_jobs, max_operations=max_operations)
        self.jobs = self.state.jobs
        self.num_jobs = self.state.num_jobs
        self.num_machines = self.state.num_machines
        self.due_dates = self.state.due_dates
        self.weights = self.state.weights

        self.action_dim = self.state.job_dim * self.state.machine_dim

        self.state.reset()
        self.obs_len = len(self.state._to_numpy())

        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_len,), dtype=np.float32)

        self.current_time: float = 0.0
        self.last_step_objective: float = 0.0
        self.last_episode_objective: float = 0.0
        self.machine_schedule: Dict[int, List[Tuple[int, float]]] = {m: [] for m in range(self.num_machines)}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        self.state.reset()
        self.current_time = 0.0
        self.last_step_objective = 0.0
        self.machine_schedule = {m: [] for m in range(self.num_machines)}
        self._apply_pending_events_up_to(self.current_time)
        obs = self.state._to_numpy()
        return obs, {}

    def _apply_pending_events_up_to(self, t: float) -> List[Dict[str, Any]]:
        info_events: List[Dict[str, Any]] = []
        events = self.handler.pop_events_up_to(t)
        for ev in events:
            etype, details = self.handler.apply_event(ev)
            info_events.append({"type": etype, **details})
            if etype == "NEW_JOB":
                job: Job = getattr(ev, "job")  # type: ignore[attr-defined]
                self.state.add_job(job)
                # refresh references
                self.jobs = self.state.jobs
            elif etype == "MACHINE_BREAKDOWN":
                down_until = details["time"] + details["repair_duration"]
                self.state.apply_machine_breakdown(details["machine_id"], down_until)
            elif etype == "PRIORITY_CHANGE":
                self.state.apply_priority_change(details["job_id"], details["new_weight"])
        return info_events

    def decode_action(self, action: int) -> Tuple[int, int]:
        job_id = action // self.state.machine_dim
        machine_id = action % self.state.machine_dim
        return job_id, machine_id

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_dim, dtype=bool)
        job_states = self.state.readable_state["job_states"]
        for job_id in range(self.state.job_dim):
            js = job_states[job_id]
            if js["left_ops"] <= 0:
                continue
            op_idx = js["current_op"]
            for m in range(self.state.machine_dim):
                if js["operations"]["process_time"][op_idx][m] > 0:
                    # Respect breakdown by ensuring effective availability is finite
                    idx = job_id * self.state.machine_dim + m
                    mask[idx] = True
        return mask

    def _get_current_objective(self) -> Dict[str, float]:
        job_states = self.state.readable_state["job_states"]
        makespan = 0.0
        twt = 0.0
        for job_id in range(self.state.job_dim):
            js = job_states[job_id]
            if js["left_ops"] == 0 and all(v == 0.0 for v in js["operations"]["finish_time"]):
                # padding job
                continue
            job_completion = 0.0
            for ft in js["operations"]["finish_time"]:
                job_completion = max(job_completion, float(ft))
            makespan = max(makespan, job_completion)

            # Use live job data if available; fallback to snapshots
            if job_id in self.state.jobs:
                due = float(self.state.jobs[job_id].due_date)
                weight = float(self.state.jobs[job_id].weight)
            else:
                due = float(self.due_dates[job_id]) if job_id < len(self.due_dates) else float("inf")
                weight = float(self.weights[job_id]) if job_id < len(self.weights) else 1.0
            tardiness = max(0.0, job_completion - due)
            twt += weight * tardiness

        objective = (1.0 - self.alpha) * makespan + self.alpha * twt
        return {"makespan": makespan, "twt": twt, "objective": objective}

    def _get_reward(self) -> float:
        current = self._get_current_objective()["objective"]
        if self.use_reward_shaping:
            reward = self.last_step_objective - current
        else:
            reward = -current if self._is_done() else 0.0
        self.last_step_objective = current
        return reward

    def _is_done(self) -> bool:
        job_states = self.state.readable_state["job_states"]
        return all(job_states[jid]["left_ops"] == 0 for jid in range(self.state.job_dim)) and not self.handler.has_pending_events()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {}
        job_id, machine_id = self.decode_action(action)

        job_states = self.state.readable_state["job_states"]
        js = job_states[job_id]
        op_idx = js["current_op"]
        op = self.jobs[job_id].operations[op_idx]

        # Tentative start time respecting precedence and current machine availability
        tentative_start = float(js["machine_available_time"][machine_id])
        tentative_start = max(tentative_start, float(self.state.machine_down_until[machine_id]))

        # Apply any events that occur before or at tentative start
        info_events: List[Dict[str, Any]] = self._apply_pending_events_up_to(tentative_start)
        if info_events:
            info["applied_events"] = info_events

        # Recompute start time after events
        js = self.state.readable_state["job_states"][job_id]
        start_time = float(js["machine_available_time"][machine_id])
        start_time = max(start_time, float(self.state.machine_down_until[machine_id]))
        proc_time = float(op.get_processing_time(machine_id))
        finish_time = start_time + proc_time

        # Schedule
        self.state.schedule_operation(job_id, machine_id, start_time, finish_time)
        self.machine_schedule[machine_id].append((op.operation_id, start_time))

        # Advance current_time to earliest machine available time across jobs/machines
        all_times = []
        for j in range(self.state.job_dim):
            all_times.extend(self.state.readable_state["job_states"][j]["machine_available_time"])
        self.current_time = float(min(all_times)) if all_times else float(finish_time)

        # Apply any events at or before the new current time
        info_events2 = self._apply_pending_events_up_to(self.current_time)
        if info_events2:
            info.setdefault("applied_events", []).extend(info_events2)

        objective_info = self._get_current_objective()
        reward = self._get_reward()
        obs = self.state._to_numpy()

        terminated = False
        if self._is_done():
            self.last_episode_objective = objective_info["objective"]
            terminated = True

        info["objective_info"] = objective_info
        info["machine_schedule"] = self.machine_schedule
        return obs, reward, terminated, False, info


