"""
FlatRLEnv: Event-driven Flexible Job Shop Scheduling Environment
Implements OpenAI Gym API for multi-objective optimization (Makespan + TWT)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import gym
from gym import spaces
import heapq

class FlatRLEnv(gym.Env):
    """
    Event-driven Flexible Job Shop Scheduling Environment.
    
    The environment simulates a flexible job shop where:
    - Each job has multiple operations that must be processed in order
    - Each operation can be processed on multiple compatible machines
    - The agent dispatches operations to machines
    - Time advances to the next completion event after each dispatch
    - Objectives: minimize makespan and total weighted tardiness
    """
    
    def __init__(self, data_handler, alpha: float = 0.5, max_jobs: Optional[int] = None, max_machines: Optional[int] = None):
        """
        Initialize the environment.
        
        Args:
            data_handler: FlexibleJobShopDataHandler instance
            alpha: Weight for makespan in reward (default 0.5)
            max_jobs: Maximum jobs for padding (default: num_jobs)
            max_machines: Maximum machines for padding (default: num_machines)
        """
        super().__init__()
        
        self.data_handler = data_handler
        self.jobs = data_handler.jobs
        self.operations = data_handler.operations
        self.num_jobs = data_handler.num_jobs
        self.num_machines = data_handler.num_machines
        self.num_operations = data_handler.num_operations
        
        # Extract due dates and weights
        self.due_dates = data_handler.get_job_due_dates()
        self.weights = data_handler.get_job_weights()
        
        self.alpha = alpha
        self.beta = 1 - alpha
        
        # Set padding dimensions
        self.max_jobs = max_jobs if max_jobs is not None else self.num_jobs
        self.max_machines = max_machines if max_machines is not None else self.num_machines
        
        # Calculate global statistics for normalization
        self._calculate_global_stats()
        
        # Action space: num_jobs * num_machines
        self.action_dim = self.num_jobs * self.num_machines
        self.action_space = spaces.Discrete(self.action_dim)
        
        # Observation: [proc_times (JxM), machine_avail_time (JxM), job_remain (J), job_weight (J), ops_left (J)]
        self.obs_len = 2 * self.num_jobs * self.num_machines + 3 * self.num_jobs
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_len,), dtype=np.float32)
        
        # Initialize state variables
        self.reset()
    
    def _calculate_global_stats(self):
        """Calculate global statistics for normalization."""
        # Calculate total processing time and average
        total_proc_time = 0
        num_operations = 0
        
        for job in self.jobs:
            for op in job.operations:
                total_proc_time += sum(op.machine_processing_times.values())
                num_operations += len(op.machine_processing_times)
        
        self.avg_processing_time = total_proc_time / num_operations if num_operations > 0 else 1.0
        
        # Calculate max weight for normalization
        self.max_weight = max(self.weights) if self.weights else 1.0
   
    def reset(self) -> torch.Tensor:
        """Reset the environment to initial state."""
        # Initialize job states
        self.job_states = []
        for job_id, job in enumerate(self.jobs):
            job_state = {
                'completed_ops': 0,
                'current_op': 0,
                'remaining_time': sum(op.min_processing_time for op in job.operations),
                'weight': self.weights[job_id],
                'finish_time': None
            }
            self.job_states.append(job_state)
        # Initialize machine states: only track finish_time (when machine is next available)
        self.machine_states = []
        for machine_id in range(self.num_machines):
            machine_state = {
                'finish_time': None  # None means available now
            }
            self.machine_states.append(machine_state)
        # Initialize time and event queue
        self.current_time = 0.0
        # Initialize operation states
        self.op_states = {}  # (job_id, op_idx) -> status
        for job_id, job in enumerate(self.jobs):
            for op_idx in range(len(job.operations)):
                self.op_states[(job_id, op_idx)] = 0  # 0: not started, 1: finished
        # Initialize makespan and TWT tracking
        self.current_makespan = 0.0
        self.current_twt = 0.0
        self.last_objective = None
        # Track full operation schedule for Gantt
        self.operation_schedules = []
        return self._get_observation()
    
    def _get_observation(self) -> torch.Tensor:
        """Get the current observation tensor."""
        # 1. Processing time: [num_jobs, num_machines]
        #    For each job's next operation on each machine (normalized), 0 if not compatible or job done
        proc_times = np.zeros((self.num_jobs, self.num_machines), dtype=np.float32)
        # 2. Machine available time: [num_jobs, num_machines]
        #    For each job and machine, earliest time the machine is available for the job's next op (normalized), -1 if not compatible or job done
        machine_avail_time = np.full((self.num_jobs, self.num_machines), -1.0, dtype=np.float32)
        max_avail_time = 1.0
        for job_id, job in enumerate(self.jobs):
            job_state = self.job_states[job_id]
            op_idx = job_state['current_op']
            if op_idx < len(job.operations):
                op = job.operations[op_idx]
                for machine_id in op.compatible_machines:
                    proc_times[job_id, machine_id] = op.get_processing_time(machine_id) / self.avg_processing_time
                    avail_time = self.machine_states[machine_id]['finish_time']
                    if avail_time is None:
                        avail_time = self.current_time
                    machine_avail_time[job_id, machine_id] = avail_time
                    if avail_time > max_avail_time:
                        max_avail_time = avail_time
        # Normalize machine available times (except -1)
        norm_machine_avail_time = np.where(machine_avail_time >= 0, machine_avail_time / max_avail_time, -1.0)
        # 3. Job remaining time: [num_jobs] (normalized)
        job_remain = np.array([
            js['remaining_time'] / self.avg_processing_time for js in self.job_states
        ], dtype=np.float32)
        # 4. Job weight: [num_jobs] (normalized)
        job_weight = np.array([
            js['weight'] / self.max_weight for js in self.job_states
        ], dtype=np.float32)
        # 5. Operation status: [num_jobs] (number of operations left, normalized)
        max_ops = max(len(job.operations) for job in self.jobs)
        ops_left = np.array([
            (len(self.jobs[jid].operations) - js['completed_ops']) / max_ops for jid, js in enumerate(self.job_states)
        ], dtype=np.float32)
        # Concatenate all state components into a flat vector
        obs = np.concatenate([
            proc_times.flatten(),
            norm_machine_avail_time.flatten(),
            job_remain,
            job_weight,
            ops_left
        ])
        return torch.tensor(obs, dtype=torch.float32)
    
    def _get_action_mask(self) -> torch.Tensor:
        """Get boolean mask for valid actions"""
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        for job_id in range(self.num_jobs):
            job_state = self.job_states[job_id]
            op_idx = job_state['current_op']
            if op_idx >= len(self.jobs[job_id].operations):
                continue  # No more ops
            op = self.jobs[job_id].operations[op_idx]
            for machine_id in op.compatible_machines:
                idx = job_id * self.num_machines + machine_id
                # Always allow scheduling: agent can always schedule on any compatible machine
                mask[idx] = True
        return mask

    def _decode_action(self, action: int) -> Tuple[int, int]:
        """Decode action index to (job_id, machine_id)."""
        job_id = action // self.num_machines
        machine_id = action % self.num_machines
        return job_id, machine_id
    
    
    def _complete_operation(self, job_id: int, op_idx: int, machine_id: int):
        """Complete an operation and update states."""
        # Update operation status
        self.op_states[(job_id, op_idx)] = 1  # Finished
        # No need to update machine idle/current_op/load
        # Update job state
        job_state = self.job_states[job_id]
        op = self.jobs[job_id].operations[op_idx]
        proc_time = op.get_processing_time(machine_id)
        job_state['completed_ops'] += 1
        job_state['remaining_time'] -= proc_time
        # Check if job is finished
        if job_state['completed_ops'] == len(self.jobs[job_id].operations):
            job_state['finish_time'] = self.current_time
            tardiness = max(0, self.current_time - getattr(self.jobs[job_id], 'due_date', 0))
            self.current_twt += job_state['weight'] * tardiness
            self.current_makespan = max(self.current_makespan, self.current_time)
        else:
            job_state['current_op'] = op_idx + 1

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action index (dispatch operations)
            
        Returns:
            observation: Current observation tensor
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        # 1. Check if there are any valid actions (i.e., at least one job with a next op and compatible machine)
        action_mask = self.get_action_mask()
        if not action_mask.any():
            # No valid actions: episode done
            done = True
            reward = 0.0
            info = {
                'makespan': self.current_makespan,
                'twt': self.current_twt,
                'current_time': self.current_time,
                'action_mask': action_mask
            }
            return self._get_observation(), reward, done, info
        # 2. Decode action to (job_id, machine_id)
        job_id, machine_id = self._decode_action(action)
        op_idx = self.job_states[job_id]['current_op']
        op = self.jobs[job_id].operations[op_idx]
        # 3. Schedule the operation: update machine available time, job state, and operation status
        #    The operation starts at the max of current time and machine available time
        machine_ready = self.machine_states[machine_id]['finish_time']
        if machine_ready is None:
            machine_ready = self.current_time
        start_time = max(self.current_time, machine_ready)
        proc_time = op.get_processing_time(machine_id)
        finish_time = start_time + proc_time
        operation_id = op.operation_id
        self.operation_schedules.append({
            'operation_id': operation_id,
            'job_id': job_id,
            'op_idx': op_idx,
            'machine_id': machine_id,
            'start_time': start_time,
            'finish_time': finish_time
        })
        # Update machine available time (finish_time)
        self.machine_states[machine_id]['finish_time'] = finish_time
        # Update operation status
        self.op_states[(job_id, op_idx)] = 1  # Mark as finished
        # Update job state
        job_state = self.job_states[job_id]
        job_state['completed_ops'] += 1
        job_state['remaining_time'] -= proc_time
        if job_state['completed_ops'] == len(self.jobs[job_id].operations):
            job_state['finish_time'] = finish_time
            tardiness = max(0, finish_time - getattr(self.jobs[job_id], 'due_date', 0))
            self.current_twt += job_state['weight'] * tardiness
            self.current_makespan = max(self.current_makespan, finish_time)
        else:
            job_state['current_op'] = op_idx + 1
        # 4. Advance current time to the finish time of this operation
        self.current_time = finish_time
        # 5. Check if all jobs are finished
        done = all(js['completed_ops'] == len(self.jobs[jid].operations) for jid, js in enumerate(self.job_states))
        if done:
            objective = self.alpha * self.current_makespan + self.beta * self.current_twt
            reward = 0.0
            if self.last_objective is not None:
                reward = self.last_objective - objective
            self.last_objective = objective
        else:
            reward = 0.0
        info = {
            'makespan': self.current_makespan,
            'twt': self.current_twt,
            'current_time': self.current_time,
            'action_mask': self.get_action_mask()
        }
        return self._get_observation(), reward, done, info
    
    def get_action_mask(self) -> torch.Tensor:
        """Get the current action mask."""
        return self._get_action_mask()
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """Get detailed schedule information for visualization."""
        schedule = {
            'job_completion_times': {},
            'machine_schedules': {},
            'makespan': self.current_makespan,
            'total_weighted_tardiness': self.current_twt,
            'operation_schedules': self.operation_schedules
        }
        # Job completion times
        for job_id, job_state in enumerate(self.job_states):
            if job_state['finish_time'] is not None:
                schedule['job_completion_times'][job_id] = job_state['finish_time']
        # Machine schedules (only finish_time is tracked)
        for machine_id, machine_state in enumerate(self.machine_states):
            schedule['machine_schedules'][machine_id] = {
                'finish_time': machine_state['finish_time']
            }
        return schedule 