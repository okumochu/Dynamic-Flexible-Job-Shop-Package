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
    
    def __init__(self, 
                 data_handler,
                 alpha: float = 0.3, 
                 beta: float = 0.7,
                 max_jobs: Optional[int] = None,
                 max_machines: Optional[int] = None):
        """
        Initialize the environment.
        
        Args:
            data_handler: FlexibleJobShopDataHandler instance
            alpha: Weight for makespan in reward (default 0.3)
            beta: Weight for TWT in reward (default 0.7)
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
        self.beta = beta
        
        # Set padding dimensions
        self.max_jobs = max_jobs if max_jobs is not None else self.num_jobs
        self.max_machines = max_machines if max_machines is not None else self.num_machines
        
        # Calculate global statistics for normalization
        self._calculate_global_stats()
        
        # Action space: (max_jobs * max_ops_per_job) * max_machines (no do-nothing)
        max_ops_per_job = max(len(job.operations) for job in self.jobs)
        self.action_dim = self.max_jobs * max_ops_per_job * self.max_machines
        self.action_space = spaces.Discrete(self.action_dim)
        
        # Observation space: (max_jobs, max_machines, 8)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.max_jobs, self.max_machines, 8), dtype=np.float32
        )
        
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
        
        # Calculate total weighted processing time for TWT normalization
        self.total_weighted_proc_time = sum(
            weight * sum(sum(op.machine_processing_times.values()) for op in job.operations)
            for job, weight in zip(self.jobs, self.weights)
        )
    
    def reset(self) -> torch.Tensor:
        """Reset the environment to initial state."""
        # Initialize job states
        self.job_states = []
        for job_id, job in enumerate(self.jobs):
            job_state = {
                'completed_ops': 0,
                'current_op': 0,
                'remaining_time': sum(op.min_processing_time for op in job.operations),
                'due_date': self.due_dates[job_id],
                'weight': self.weights[job_id],
                'finish_time': None
            }
            self.job_states.append(job_state)
        
        # Initialize machine states
        self.machine_states = []
        for machine_id in range(self.num_machines):
            machine_state = {
                'idle': True,
                'current_op': None,
                'finish_time': None,
                'queue_load': 0.0
            }
            self.machine_states.append(machine_state)
        
        # Initialize time and event queue
        self.current_time = 0.0
        self.event_queue = []  # Priority queue of (finish_time, machine_id, op_id, job_id)
        
        # Initialize operation states
        self.op_states = {}  # (job_id, op_idx) -> status
        for job_id, job in enumerate(self.jobs):
            for op_idx in range(len(job.operations)):
                self.op_states[(job_id, op_idx)] = 0  # 0: not started, 0.5: in process, 1: finished
        
        # Initialize makespan and TWT tracking
        self.current_makespan = 0.0
        self.current_twt = 0.0
        self.previous_makespan = 0.0
        self.previous_twt = 0.0
        
        return self._get_observation()
    
    def _get_observation(self) -> torch.Tensor:
        """Get the current observation tensor."""
        obs = np.zeros((self.max_jobs, self.max_machines, 8), dtype=np.float32)
        
        for job_id in range(len(self.jobs)):
            if job_id >= self.max_jobs:
                break
                
            job = self.jobs[job_id]
            job_state = self.job_states[job_id]
            
            for op_idx in range(len(job.operations)):
                if op_idx >= len(job.operations):  # Padding
                    obs[job_id, :, 0] = -1  # Invalid machine
                    continue
                
                op = job.operations[op_idx]
                
                for machine_idx, machine_id in enumerate(op.compatible_machines):
                    if machine_idx >= self.max_machines:
                        break
                    
                    proc_time = op.get_processing_time(machine_id)
                    
                    # Channel 0: Processing time (normalized)
                    obs[job_id, machine_idx, 0] = proc_time / self.avg_processing_time
                    
                    # Channel 1: Ready flag (1 if next operation and predecessors finished)
                    is_next_op = (op_idx == job_state['current_op'])
                    predecessors_finished = all(
                        self.op_states.get((job_id, prev_op), 1) == 1 
                        for prev_op in range(op_idx)
                    )
                    obs[job_id, machine_idx, 1] = 1.0 if (is_next_op and predecessors_finished) else 0.0
                    
                    # Channel 2: Machine idle status
                    obs[job_id, machine_idx, 2] = 1.0 if self.machine_states[machine_id]['idle'] else 0.0
                    
                    # Channel 3: Queue load on machine (normalized)
                    obs[job_id, machine_idx, 3] = self.machine_states[machine_id]['queue_load'] / self.avg_processing_time
                    
                    # Channel 4: Job remaining time (normalized)
                    obs[job_id, machine_idx, 4] = job_state['remaining_time'] / self.avg_processing_time
                    
                    # Channel 5: Due slack (normalized)
                    due_slack = job_state['due_date'] - self.current_time
                    obs[job_id, machine_idx, 5] = due_slack / self.avg_processing_time
                    
                    # Channel 6: Weight (normalized)
                    obs[job_id, machine_idx, 6] = job_state['weight'] / self.max_weight
                    
                    # Channel 7: Operation status
                    obs[job_id, machine_idx, 7] = self.op_states.get((job_id, op_idx), 0)
                
                # Fill remaining machines with invalid values
                for machine_idx in range(len(op.compatible_machines), self.max_machines):
                    obs[job_id, machine_idx, 0] = -1
        
        # Fill remaining jobs with zeros and invalid machines
        for job_id in range(len(self.jobs), self.max_jobs):
            obs[job_id, :, 0] = -1
        
        return torch.tensor(obs, dtype=torch.float32)
    
    def _get_action_mask(self) -> torch.Tensor:
        """Get boolean mask for valid actions (no do-nothing)."""
        mask = torch.zeros(self.action_dim, dtype=torch.bool)
        action_idx = 0
        max_ops_per_job = max(len(job.operations) for job in self.jobs)
        for job_id in range(self.max_jobs):
            if job_id >= len(self.jobs):
                action_idx += max_ops_per_job * self.max_machines
                continue
            job = self.jobs[job_id]
            job_state = self.job_states[job_id]
            for op_idx in range(max_ops_per_job):
                if op_idx >= len(job.operations):
                    action_idx += self.max_machines
                    continue
                op = job.operations[op_idx]
                is_next_op = (op_idx == job_state['current_op'])
                predecessors_finished = all(
                    self.op_states.get((job_id, prev_op), 1) == 1 
                    for prev_op in range(op_idx)
                )
                is_ready = is_next_op and predecessors_finished
                for machine_idx in range(self.max_machines):
                    if machine_idx < len(op.compatible_machines):
                        machine_id = op.compatible_machines[machine_idx]
                        can_process = True
                    else:
                        can_process = False
                    is_valid = (
                        is_ready and
                        can_process and
                        self.machine_states[machine_id]['idle'] if can_process else False
                    )
                    mask[action_idx] = is_valid
                    action_idx += 1
        return mask

    def _decode_action(self, action: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Decode action index to (job_id, op_idx, machine_id)."""
        action_idx = action
        max_ops_per_job = max(len(job.operations) for job in self.jobs)
        job_id = action_idx // (max_ops_per_job * self.max_machines)
        remaining = action_idx % (max_ops_per_job * self.max_machines)
        op_idx = remaining // self.max_machines
        machine_idx = remaining % self.max_machines
        if job_id >= len(self.jobs) or op_idx >= len(self.jobs[job_id].operations):
            return None, None, None
        op = self.jobs[job_id].operations[op_idx]
        if machine_idx >= len(op.compatible_machines):
            return None, None, None
        machine_id = op.compatible_machines[machine_idx]
        return job_id, op_idx, machine_id
    
    def _advance_time(self):
        """Advance time to the next completion event."""
        if not self.event_queue:
            # No events scheduled, advance by a small amount
            self.current_time += 1.0
            return
        
        # Get the next completion event
        finish_time, machine_id, op_id, job_id = heapq.heappop(self.event_queue)
        self.current_time = finish_time
        
        # Complete the operation
        self._complete_operation(job_id, op_id, machine_id)
        
        # Process any other operations that finish at the same time
        while self.event_queue and self.event_queue[0][0] == finish_time:
            next_finish_time, next_machine_id, next_op_id, next_job_id = heapq.heappop(self.event_queue)
            self._complete_operation(next_job_id, next_op_id, next_machine_id)
    
    def _complete_operation(self, job_id: int, op_idx: int, machine_id: int):
        """Complete an operation and update states."""
        # Update operation status
        self.op_states[(job_id, op_idx)] = 1  # Finished
        
        # Update machine state
        self.machine_states[machine_id]['idle'] = True
        self.machine_states[machine_id]['current_op'] = None
        self.machine_states[machine_id]['finish_time'] = None
        
        # Update job state
        job_state = self.job_states[job_id]
        op = self.jobs[job_id].operations[op_idx]
        proc_time = op.get_processing_time(machine_id)
        
        job_state['completed_ops'] += 1
        job_state['remaining_time'] -= proc_time
        
        # Check if job is finished
        if job_state['completed_ops'] == len(self.jobs[job_id].operations):
            job_state['finish_time'] = self.current_time
            # Calculate tardiness
            tardiness = max(0, self.current_time - job_state['due_date'])
            self.current_twt += job_state['weight'] * tardiness
            self.current_makespan = max(self.current_makespan, self.current_time)
        else:
            # Move to next operation
            job_state['current_op'] = op_idx + 1
    
    def _assign_operation(self, job_id: int, op_idx: int, machine_id: int):
        """Assign an operation to a machine."""
        op = self.jobs[job_id].operations[op_idx]
        proc_time = op.get_processing_time(machine_id)
        
        # Update operation status
        self.op_states[(job_id, op_idx)] = 0.5  # In process
        
        # Update machine state
        finish_time = self.current_time + proc_time
        self.machine_states[machine_id]['idle'] = False
        self.machine_states[machine_id]['current_op'] = (job_id, op_idx)
        self.machine_states[machine_id]['finish_time'] = finish_time
        self.machine_states[machine_id]['queue_load'] += proc_time
        
        # Schedule completion event
        heapq.heappush(self.event_queue, (finish_time, machine_id, op_idx, job_id))
    
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
        self.previous_makespan = self.current_makespan
        self.previous_twt = self.current_twt
        job_id, op_idx, machine_id = self._decode_action(action)
        if job_id is None or op_idx is None or machine_id is None:
            raise ValueError(f"Invalid action: {action}")
        # Dispatch operation
        self._assign_operation(job_id, op_idx, machine_id)
        self._advance_time()
        done = all(job_state['completed_ops'] == len(job.operations) 
                  for job, job_state in zip(self.jobs, self.job_states))
        reward = self._calculate_reward()
        action_mask = self._get_action_mask()
        if not action_mask.any():
            reward = -1.0
            done = True
        info = {
            'makespan': self.current_makespan,
            'twt': self.current_twt,
            'current_time': self.current_time,
            'action_mask': action_mask
        }
        return self._get_observation(), reward, done, info
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on makespan and TWT changes."""
        # Calculate changes
        delta_makespan = self.current_makespan - self.previous_makespan
        delta_twt = self.current_twt - self.previous_twt
        
        # Calculate average cycle time
        avg_cycle = self.avg_processing_time
        
        # Calculate reward components
        makespan_component = delta_makespan / avg_cycle if avg_cycle > 0 else 0
        twt_component = delta_twt / self.total_weighted_proc_time if self.total_weighted_proc_time > 0 else 0
        
        # Combine with weights
        reward = -(self.alpha * makespan_component + self.beta * twt_component)
        
        return reward
    
    def get_action_mask(self) -> torch.Tensor:
        """Get the current action mask."""
        return self._get_action_mask()
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """Get detailed schedule information for visualization."""
        schedule = {
            'job_completion_times': {},
            'machine_schedules': {},
            'makespan': self.current_makespan,
            'total_weighted_tardiness': self.current_twt
        }
        
        # Job completion times
        for job_id, job_state in enumerate(self.job_states):
            if job_state['finish_time'] is not None:
                schedule['job_completion_times'][job_id] = job_state['finish_time']
        
        # Machine schedules (simplified - would need more detailed tracking for full Gantt)
        for machine_id, machine_state in enumerate(self.machine_states):
            schedule['machine_schedules'][machine_id] = {
                'idle': machine_state['idle'],
                'current_operation': machine_state['current_op'],
                'finish_time': machine_state['finish_time']
            }
        
        return schedule 