"""
FlatRLState: State management class for Flat RL Environment
Handles state representation, action masking, and observation generation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler

class FlatRLState:
    """
    State management class for Flat RL Environment.
    
    This class encapsulates all state-related functionality:
    - State dimensions and space definitions
    - Action masking logic
    - Observation generation
    - State initialization and updates
    - Global statistics for normalization
    """
    
    def __init__(self, data_handler: FlexibleJobShopDataHandler, 
                 max_jobs: Optional[int] = None, 
                 max_machines: Optional[int] = None,
                 max_operations: Optional[int] = None):
        """
        Initialize the state manager.
        
        Args:
            data_handler: FlexibleJobShopDataHandler instance
            max_jobs: Maximum jobs for padding (default: num_jobs)
            max_machines: Maximum machines for padding (default: num_machines)
        """
        self.data_handler = data_handler
        self.jobs = data_handler.jobs
        self.operations = data_handler.operations
        self.num_jobs = data_handler.num_jobs
        self.num_machines = data_handler.num_machines
        self.num_operations = data_handler.num_operations
        
        # Extract due dates and weights
        self.due_dates = data_handler.get_jobs_due_date()
        self.weights = data_handler.get_jobs_weight()
        
        # Set padding dimensions
        self.max_jobs = max_jobs if max_jobs is not None else self.num_jobs
        self.max_machines = max_machines if max_machines is not None else self.num_machines
        self.max_ops_per_job = max_operations if max_operations is not None else max(len(job.operations) for job in self.jobs)
        
        # Calculate global statistics for normalization
        self.avg_processing_time = data_handler.get_average_processing_time()
        self.max_weight = data_handler.get_max_weight()
        self.max_due_date = data_handler.get_max_due_date()
        
        # State dimensions
        self.action_dim = self.num_jobs * self.num_machines
        # Updated observation length: original + makespan/TWT + all ops processing times + all ops start times
        self.obs_len = (2 * self.num_jobs * self.num_machines + 3 * self.num_jobs +  # Original components
                       2 +  # Makespan and TWT
                       self.max_jobs * self.max_machines * self.max_ops_per_job +  # All ops processing times
                       self.max_jobs * self.max_machines * self.max_ops_per_job)   # All ops start times
        
        # Initialize state variables
        self.reset()
    
    def reset(self):
        """Reset the state to initial values."""
        # Initialize job states
        self.job_states = []
        for job_id, job in enumerate(self.jobs):
            job_state = {
                'completed_ops': 0,
                'current_op': 0,
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
        self.op_states = {}  # (job_id, op_position) -> status
        for job_id, job in enumerate(self.jobs):
            for op_position in range(self.max_ops_per_job):
                self.op_states[(job_id, op_position)] = 0  # 0: not started, 1: finished
        
        # Initialize operation start times
        self.op_start_times = {}  # (job_id, op_position) -> start_time
        for job_id, job in enumerate(self.jobs):
            for op_position in range(self.max_ops_per_job):
                self.op_start_times[(job_id, op_position)] = None  # None means not started
        
        # Initialize makespan and TWT tracking
        self.current_makespan = 0.0
        self.current_twt = 0.0
        
        # Track full operation schedule for Gantt
        self.operation_schedules = []
    
    def get_observation(self) -> torch.Tensor:
        """Get the current observation tensor."""
        # 1. All operations processing times: [num_jobs, num_machines, num_operations]
        all_proc_times = np.zeros((self.num_jobs, self.num_machines, self.num_operations), dtype=np.float16)
        
        # 2. Machine available time: [num_jobs, num_machines]
        machine_avail_time = np.zeros((self.num_jobs, self.num_machines), dtype=np.float16)
        max_avail_time = 1.0

        for job_id, job in enumerate(self.jobs):
            job_state = self.job_states[job_id]
            current_op_position = job_state['current_op']
            num_ops = len(job.operations)

            for op_position in range(self.num_operations):
                if op_position < num_ops:
                    op = job.operations[op_position]
                    op_finished = self.op_states[(job_id, op_position)] == 1
                    for machine_id in range(self.num_machines):
                        if op_finished:
                            # Set all machines to 0 for finished operations
                            all_proc_times[job_id, machine_id, op_position] = 0
                        else:
                            if machine_id in op.compatible_machines:
                                all_proc_times[job_id, machine_id, op_position] = op.get_processing_time(machine_id) / self.avg_processing_time
                            else:
                                all_proc_times[job_id, machine_id, op_position] = 0
                else:
                    # Padding for operations beyond actual job length
                    all_proc_times[job_id, :, op_position] = 0

            # Machine available time for the job's next operation
            if current_op_position >= num_ops:
                # Job is done: set all machines to 0
                machine_avail_time[job_id, :] = 0
            else:
                op = job.operations[current_op_position]
                for machine_id in range(self.num_machines):
                    if machine_id not in op.compatible_machines:
                        machine_avail_time[job_id, machine_id] = -1
                    else:
                        avail_time = self.machine_states[machine_id]['finish_time']
                        if avail_time is None:
                            avail_time = self.current_time
                        machine_avail_time[job_id, machine_id] = avail_time
                        if avail_time > max_avail_time:
                            max_avail_time = avail_time
        
        # Normalize machine available times (except -1 and 0)
        norm_machine_avail_time = np.where(machine_avail_time > 0, machine_avail_time / max_avail_time, machine_avail_time)
        
        # 4. Job weight: [num_jobs] (normalized)
        job_weight = np.array([
            js['weight'] / self.max_weight for js in self.job_states
        ], dtype=np.float16)
        
        # 5. Operation status: [num_jobs] (number of operations left, normalized)
        max_ops = max(len(job.operations) for job in self.jobs)
        ops_left = np.array([
            (len(self.jobs[jid].operations) - js['completed_ops']) / max_ops for jid, js in enumerate(self.job_states)
        ], dtype=np.float16)
        
        # 6. Current makespan and TWT: [2] (normalized)
        # Normalize by the maximum possible values for stability
        max_possible_makespan = sum(sum(op.machine_processing_times.values()) for job in self.jobs for op in job.operations)
        max_possible_twt = sum(weight * max_possible_makespan for weight in self.weights)
        
        makespan_twt = np.array([
            self.current_makespan / max_possible_makespan if max_possible_makespan > 0 else 0.0,
            self.current_twt / max_possible_twt if max_possible_twt > 0 else 0.0
        ], dtype=np.float16)
        
        
        # 8. All operations start times: [max_jobs, max_machines, max_ops_per_job]
        all_start_times = np.zeros((self.max_jobs, self.max_machines, self.max_ops_per_job), dtype=np.float16)
        max_start_time = 1.0  # For normalization
        
        for job_id, job in enumerate(self.jobs):
            for op_position, op in enumerate(job.operations):
                if op_position < self.max_ops_per_job:  # Ensure we don't exceed padding
                    start_time = self.op_start_times[(job_id, op_position)]
                    if start_time is not None:  # Operation has been started
                        if start_time > max_start_time:
                            max_start_time = start_time
                        for machine_id in range(self.max_machines):
                            if machine_id < self.num_machines:
                                if machine_id in op.compatible_machines:
                                    all_start_times[job_id, machine_id, op_position] = start_time
                                else:
                                    all_start_times[job_id, machine_id, op_position] = -1
                            else:
                                # Padding: set to 0
                                all_start_times[job_id, machine_id, op_position] = 0
                    else:  # Operation not started
                        for machine_id in range(self.max_machines):
                            if machine_id < self.num_machines:
                                if machine_id in op.compatible_machines:
                                    all_start_times[job_id, machine_id, op_position] = 0  # Not started yet
                                else:
                                    all_start_times[job_id, machine_id, op_position] = -1
                            else:
                                # Padding: set to 0
                                all_start_times[job_id, machine_id, op_position] = 0
            # Fill remaining operations with 0 (padding)
            for op_position in range(len(job.operations), self.max_ops_per_job):
                all_start_times[job_id, :, op_position] = 0
        
        # Normalize start times (except -1 and 0)
        norm_all_start_times = np.where(all_start_times > 0, all_start_times / max_start_time, all_start_times)
        
        # Concatenate all state components into a flat vector
        obs = np.concatenate([
            norm_machine_avail_time.flatten(),
            job_weight,
            ops_left,
            makespan_twt,
            all_proc_times.flatten(),
            norm_all_start_times.flatten()
        ])
        
        return torch.tensor(obs, dtype=torch.float16)
    
    def get_action_mask(self) -> torch.Tensor:
        """Get boolean mask for valid actions."""
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
    
    def decode_action(self, action: int) -> Tuple[int, int]:
        """Decode action index to (job_id, machine_id)."""
        job_id = action // self.num_machines
        machine_id = action % self.num_machines
        return job_id, machine_id
    
    def schedule_operation(self, job_id: int, machine_id: int, start_time: float, finish_time: float):
        """Schedule an operation and update state."""
        op_idx = self.job_states[job_id]['current_op']
        op = self.jobs[job_id].operations[op_idx]
        proc_time = op.get_processing_time(machine_id)
        operation_id = op.operation_id
        
        # Record operation schedule
        self.operation_schedules.append({
            'operation_id': operation_id,
            'job_id': job_id,
            'op_idx': op_idx,
            'machine_id': machine_id,
            'start_time': start_time,
            'finish_time': finish_time
        })
        
        # Update machine available time
        self.machine_states[machine_id]['finish_time'] = finish_time
        
        # Update operation status and start time
        self.op_states[(job_id, op_idx)] = 1  # Mark as finished
        self.op_start_times[(job_id, op_idx)] = start_time  # Record start time
        
        # Update job state
        job_state = self.job_states[job_id]
        job_state['completed_ops'] += 1
        
        if job_state['completed_ops'] == len(self.jobs[job_id].operations):
            job_state['finish_time'] = finish_time
            tardiness = max(0, finish_time - getattr(self.jobs[job_id], 'due_date', 0))
            self.current_twt += job_state['weight'] * tardiness
            self.current_makespan = max(self.current_makespan, finish_time)
        else:
            job_state['current_op'] = op_idx + 1
    
    def advance_time(self, new_time: float):
        """Advance the current time."""
        self.current_time = new_time
    
    def is_done(self) -> bool:
        """Check if all jobs are finished."""
        return all(js['completed_ops'] == len(self.jobs[jid].operations) 
                  for jid, js in enumerate(self.job_states))
    
    def has_valid_actions(self) -> bool:
        """Check if there are any valid actions available."""
        return bool(self.get_action_mask().any().item())
    
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
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state."""
        return {
            'current_time': self.current_time,
            'current_makespan': self.current_makespan,
            'current_twt': self.current_twt,
            'jobs_completed': sum(1 for js in self.job_states if js['completed_ops'] == len(self.jobs[0].operations)),
            'total_jobs': self.num_jobs,
            'has_valid_actions': self.has_valid_actions(),
            'is_done': self.is_done()
        } 