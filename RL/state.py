"""
FlatRLState: State management class for Flat RL Environment
Handles state representation and observation generation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
from benchmarks.static_benchmark.jobshop_components import Job

class State:
    """
    State management class for Flat RL Environment.
    
    This class focuses on state representation and management:
    - State dimensions and space definitions
    - Observation generation
    - State initialization and updates
    - Global statistics for normalization
    """
    
    def __init__(self, data_handler: FlexibleJobShopDataHandler, 
                 max_jobs: Optional[int] = None, 
                 max_operations: Optional[int] = None):
        """
        Initialize the state manager.
        
        Args:
            data_handler: FlexibleJobShopDataHandler instance
            max_jobs: Maximum jobs for padding (default: num_jobs)
            max_machines: Maximum machines for padding (default: num_machines)
        """
        self.data_handler = data_handler
        self.jobs : Dict[int, Job] = data_handler.jobs
        self.num_jobs = data_handler.num_jobs
        self.num_machines = data_handler.num_machines
        
        # Extract due dates and weights
        self.due_dates = data_handler.get_jobs_due_date()
        self.weights = data_handler.get_jobs_weight()
        
        # Set padding dimensions
        self.job_dim = max_jobs if max_jobs is not None else self.num_jobs
        self.machine_dim = self.num_machines # no padding for machines, the machine is fixed
        self.operation_dim = max_operations if max_operations is not None else max(len(job.operations) for job in self.jobs.values())
        
        # Calculate global statistics for normalization
        self.avg_processing_time = data_handler.get_average_processing_time()
        self.max_weight = data_handler.get_max_weight()
        self.max_due_date = data_handler.get_max_due_date()
    
            
        # Initialize state variables
        self.obs_dim, self.action_dim = self.reset()
    
    def _to_numpy(self) -> np.ndarray:
        """Convert a readable state to a flat 1D numpy array observation vector (robust, index-based)."""
        job_states = self.readable_state['job_states']
        obs = []
        
        # Calculate max machine available time and max operation start time for normalization
        max_machine_available_time = max(
            max(job_state['machine_available_time']) 
            for job_state in job_states.values()
        ) if any(job_states.values()) else 1.0
        
        max_operation_start_time = max(
            max(max(op_start_times) for op_start_times in job_state['operations']['operation_start_time']) 
            for job_state in job_states.values()
        ) if any(job_states.values()) else 1.0
        
        # Avoid division by zero
        max_machine_available_time = max(max_machine_available_time, 1.0)
        max_operation_start_time = max(max_operation_start_time, 1.0)
        
        for job_id in range(self.job_dim):
            v = job_states[job_id]
            obs.append(v['left_ops']/self.operation_dim) # only left ops, current op is for recording purpose only
            obs.append(v['weight']/self.max_weight)
            for op_pos in range(self.operation_dim):
                # Fix: process_time is [operation][machine], so we need to handle it properly
                # Take the minimum processing time across all machines for this operation
                if op_pos < len(v['operations']['process_time']) and isinstance(v['operations']['process_time'][op_pos], list):
                    compatible_proc_times = [pt for pt in v['operations']['process_time'][op_pos] if pt > 0]
                    min_proc_time = min(compatible_proc_times) if compatible_proc_times else 0
                    obs.append(min_proc_time / self.avg_processing_time)
                else:
                    obs.append(0.0) # No processing time if op doesn't exist
                # Add operation start time for each machine (matrix representation)
                for machine_id in range(self.machine_dim):
                    obs.append(v['operations']['operation_start_time'][op_pos][machine_id]/max_operation_start_time)
            for machine_id in range(self.machine_dim):
                obs.append(v['machine_available_time'][machine_id]/max_machine_available_time)
        return np.array(obs, dtype=np.float32)
    
    def reset(self):
        """Firstly, construct readable state. Then, reset the state to initial values.
            operation_start_time is a matrix, each row is a operation, each column is a machine
            finish_time is a vector, each element is a operation finish time
            machine_available_time is a vector, each element is a machine available time
            process_time is a matrix, each row is a operation, each column is a machine
        """
        
        # Initialize job states
        job_states = {}
        for job_id in range(self.job_dim):

            # If job is not in the jobs dict, it is padding
            padding = False if job_id in self.jobs.keys() else True

            job_states[job_id] = {
                # left_ops is normalized by operation_dim
                'left_ops': 0 if padding else len(self.jobs[job_id].operations),
                'current_op': 0,
                'weight': 0.0 if padding else self.weights[job_id]/self.max_weight,
                "operations": {
                    "process_time":[],
                    "operation_start_time":[[0] * self.machine_dim for _ in range(self.operation_dim)],
                    "finish_time": [0] * self.operation_dim
                },
                "machine_available_time": [0] * self.machine_dim
            }

            # process time and start time are normalized by machine_dim
            if padding:
                job_states[job_id]["operations"]["process_time"] = [[0] * self.machine_dim for _ in range(self.operation_dim)]
            else:
                # fill with actual process time and start time
                # still need padding to fill the gap of (operation_dim - len(self.jobs[job_id].operations))
                for op_position in range(self.operation_dim):
                    if op_position < len(self.jobs[job_id].operations):
                        op = self.jobs[job_id].operations[op_position]
                        # Only get processing time for compatible machines, use 0 for others
                        proc_times = []
                        for machine_id in range(self.machine_dim):
                            if machine_id in op.compatible_machines:
                                proc_times.append(op.get_processing_time(machine_id))
                            else:
                                proc_times.append(0)  # Invalid/incompatible machine
                        job_states[job_id]["operations"]["process_time"].append(proc_times)
                    else:
                        job_states[job_id]["operations"]["process_time"].append([0] * self.machine_dim)        
        
        self.readable_state = {
            'job_states': job_states
        }
        
        obs_dim = len(self._to_numpy())
        action_dim = self.num_jobs * self.num_machines
        return obs_dim, action_dim
    
    
    def schedule_operation(self, job_id: int, machine_id: int, start_time: float, finish_time: float):
        """Schedule an operation and update state, then return the new observation vector."""
        # Update job state in self.readable_state
        job_states = self.readable_state['job_states']
        js = job_states[job_id]
        op_position = js['current_op']

        # Update operation_start_time for this operation on the specific machine
        js['operations']['operation_start_time'][op_position][machine_id] = start_time
        
        # Update finish time for this operation
        js['operations']['finish_time'][op_position] = finish_time

        # Mark operation as finished by incrementing current_op and decrementing left_ops
        js['current_op'] += 1
        js['left_ops'] = js['left_ops'] - 1

        # Update machine_available_time for all jobs (set finish_time for the machine)
        # Consider constraint: machine can't start before it's available or before previous operation finishes
        for job_idx in range(self.num_jobs):
            job_state = job_states[job_idx]
            job_state['machine_available_time'][machine_id] = max(
                job_state['machine_available_time'][machine_id],  # earliest machine available time
                finish_time  # last operation finish time
            )

        # Return the new observation vector
        return self._to_numpy()
    
    def is_done(self) -> bool:
        """Check if all jobs are finished. i.e. all left_ops are 0"""
        job_states = self.readable_state['job_states']
        return all(js.get('left_ops', 0) == 0 for js in job_states.values())