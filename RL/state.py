"""
FlatRLState: State management class for Flat RL Environment
Handles state representation and observation generation
"""

import numpy as np
from typing import Dict, Optional
from benchmarks.data_handler import FlexibleJobShopDataHandler
from benchmarks.jobshop_components import Job

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
            max_operations: Maximum operations per job for padding (default: inferred from data)
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
        self.max_processing_time = data_handler.get_max_processing_time()
        self.max_weight = data_handler.get_max_weight()
        self.max_due_date = data_handler.get_max_due_date()
        self.max_machine_available_time = 1
        self.max_operation_start_time = 1
    
            
        # Initialize state variables
        self.obs_dim = self.reset()
    
    def _to_numpy(self) -> np.ndarray:
        """Convert a readable state to a flat 1D numpy array
        Observation vector:
        left_ops / weight / due_date / machine_available_time / process_time / operation_start_time
        """
        job_states = self.readable_state['job_states']
        obs = []
        
        for job_id in range(self.job_dim):
            v = job_states[job_id]
            
            # Job-level features
            obs.append(v['left_ops']/self.operation_dim)  # Normalized remaining operations
            obs.append(v['weight']/self.max_weight)  # Normalized job weight
            obs.append(v['due_date']/self.max_due_date)  # Normalized due date

            # Machine available time for each machine
            available_times = np.array(v['machine_available_time'], dtype=np.float32)
            obs.extend(available_times / self.max_machine_available_time)
            
            for op_pos in range(self.operation_dim):
                # Process time for each machine
                process_times = np.array(v['operations']['process_time'][op_pos], dtype=np.float32)
                obs.extend(process_times / self.max_processing_time)
                
                # Operation start time for each machine
                start_times = np.array(v['operations']['operation_start_time'][op_pos], dtype=np.float32)
                obs.extend(start_times / self.max_operation_start_time)
        
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
                'weight': 0.0 if padding else self.weights[job_id],
                'due_date': 0.0 if padding else self.due_dates[job_id],
                "operations": {
                    "process_time":[],
                    "operation_start_time":[[0] * self.machine_dim for _ in range(self.operation_dim)],
                    "finish_time": [0] * self.operation_dim
                },
                "machine_available_time": [0] * self.machine_dim
            }

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
                                proc_times.append(0) 
                        job_states[job_id]["operations"]["process_time"].append(proc_times)
                    else:
                        job_states[job_id]["operations"]["process_time"].append([0] * self.machine_dim)        
        
        # reset for next episode
        self.max_machine_available_time = 1
        self.max_operation_start_time = 1
        
        self.readable_state = {
            'job_states': job_states
        }
        
        obs_dim = len(self._to_numpy())
        # Action space is decided by each RL environment; only return obs_dim here
        return obs_dim
    
    
    def schedule_operation(self, job_id: int, machine_id: int, start_time: float, finish_time: float):
        """Schedule an operation and update state, then return the new observation vector."""
        # Update job state in self.readable_state
        job_states = self.readable_state['job_states']
        js = job_states[job_id]
        op_position = js['current_op']

        # Update operation_start_time for this operation on the specific machine
        js['operations']['operation_start_time'][op_position][machine_id] = start_time
        self.max_operation_start_time = max(self.max_operation_start_time, start_time)
        
        # Update finish time for this operation
        js['operations']['finish_time'][op_position] = finish_time

        # Mark operation as finished by incrementing current_op and decrementing left_ops
        js['current_op'] += 1
        js['left_ops'] = js['left_ops'] - 1

        # Next operation of this job should start after the current operation finishes
        if js['left_ops'] > 0:  # If job still has operations
            for m_id in range(self.machine_dim):
                js['machine_available_time'][m_id] = max(js['machine_available_time'][m_id], finish_time)

        # Next job on this machine should start after the current job finishes.
        for job_idx in range(self.job_dim):
            job_state = job_states[job_idx]
            if job_state['left_ops'] > 0:
                if job_state['machine_available_time'][machine_id] < finish_time:
                    job_state['machine_available_time'][machine_id] = finish_time
                    self.max_machine_available_time = max(self.max_machine_available_time, finish_time)
        
       
    
    def schedule_operation_with_idleness(self, job_id: int, machine_id: int, start_time: float, proc_time: float, idleness_duration: float):
        """Schedule an operation with an additional idleness duration and update state.
        Args:
            job_id: ID of the job
            machine_id: ID of the machine
            start_time: Original earliest start time for the operation
            proc_time: Processing time for the operation
            idleness_duration: Additional time to add to start_time (>=0)
        """
        adjusted_start_time = start_time + idleness_duration
        finish_time = adjusted_start_time + proc_time

        # Use the existing schedule_operation logic with adjusted times
        self.schedule_operation(job_id, machine_id, adjusted_start_time, finish_time)


    def schedule_idle_machine(self, machine_id: int):
        """Schedule an idle machine for one second.
        Compute the earliest start time across compatible operations and increment by 1.
        Only update machine_available_time for jobs with the smallest value.
        """
        job_states = self.readable_state['job_states']
        earliest_start_time = float('inf')
        min_job_indices = []
        
        # Find all jobs with the earliest start time among compatible jobs
        for job_idx in range(self.job_dim):
            job_state = job_states[job_idx]
            # Check if this job has operations compatible with this machine
            if job_state['left_ops'] > 0:  # Job still has operations to process
                op_idx = job_state['current_op']
                # Check if this operation is compatible with this machine
                if job_state["operations"]["process_time"][op_idx][machine_id] > 0:
                    current_time = job_state['machine_available_time'][machine_id]
                    if current_time < earliest_start_time:
                        earliest_start_time = current_time
                        min_job_indices = [job_idx] # reset the list when find a new earliest start time
                    elif current_time == earliest_start_time:
                        min_job_indices.append(job_idx)
        
        # Only update machine_available_time for jobs with the smallest value
        for job_idx in min_job_indices:
            job_states[job_idx]['machine_available_time'][machine_id] = earliest_start_time + 1
    
    def is_done(self) -> bool:
        """Check if all jobs are finished. i.e. all left_ops are 0"""
        job_states = self.readable_state['job_states']
        return all(js.get('left_ops', 0) == 0 for js in job_states.values())
    
