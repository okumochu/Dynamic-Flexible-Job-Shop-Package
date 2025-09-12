import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional
from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler


class GraphState:
    """
    Graph state representation for Flexible Job Shop Scheduling Problem (FJSP).
    
    Manages the heterogeneous graph structure with operation and machine nodes,
    and handles dynamic state updates during scheduling.
    """
    
    def __init__(self, problem_data: FlexibleJobShopDataHandler, device: Optional[str] = None):
        """
        Initialize the graph state from FJSP problem data.
        
        Args:
            problem_data: FlexibleJobShopDataHandler instance containing the problem definition
        """
        self.problem_data = problem_data
        self.num_operations = problem_data.num_operations
        self.num_machines = problem_data.num_machines
        self.num_jobs = problem_data.num_jobs
        
        # Track current state
        self.current_time = 0
        self.operation_status = np.zeros(self.num_operations, dtype=int)  # 0: unscheduled, 1: scheduled
        self.operation_completion_times = np.zeros(self.num_operations, dtype=float)
        self.machine_available_times = np.zeros(self.num_machines, dtype=float)
        self.machine_workloads = np.zeros(self.num_machines, dtype=float)
        
        # Track which operations are ready to be scheduled (not yet scheduled and predecessors done)
        self.ready_operations = set()
        
        # Track operation-machine assignments
        self.operation_machine_assignments = {}  # op_id -> machine_id
        
        # Track the last operation scheduled on each machine (for disjunctive arcs)
        self.last_op_on_machine: Dict[int, Optional[int]] = {m_id: None for m_id in range(self.num_machines)}
        
        # PERFORMANCE OPTIMIZATION: Store device explicitly to avoid repeated queries
        if device is None or device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # PERFORMANCE OPTIMIZATION: Batch edge updates instead of incremental torch.cat
        self.pending_machine_precedes_edges = []  # List of (src, dst, attr) tuples
        
        # Build the heterogeneous graph
        self.hetero_data = self._build_initial_graph()
        self._update_ready_operations()
        # Update graph features to reflect the correct operation status after determining ready operations
        self._update_graph_features()
    
    def _build_initial_graph(self) -> HeteroData:
        """
        Build the initial heterogeneous graph structure with hierarchical job-operation-machine representation.
        
        Graph Structure:
        - Job nodes: High-level job information (due dates, priorities, progress)
        - Operation nodes: Operation-specific information (processing times, status)  
        - Machine nodes: Resource availability and workload information
        """
        data = HeteroData()
        
        # Build job nodes and features (NEW: hierarchical structure)
        job_features = self._build_job_features()
        data['job'].x = torch.tensor(job_features, dtype=torch.float32, device=self.device)
        
        # Build operation nodes and features (MODIFIED: simplified, job-level info moved to job nodes)
        op_features = self._build_operation_features()
        data['op'].x = torch.tensor(op_features, dtype=torch.float32, device=self.device)
        
        # Build machine nodes and features (UNCHANGED: machine-level information)
        machine_features = self._build_machine_features()
        data['machine'].x = torch.tensor(machine_features, dtype=torch.float32, device=self.device)
        
        # Build edges (EXTENDED: includes job-operation hierarchy)
        self._build_edges(data)
        
        return data
    
    def _build_job_features(self) -> np.ndarray:
        """
        Build initial job node features (NEW: hierarchical structure).
        
        Returns:
            Array of shape (num_jobs, 7) with normalized features:
            [due_date, num_total_ops, total_workload_estimate, priority_weight,
             num_remaining_ops, remaining_workload, job_status]
        """
        features = np.zeros((self.num_jobs, 7))
        
        # Get normalization factors
        max_due_date = self.problem_data.get_max_due_date()
        max_ops_per_job = max(len(self.problem_data.get_job_operations(job_id)) for job_id in range(self.num_jobs))
        max_proc_time = self.problem_data.get_max_processing_time()
        due_dates = self.problem_data.get_jobs_due_date()
        weights = self.problem_data.get_jobs_weight()
        max_weight = max(weights) if weights else 1.0
        
        for job_id in range(self.num_jobs):
            job_operations = self.problem_data.get_job_operations(job_id)
            
            # Static features
            features[job_id, 0] = due_dates[job_id] / max_due_date  # due_date (normalized)
            features[job_id, 1] = len(job_operations) / max_ops_per_job  # num_total_ops (normalized)
            
            # Calculate total workload estimate (minimum processing time sum)
            total_workload = sum(min(op.get_processing_time(m_id) for m_id in op.compatible_machines) 
                               for op in job_operations)
            features[job_id, 2] = total_workload / (max_proc_time * max_ops_per_job)  # total_workload_estimate (normalized)
            
            features[job_id, 3] = weights[job_id] / max_weight  # priority_weight (normalized)
            
            # Dynamic features (will be updated during episode)
            remaining_ops = sum(1 for op in job_operations if self.operation_status[op.operation_id] == 0)
            features[job_id, 4] = remaining_ops / len(job_operations)  # num_remaining_ops (normalized)
            
            # Remaining workload (estimated)
            remaining_workload = sum(min(op.get_processing_time(m_id) for m_id in op.compatible_machines) 
                                   for op in job_operations if self.operation_status[op.operation_id] == 0)
            features[job_id, 5] = remaining_workload / (max_proc_time * max_ops_per_job)  # remaining_workload (normalized)
            
            # Job status: 0=not started, 1=in progress, 2=completed
            if all(self.operation_status[op.operation_id] == 1 for op in job_operations):
                job_status = 2  # completed
            elif any(self.operation_status[op.operation_id] == 1 for op in job_operations):
                job_status = 1  # in progress
            else:
                job_status = 0  # not started
            features[job_id, 6] = job_status / 2.0  # job_status (normalized)
            
        return features
    
    @classmethod
    def get_feature_dimensions(cls) -> tuple[int, int, int]:
        """
        Get the feature dimensions for each node type.
        
        Returns:
            Tuple of (op_feature_dim, machine_feature_dim, job_feature_dim)
        """
        OP_FEATURE_DIM = 8  # [proc_time_mean, proc_time_std, proc_time_min, proc_time_max, 
                           #  status, completion_time, earliest_start_time, num_compatible_machines]
        
        MACHINE_FEATURE_DIM = 7  # [available_time, total_workload, num_compatible_ready_ops,
                                #  ready_ops_proc_time_mean, ready_ops_proc_time_min, 
                                #  ready_ops_proc_time_max, ready_ops_proc_time_std]
                                
        JOB_FEATURE_DIM = 7  # [due_date, num_total_ops, total_workload_estimate, priority_weight,
                            #  num_remaining_ops, remaining_workload, job_status]
                            
        return OP_FEATURE_DIM, MACHINE_FEATURE_DIM, JOB_FEATURE_DIM
    
    def _build_operation_features(self) -> np.ndarray:
        """
        Build initial operation node features (SIMPLIFIED: job-level info moved to job nodes).
        
        Returns:
            Array of shape (num_operations, 8) with normalized features:
            [proc_time_mean, proc_time_std, proc_time_min, proc_time_max, 
             status, completion_time, earliest_start_time, num_compatible_machines]
             
        Note: job_progress removed - now available through job nodes via graph edges
        """
        features = np.zeros((self.num_operations, 8))
        
        # Get global normalization factors
        max_proc_time = self.problem_data.get_max_processing_time()
        max_due_date = self.problem_data.get_max_due_date()
        
        for op_id in range(self.num_operations):
            operation = self.problem_data.get_operation(op_id)
            compatible_machines = operation.compatible_machines
            processing_times = [operation.get_processing_time(m_id) for m_id in compatible_machines]
            
            # Processing time statistics (normalized by max processing time)
            features[op_id, 0] = np.mean(processing_times) / max_proc_time  # proc_time_mean
            features[op_id, 1] = np.std(processing_times) / max_proc_time   # proc_time_std  
            features[op_id, 2] = np.min(processing_times) / max_proc_time   # proc_time_min
            features[op_id, 3] = np.max(processing_times) / max_proc_time   # proc_time_max
            
            # Dynamic features (will be updated)
            features[op_id, 4] = self.operation_status[op_id]  # status (0: unscheduled, 1: scheduled)
            features[op_id, 5] = self.operation_completion_times[op_id] / max_due_date  # completion_time  
            features[op_id, 6] = self._get_earliest_start_time(op_id) / max_due_date  # earliest_start_time
            
            # Compatibility features
            features[op_id, 7] = len(compatible_machines) / self.num_machines  # num_compatible_machines (normalized)
            
        return features
    
    def _build_machine_features(self) -> np.ndarray:
        """
        Build initial machine node features.
        
        Returns:
            Array of shape (num_machines, 7) with normalized features:
            [available_time, total_workload, num_compatible_ready_ops,
             ready_ops_proc_time_mean, ready_ops_proc_time_min, 
             ready_ops_proc_time_max, ready_ops_proc_time_std]
        """
        features = np.zeros((self.num_machines, 7))
        
        # Get normalization factors
        max_due_date = self.problem_data.get_max_due_date()
        max_workload = sum(self.problem_data.get_machine_load(m_id) for m_id in range(self.num_machines))
        max_workload = max(max_workload, 1.0)  # Avoid division by zero
        
        # Get max processing time for normalization
        max_proc_time = self.problem_data.get_max_processing_time()
        
        for m_id in range(self.num_machines):
            features[m_id, 0] = self.machine_available_times[m_id] / max_due_date      # available_time
            features[m_id, 1] = self.machine_workloads[m_id] / max_workload           # total_workload
            
            # Calculate statistics for ready operations that can be processed by this machine
            ready_proc_times = []
            for op_id in self.ready_operations:
                operation = self.problem_data.get_operation(op_id)
                if m_id in operation.compatible_machines:
                    proc_time = operation.get_processing_time(m_id)
                    ready_proc_times.append(proc_time)
            
            # Set processing time statistics features
            if ready_proc_times:
                features[m_id, 2] = len(ready_proc_times) / max(len(self.ready_operations), 1)  # num_compatible_ready_ops (normalized)
                features[m_id, 3] = np.mean(ready_proc_times) / max_proc_time  # ready_ops_proc_time_mean
                features[m_id, 4] = np.min(ready_proc_times) / max_proc_time   # ready_ops_proc_time_min
                features[m_id, 5] = np.max(ready_proc_times) / max_proc_time   # ready_ops_proc_time_max
                features[m_id, 6] = np.std(ready_proc_times) / max_proc_time   # ready_ops_proc_time_std
            else:
                # No compatible ready operations
                features[m_id, 2] = 0.0  # num_compatible_ready_ops
                features[m_id, 3] = 0.0  # ready_ops_proc_time_mean
                features[m_id, 4] = 0.0  # ready_ops_proc_time_min  
                features[m_id, 5] = 0.0  # ready_ops_proc_time_max
                features[m_id, 6] = 0.0  # ready_ops_proc_time_std
            
        return features
    
    def _build_edges(self, data: HeteroData):
        """Build all edge types in the heterogeneous graph (EXTENDED: includes job hierarchy)."""
        
        # 1. Job-operation hierarchy edges (NEW: hierarchical structure)
        job_op_edges, op_job_edges = self._get_job_operation_edges()
        if len(job_op_edges) > 0:
            data['job', 'contains', 'op'].edge_index = torch.tensor(job_op_edges, dtype=torch.long, device=self.device).t()
        if len(op_job_edges) > 0:
            data['op', 'belongs_to', 'job'].edge_index = torch.tensor(op_job_edges, dtype=torch.long, device=self.device).t()
        
        # 2. Operation precedence edges ('op', 'precedes', 'op')
        precedence_edges = self._get_precedence_edges()
        if len(precedence_edges) > 0:
            data['op', 'precedes', 'op'].edge_index = torch.tensor(precedence_edges, dtype=torch.long, device=self.device).t()
        
        # 3. Operation-machine compatibility edges ('op', 'on_machine', 'machine') 
        # and ('machine', 'can_process', 'op')
        op_machine_edges, machine_op_edges = self._get_operation_machine_edges()
        
        if len(op_machine_edges) > 0:
            data['op', 'on_machine', 'machine'].edge_index = torch.tensor(op_machine_edges, dtype=torch.long, device=self.device).t()
        if len(machine_op_edges) > 0:
            data['machine', 'can_process', 'op'].edge_index = torch.tensor(machine_op_edges, dtype=torch.long, device=self.device).t()
    
    def _get_job_operation_edges(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Get hierarchical edges between jobs and their operations (NEW: job nodes).
        
        Returns:
            Tuple of (job_to_op_edges, op_to_job_edges) for bidirectional connectivity
        """
        job_op_edges = []  # (job_id, op_id) for 'job contains op'
        op_job_edges = []  # (op_id, job_id) for 'op belongs_to job'
        
        for job_id in range(self.num_jobs):
            job_operations = self.problem_data.get_job_operations(job_id)
            for operation in job_operations:
                op_id = operation.operation_id
                job_op_edges.append((job_id, op_id))
                op_job_edges.append((op_id, job_id))
                
        return job_op_edges, op_job_edges
    
    def _get_precedence_edges(self) -> List[Tuple[int, int]]:
        """Get precedence edges between operations within jobs."""
        edges = []
        
        for job_id in range(self.num_jobs):
            job_operations = self.problem_data.get_job_operations(job_id)
            
            # Create precedence edges within the job
            for i in range(len(job_operations) - 1):
                pred_op_id = job_operations[i].operation_id
                succ_op_id = job_operations[i + 1].operation_id
                edges.append((pred_op_id, succ_op_id))
                
        return edges
    
    def _get_operation_machine_edges(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Get edges between operations and machines they can be processed on."""
        op_machine_edges = []
        machine_op_edges = []
        
        for op_id in range(self.num_operations):
            operation = self.problem_data.get_operation(op_id)
            for machine_id in operation.compatible_machines:
                op_machine_edges.append((op_id, machine_id))
                machine_op_edges.append((machine_id, op_id))
                
        return op_machine_edges, machine_op_edges
    
    def _get_earliest_start_time(self, op_id: int) -> float:
        """Calculate the earliest possible start time for an operation."""
        job_id, op_position = self.problem_data.get_operation_info(op_id)
        
        if op_position == 0:
            # First operation in job can start immediately
            return 0.0
        else:
            # Must wait for previous operation to complete
            job_operations = self.problem_data.get_job_operations(job_id)
            prev_op = job_operations[op_position - 1]
            return self.operation_completion_times[prev_op.operation_id]
    
    def _update_ready_operations(self):
        """Update which operations are ready to be scheduled."""
        self.ready_operations.clear()
        
        for op_id in range(self.num_operations):
            if self.operation_status[op_id] == 0:  # unscheduled
                job_id, op_position = self.problem_data.get_operation_info(op_id)
                
                if op_position == 0:
                    # First operation in job is ready
                    self.ready_operations.add(op_id)
                else:
                    # Check if previous operation is scheduled
                    job_operations = self.problem_data.get_job_operations(job_id)
                    prev_op = job_operations[op_position - 1]
                    if self.operation_status[prev_op.operation_id] == 1:  # scheduled
                        self.ready_operations.add(op_id)
    
    def update_state(self, op_idx: int, machine_idx: int):
        """
        Update the graph state after scheduling an operation on a machine.
        
        Args:
            op_idx: Operation index to schedule
            machine_idx: Machine index to schedule the operation on
        """
        if op_idx not in self.ready_operations:
            raise ValueError(f"Operation {op_idx} is not ready to be scheduled")
        
        operation = self.problem_data.get_operation(op_idx)
        if machine_idx not in operation.compatible_machines:
            raise ValueError(f"Operation {op_idx} cannot be processed on machine {machine_idx}")
        
        # Calculate processing time and completion time
        processing_time = operation.get_processing_time(machine_idx)
        start_time = max(self.machine_available_times[machine_idx], 
                        self._get_earliest_start_time(op_idx))
        completion_time = start_time + processing_time
        
        # Update operation state
        self.operation_status[op_idx] = 1  # scheduled
        self.operation_completion_times[op_idx] = completion_time
        self.operation_machine_assignments[op_idx] = machine_idx
        
        # Update machine state
        self.machine_available_times[machine_idx] = completion_time
        self.machine_workloads[machine_idx] += processing_time
        
        # Update current time
        self.current_time = max(self.current_time, completion_time)
        
        # Update ready operations
        self._update_ready_operations()
        
        # Update graph features and optionally edges
        self._update_graph_features()
        self._update_dynamic_edges(op_idx, machine_idx)
    
    def _update_graph_features(self):
        """Update dynamic features in the heterogeneous graph (EXTENDED: includes job features)."""
        max_due_date = self.problem_data.get_max_due_date()
        max_workload = max(self.machine_workloads.max(), 1.0)
        
        # Update job features (NEW: dynamic job-level information)
        self._update_job_features()
        
        # Update operation features (status, completion_time, earliest_start_time)
        # NOTE: Indices shifted due to job_progress removal (8 features instead of 9)
        for op_id in range(self.num_operations):
            self.hetero_data['op'].x[op_id, 4] = self.operation_status[op_id]  # status (0: unscheduled, 1: scheduled)
            self.hetero_data['op'].x[op_id, 5] = self.operation_completion_times[op_id] / max_due_date  # completion_time
            self.hetero_data['op'].x[op_id, 6] = self._get_earliest_start_time(op_id) / max_due_date  # earliest_start_time
        
        # Update machine features (available_time, total_workload, ready ops statistics)
        max_proc_time = self.problem_data.get_max_processing_time()
        
        for m_id in range(self.num_machines):
            self.hetero_data['machine'].x[m_id, 0] = self.machine_available_times[m_id] / max_due_date  # available_time
            self.hetero_data['machine'].x[m_id, 1] = self.machine_workloads[m_id] / max_workload  # total_workload
            
            # Calculate statistics for ready operations that can be processed by this machine
            ready_proc_times = []
            for op_id in self.ready_operations:
                operation = self.problem_data.get_operation(op_id)
                if m_id in operation.compatible_machines:
                    proc_time = operation.get_processing_time(m_id)
                    ready_proc_times.append(proc_time)
            
            # Update processing time statistics features
            if ready_proc_times:
                self.hetero_data['machine'].x[m_id, 2] = len(ready_proc_times) / max(len(self.ready_operations), 1)  # num_compatible_ready_ops
                self.hetero_data['machine'].x[m_id, 3] = np.mean(ready_proc_times) / max_proc_time  # ready_ops_proc_time_mean
                self.hetero_data['machine'].x[m_id, 4] = np.min(ready_proc_times) / max_proc_time   # ready_ops_proc_time_min
                self.hetero_data['machine'].x[m_id, 5] = np.max(ready_proc_times) / max_proc_time   # ready_ops_proc_time_max
                self.hetero_data['machine'].x[m_id, 6] = np.std(ready_proc_times) / max_proc_time   # ready_ops_proc_time_std
            else:
                # No compatible ready operations
                self.hetero_data['machine'].x[m_id, 2] = 0.0  # num_compatible_ready_ops
                self.hetero_data['machine'].x[m_id, 3] = 0.0  # ready_ops_proc_time_mean
                self.hetero_data['machine'].x[m_id, 4] = 0.0  # ready_ops_proc_time_min
                self.hetero_data['machine'].x[m_id, 5] = 0.0  # ready_ops_proc_time_max
                self.hetero_data['machine'].x[m_id, 6] = 0.0  # ready_ops_proc_time_std
    
    def _update_job_features(self):
        """
        Update dynamic job node features (NEW: hierarchical structure).
        
        Updates: num_remaining_ops, remaining_workload, job_status
        """
        max_ops_per_job = max(len(self.problem_data.get_job_operations(job_id)) for job_id in range(self.num_jobs))
        max_proc_time = self.problem_data.get_max_processing_time()
        
        for job_id in range(self.num_jobs):
            job_operations = self.problem_data.get_job_operations(job_id)
            
            # Update dynamic features
            remaining_ops = sum(1 for op in job_operations if self.operation_status[op.operation_id] == 0)
            self.hetero_data['job'].x[job_id, 4] = remaining_ops / len(job_operations)  # num_remaining_ops (normalized)
            
            # Remaining workload (estimated)
            remaining_workload = sum(min(op.get_processing_time(m_id) for m_id in op.compatible_machines) 
                                   for op in job_operations if self.operation_status[op.operation_id] == 0)
            self.hetero_data['job'].x[job_id, 5] = remaining_workload / (max_proc_time * max_ops_per_job)  # remaining_workload (normalized)
            
            # Job status: 0=not started, 1=in progress, 2=completed
            if all(self.operation_status[op.operation_id] == 1 for op in job_operations):
                job_status = 2  # completed
            elif any(self.operation_status[op.operation_id] == 1 for op in job_operations):
                job_status = 1  # in progress
            else:
                job_status = 0  # not started
            self.hetero_data['job'].x[job_id, 6] = job_status / 2.0  # job_status (normalized)
    
    def get_observation(self) -> HeteroData:
        """
        Get the current graph observation.
        
        PERFORMANCE OPTIMIZED: Flushes pending edge updates before returning observation.
        
        Returns:
            Current HeteroData object representing the state
        """
        # Flush any pending edge updates before returning observation
        self._flush_pending_edges()
        return self.hetero_data
    
    def get_ready_operations(self) -> List[int]:
        """Get list of operations that are ready to be scheduled."""
        return list(self.ready_operations)
    
    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """Get list of valid (operation, machine) action pairs."""
        valid_actions = []
        ready_ops = self.get_ready_operations()
        
        for op_id in ready_ops:
            operation = self.problem_data.get_operation(op_id)
            for machine_id in operation.compatible_machines:
                valid_actions.append((op_id, machine_id))
                
        return valid_actions
    
    def is_done(self) -> bool:
        """Check if all operations are completed."""
        return np.all(self.operation_status == 1)  # all scheduled
    
    def get_makespan(self) -> float:
        """Get current makespan (maximum completion time so far)."""
        # Return current maximum completion time, even if not all operations are done
        scheduled_ops_mask = self.operation_status == 1
        if np.any(scheduled_ops_mask):
            return self.operation_completion_times[scheduled_ops_mask].max()
        else:
            return 0.0  # No operations scheduled yet
    
    
    def reset(self):
        """Reset the state to initial conditions."""
        self.current_time = 0
        self.operation_status.fill(0)  # all unscheduled
        self.operation_completion_times.fill(0)
        self.machine_available_times.fill(0)
        self.machine_workloads.fill(0)
        self.ready_operations.clear()
        self.operation_machine_assignments.clear()
        self.last_op_on_machine = {m_id: None for m_id in range(self.num_machines)}
        
        # PERFORMANCE OPTIMIZATION: Clear pending edge updates
        self.pending_machine_precedes_edges.clear()
        
        # Rebuild the graph with fresh features
        self.hetero_data = self._build_initial_graph()
        self._update_ready_operations()
        # Update graph features to reflect the correct operation status after determining ready operations
        self._update_graph_features()
    
    def _update_dynamic_edges(self, scheduled_op_idx: int, machine_idx: int):
        """
        Dynamically add disjunctive arcs between operations on the same machine.
        
        PERFORMANCE OPTIMIZED: Batches edge updates instead of incremental torch.cat.
        
        Args:
            scheduled_op_idx: The operation that was just scheduled
            machine_idx: The machine it was assigned to
        """
        # Find the last operation that was scheduled on this machine
        last_op_on_this_machine = self.last_op_on_machine[machine_idx]
        
        # If there was a previous operation, queue the edge for batch update
        if last_op_on_this_machine is not None:
            # Get processing time of the last operation (this is the temporal delay ΔT)
            last_operation = self.problem_data.get_operation(last_op_on_this_machine)
            last_op_assigned_machine = self.operation_machine_assignments.get(last_op_on_this_machine, machine_idx)
            last_op_processing_time = last_operation.get_processing_time(last_op_assigned_machine)
            
            # Queue the edge for batch update (src, dst, attr)
            self.pending_machine_precedes_edges.append((
                last_op_on_this_machine, 
                scheduled_op_idx, 
                last_op_processing_time
            ))
        
        # CRITICAL: Update the tracker to remember that the current operation is now 
        # the last one scheduled on this machine
        self.last_op_on_machine[machine_idx] = scheduled_op_idx
        
        # Optional: Also add assignment edges for completeness
        self._add_assignment_edges(scheduled_op_idx, machine_idx)
    
    def _flush_pending_edges(self):
        """
        PERFORMANCE OPTIMIZATION: Batch update all pending machine_precedes edges.
        
        This method converts O(n) individual torch.cat operations into a single 
        large concatenation, significantly improving performance for long episodes.
        """
        if not self.pending_machine_precedes_edges:
            return  # No pending edges to flush
            
        edge_type = ('op', 'machine_precedes', 'op')
        
        # Initialize edge type if it doesn't exist
        if edge_type not in self.hetero_data.edge_types:
            self.hetero_data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            self.hetero_data[edge_type].edge_attr = torch.empty((0, 1), dtype=torch.float32, device=self.device)
        
        # Convert pending edges to tensors
        src_nodes = []
        dst_nodes = []
        edge_attrs = []
        
        for src, dst, attr in self.pending_machine_precedes_edges:
            src_nodes.append(src)
            dst_nodes.append(dst)
            edge_attrs.append(attr)
        
        # Create new edge tensors
        new_edges = torch.tensor([src_nodes, dst_nodes], dtype=torch.long, device=self.device)
        new_attrs = torch.tensor([[attr] for attr in edge_attrs], dtype=torch.float32, device=self.device)
        
        # Single concatenation instead of multiple small ones
        self.hetero_data[edge_type].edge_index = torch.cat(
            [self.hetero_data[edge_type].edge_index, new_edges], dim=1
        )
        self.hetero_data[edge_type].edge_attr = torch.cat(
            [self.hetero_data[edge_type].edge_attr, new_attrs], dim=0
        )
        
        # Clear pending edges
        self.pending_machine_precedes_edges.clear()
    
    def _add_assignment_edges(self, scheduled_op_idx: int, machine_idx: int):
        """
        Add assignment edges between operation and machine.
        
        Args:
            scheduled_op_idx: The operation that was just scheduled
            machine_idx: The machine it was assigned to
        """
        # Create assignment edge type if it doesn't exist
        if ('op', 'assigned_to', 'machine') not in self.hetero_data.edge_types:
            # PERFORMANCE OPTIMIZATION: Use stored device instead of querying
            self.hetero_data['op', 'assigned_to', 'machine'].edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        # Add the new assignment edge
        current_edges = self.hetero_data['op', 'assigned_to', 'machine'].edge_index
        new_edge = torch.tensor([[scheduled_op_idx], [machine_idx]], dtype=torch.long, device=self.device)
        updated_edges = torch.cat([current_edges, new_edge], dim=1)
        self.hetero_data['op', 'assigned_to', 'machine'].edge_index = updated_edges
        
        # Also add reverse edge for symmetry
        if ('machine', 'processes', 'op') not in self.hetero_data.edge_types:
            self.hetero_data['machine', 'processes', 'op'].edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        current_reverse_edges = self.hetero_data['machine', 'processes', 'op'].edge_index
        new_reverse_edge = torch.tensor([[machine_idx], [scheduled_op_idx]], dtype=torch.long, device=self.device)
        updated_reverse_edges = torch.cat([current_reverse_edges, new_reverse_edge], dim=1)
        self.hetero_data['machine', 'processes', 'op'].edge_index = updated_reverse_edges
