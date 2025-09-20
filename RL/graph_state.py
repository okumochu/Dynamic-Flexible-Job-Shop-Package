import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional
from benchmarks.data_handler import FlexibleJobShopDataHandler


class GraphState:
    """
    Graph state representation for Flexible Job Shop Scheduling Problem (FJSP).
    
    Manages the heterogeneous graph structure with operation and machine nodes,
    and handles dynamic state updates during scheduling.
    """
    
    def __init__(self, problem_data: FlexibleJobShopDataHandler, device: str):
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
        self.operation_status = np.zeros(self.num_operations, dtype=int)  # 0: unscheduled, 1: scheduled
        self.operation_completion_times = np.zeros(self.num_operations, dtype=float)
        self.machine_available_times = np.zeros(self.num_machines, dtype=float)
        
        # Track job timing (raw values, not normalized)
        self.job_start_times = np.zeros(self.num_jobs, dtype=float)  # Raw start time for each job
        self.job_finished_times = np.zeros(self.num_jobs, dtype=float)  # Raw finished time for each job
        
        # Track which operations are ready to be scheduled (not yet scheduled and predecessors done)
        self.ready_operations = set()
        
        # Track operation-machine assignments
        self.operation_machine_assignments = {}  # op_id -> machine_id
        
        # Track the last operation scheduled on each machine (for disjunctive arcs)
        self.last_op_on_machine: Dict[int, Optional[int]] = {m_id: None for m_id in range(self.num_machines)}
        
        # Set device
        self.device = device
        
        # Feature dimensions - will be set when features are built
        self.job_feature_dim = None
        self.op_feature_dim = None
        self.machine_feature_dim = None
        
        # Build the heterogeneous graph
        self.hetero_data = self._build_initial_graph()
        self._update_ready_operations()
        # Update graph features to reflect the correct operation status after determining ready operations
        self._update_graph_features(initialization=True)
    
    def _build_initial_graph(self) -> HeteroData:
        """
        Build the initial heterogeneous graph structure with hierarchical job-operation-machine representation.
        
        Graph Structure:
        - Job nodes: High-level job information (due dates, priorities, progress)
        - Operation nodes: Operation-specific information (processing times, status)  
        - Machine nodes: Resource availability and workload information
        """
        data = HeteroData()
        
        # Build job nodes and features
        job_features = self._build_job_features()
        data['job'].x = torch.tensor(job_features, dtype=torch.float32, device=self.device)
        
        # Build operation nodes and features
        op_features = self._build_operation_features()
        data['op'].x = torch.tensor(op_features, dtype=torch.float32, device=self.device)
        
        # Build machine nodes and features
        machine_features = self._build_machine_features()
        data['machine'].x = torch.tensor(machine_features, dtype=torch.float32, device=self.device)
        
        # Build edges
        self._build_edges(data)
        
        return data
    
    def _build_job_features(self) -> np.ndarray:
        """
        Build initial job node features
        
        Returns:
            Array of shape (num_jobs, 4) with normalized features:
            [num_total_ops, num_remaining_ops, start_time, finished_time]
        """
        features = np.zeros((self.num_jobs, 4))
        
        # Get normalization factors
        max_ops_per_job = max(len(self.problem_data.get_job_operations(job_id)) for job_id in range(self.num_jobs))
        current_makespan = max(self.get_makespan(), 1.0)  # Use current makespan for normalization
        
        for job_id in range(self.num_jobs):
            job_operations = self.problem_data.get_job_operations(job_id)
            
            # Static features
            features[job_id, 0] = len(job_operations) / max_ops_per_job  # num_total_ops (normalized)
            
            # Dynamic features (will be updated during episode)
            remaining_ops = sum(1 for op in job_operations if self.operation_status[op.operation_id] == 0)
            features[job_id, 1] = remaining_ops / len(job_operations)  # num_remaining_ops (normalized)
            
            # Job timing features (initialized as 0, normalized by makespan)
            features[job_id, 2] = 0.0  # start_time (normalized)
            features[job_id, 3] = 0.0  # finished_time (normalized)
        
        # Set feature dimension dynamically
        self.job_feature_dim = features.shape[1]
            
        return features
    
    def get_feature_dimensions(self) -> tuple[int, int, int]:
        """
        Get the feature dimensions for each node type.
        
        Returns:
            Tuple of (op_feature_dim, machine_feature_dim, job_feature_dim)
        """
        return self.op_feature_dim, self.machine_feature_dim, self.job_feature_dim
    
    def _build_operation_features(self) -> np.ndarray:
        """
        Build initial operation node features (SIMPLIFIED: job-level info moved to job nodes).
        
        Returns:
            Array of shape (num_operations, 9) with normalized features:
            [proc_time_mean, proc_time_std, proc_time_min, proc_time_max, 
             status, start_time, completion_time, earliest_start_time, num_compatible_machines]
             
        Note: job_progress removed - now available through job nodes via graph edges
        """
        features = np.zeros((self.num_operations, 9))
        
        # Get global normalization factors
        max_proc_time = self.problem_data.get_max_processing_time()
        current_makespan = max(self.get_makespan(), 1.0)  # Use current makespan for normalization
        
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
            
            # Start time and completion time (normalized by current makespan, 0 if not scheduled)
            if self.operation_status[op_id] == 1:  # scheduled
                start_time = self.operation_completion_times[op_id] - operation.get_processing_time(self.operation_machine_assignments[op_id])
                features[op_id, 5] = start_time / current_makespan  # start_time
                features[op_id, 6] = self.operation_completion_times[op_id] / current_makespan  # completion_time
            else:  # not scheduled
                features[op_id, 5] = 0.0  # start_time
                features[op_id, 6] = 0.0  # completion_time
            
            features[op_id, 7] = self._get_earliest_start_time(op_id) / current_makespan  # earliest_start_time
            
            # Compatibility features
            features[op_id, 8] = len(compatible_machines) / self.num_machines  # num_compatible_machines (normalized)
        
        # Set feature dimension dynamically
        self.op_feature_dim = features.shape[1]
            
        return features
    
    def _build_machine_features(self) -> np.ndarray:
        """
        Build initial machine node features.
        
        Returns:
            Array of shape (num_machines, 6) with normalized features:
            [available_time, num_compatible_ready_ops,
             ready_ops_proc_time_mean, ready_ops_proc_time_min,
             ready_ops_proc_time_max, ready_ops_proc_time_std]
        """
        features = np.zeros((self.num_machines, 6))
        
        # Get normalization factors
        current_makespan = max(self.get_makespan(), 1.0)  # Use current makespan for normalization
        
        # Get max processing time for normalization
        max_proc_time = self.problem_data.get_max_processing_time()
        
        for m_id in range(self.num_machines):
            features[m_id, 0] = self.machine_available_times[m_id] / current_makespan  # available_time
            
            # Calculate statistics for ready operations that can be processed by this machine
            ready_proc_times = []
            for op_id in self.ready_operations:
                operation = self.problem_data.get_operation(op_id)
                if m_id in operation.compatible_machines:
                    proc_time = operation.get_processing_time(m_id)
                    ready_proc_times.append(proc_time)
            
            # Set processing time statistics features
            if ready_proc_times:
                features[m_id, 1] = len(ready_proc_times) / max(len(self.ready_operations), 1)  # num_compatible_ready_ops (normalized)
                features[m_id, 2] = np.mean(ready_proc_times) / max_proc_time  # ready_ops_proc_time_mean
                features[m_id, 3] = np.min(ready_proc_times) / max_proc_time   # ready_ops_proc_time_min
                features[m_id, 4] = np.max(ready_proc_times) / max_proc_time   # ready_ops_proc_time_max
                features[m_id, 5] = np.std(ready_proc_times) / max_proc_time   # ready_ops_proc_time_std
            else:
                # No compatible ready operations
                features[m_id, 1] = 0.0  # num_compatible_ready_ops
                features[m_id, 2] = 0.0  # ready_ops_proc_time_mean
                features[m_id, 3] = 0.0  # ready_ops_proc_time_min  
                features[m_id, 4] = 0.0  # ready_ops_proc_time_max
                features[m_id, 5] = 0.0  # ready_ops_proc_time_std
        
        # Set feature dimension dynamically
        self.machine_feature_dim = features.shape[1]
            
        return features
    
    def _build_edges(self, data: HeteroData):
        """Build all edge types in the heterogeneous graph"""
        
        # 1. Job-operation hierarchy edges
        job_sources, op_targets, op_sources, job_targets = self._get_job_operation_edges()
        if len(job_sources) > 0:
            data['job', 'contains', 'op'].edge_index = torch.tensor([job_sources, op_targets], dtype=torch.long, device=self.device)
        if len(op_sources) > 0:
            data['op', 'belongs_to', 'job'].edge_index = torch.tensor([op_sources, job_targets], dtype=torch.long, device=self.device)
        
        # 2. Operation precedence edges ('op', 'precedes', 'op')
        precedence_sources, precedence_targets = self._get_precedence_edges()
        if len(precedence_sources) > 0:
            data['op', 'precedes', 'op'].edge_index = torch.tensor([precedence_sources, precedence_targets], dtype=torch.long, device=self.device)
        
        # 3. Operation-machine compatibility edges ('op', 'on_machine', 'machine') 
        # and ('machine', 'can_process', 'op')
        op_sources, machine_targets, machine_sources, op_targets = self._get_operation_machine_edges()
        
        if len(op_sources) > 0:
            data['op', 'on_machine', 'machine'].edge_index = torch.tensor([op_sources, machine_targets], dtype=torch.long, device=self.device)
        if len(machine_sources) > 0:
            data['machine', 'can_process', 'op'].edge_index = torch.tensor([machine_sources, op_targets], dtype=torch.long, device=self.device)
        
        # 4. Initialize dynamic edge types (empty initially, populated during scheduling)
        # Machine precedence edges (disjunctive arcs)
        data['op', 'machine_precedes', 'op'].edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        data['op', 'machine_precedes', 'op'].edge_attr = torch.empty((0, 1), dtype=torch.float32, device=self.device)
        
        # Assignment edges (operation -> machine)
        data['op', 'assigned_to', 'machine'].edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        # Assignment edges (machine -> operation) 
        data['machine', 'processes', 'op'].edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
    
    def _get_job_operation_edges(self) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Get hierarchical edges between jobs and their operations in PyG format
        
        Returns:
            Tuple of (job_sources, op_targets, op_sources, job_targets) for bidirectional connectivity
        """
        job_sources = []  # Source job nodes
        op_targets = []   # Target operation nodes
        op_sources = []   # Source operation nodes  
        job_targets = []  # Target job nodes
        
        for job_id in range(self.num_jobs):
            job_operations = self.problem_data.get_job_operations(job_id)
            for operation in job_operations:
                op_id = operation.operation_id
                # Job -> Operation edges
                job_sources.append(job_id)
                op_targets.append(op_id)
                # Operation -> Job edges
                op_sources.append(op_id)
                job_targets.append(job_id)
                
        return job_sources, op_targets, op_sources, job_targets
    
    def _get_precedence_edges(self) -> Tuple[List[int], List[int]]:
        """Get bidirectional precedence edges between operations within jobs in PyG format."""
        sources = []
        targets = []
        
        for job_id in range(self.num_jobs):
            job_operations = self.problem_data.get_job_operations(job_id)
            
            # Create bidirectional precedence edges within the job
            for i in range(len(job_operations) - 1):
                pred_op_id = job_operations[i].operation_id
                succ_op_id = job_operations[i + 1].operation_id
                # Add both directions for proper message passing
                sources.extend([pred_op_id, succ_op_id])  # predecessor -> successor, successor -> predecessor
                targets.extend([succ_op_id, pred_op_id])  # successor -> predecessor, predecessor -> successor
                
        return sources, targets
    
    def _get_operation_machine_edges(self) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Get edges between operations and machines they can be processed on in PyG format."""
        op_sources = []
        machine_targets = []
        machine_sources = []
        op_targets = []
        
        for op_id in range(self.num_operations):
            operation = self.problem_data.get_operation(op_id)
            for machine_id in operation.compatible_machines:
                # Operation -> Machine edges
                op_sources.append(op_id)
                machine_targets.append(machine_id)
                # Machine -> Operation edges
                machine_sources.append(machine_id)
                op_targets.append(op_id)
                
        return op_sources, machine_targets, machine_sources, op_targets
    
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
        
        operation = self.problem_data.get_operation(op_idx)
        
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
        
        # Update ready operations
        self._update_ready_operations()
        
        # Update graph features and optionally edges
        self._update_graph_features(op_idx, start_time, completion_time)
        self._update_dynamic_edges(op_idx, machine_idx)
    
    def _update_graph_features(self, op_idx: int = None, start_time: float = None, completion_time: float = None, initialization: bool = False):
        """Update dynamic features in the heterogeneous graph"""
        current_makespan = max(self.get_makespan(), 1.0)  # Use current makespan for normalization
        
        # Update job features
        self._update_job_features(op_idx, start_time, completion_time, initialization)
        
        for op_id in range(self.num_operations):
            # Debug: Check tensor dimensions and op_id bounds
            try:
                self.hetero_data['op'].x[op_id, 4] = self.operation_status[op_id]  # status (0: unscheduled, 1: scheduled)
                
                # Start time and completion time (normalized by current makespan, 0 if not scheduled)
                if self.operation_status[op_id] == 1:  # scheduled
                    operation = self.problem_data.get_operation(op_id)
                    start_time = self.operation_completion_times[op_id] - operation.get_processing_time(self.operation_machine_assignments[op_id])
                    self.hetero_data['op'].x[op_id, 5] = start_time / current_makespan  # start_time
                    self.hetero_data['op'].x[op_id, 6] = self.operation_completion_times[op_id] / current_makespan  # completion_time
                else:  # not scheduled
                    self.hetero_data['op'].x[op_id, 5] = 0.0  # start_time
                    self.hetero_data['op'].x[op_id, 6] = 0.0  # completion_time
                
                self.hetero_data['op'].x[op_id, 7] = self._get_earliest_start_time(op_id) / current_makespan  # earliest_start_time
            except Exception as e:
                print(f"Error updating op_id {op_id}: {e}")
                print(f"Tensor shape: {self.hetero_data['op'].x.shape}")
                print(f"num_operations: {self.num_operations}")
                print(f"op_id: {op_id}")
                raise
        
        # Update machine features (available_time, total_workload, ready ops statistics)
        max_proc_time = self.problem_data.get_max_processing_time()
        
        for m_id in range(self.num_machines):
            self.hetero_data['machine'].x[m_id, 0] = self.machine_available_times[m_id] / current_makespan  # available_time
            
            # Calculate statistics for ready operations that can be processed by this machine
            ready_proc_times = []
            for op_id in self.ready_operations:
                operation = self.problem_data.get_operation(op_id)
                if m_id in operation.compatible_machines:
                    proc_time = operation.get_processing_time(m_id)
                    ready_proc_times.append(proc_time)
            
            # Update processing time statistics features
            if ready_proc_times:
                self.hetero_data['machine'].x[m_id, 1] = len(ready_proc_times) / max(len(self.ready_operations), 1)  # num_compatible_ready_ops
                self.hetero_data['machine'].x[m_id, 2] = np.mean(ready_proc_times) / max_proc_time  # ready_ops_proc_time_mean
                self.hetero_data['machine'].x[m_id, 3] = np.min(ready_proc_times) / max_proc_time   # ready_ops_proc_time_min
                self.hetero_data['machine'].x[m_id, 4] = np.max(ready_proc_times) / max_proc_time   # ready_ops_proc_time_max
                self.hetero_data['machine'].x[m_id, 5] = np.std(ready_proc_times) / max_proc_time   # ready_ops_proc_time_std
            else:
                # No compatible ready operations
                self.hetero_data['machine'].x[m_id, 1] = 0.0  # num_compatible_ready_ops
                self.hetero_data['machine'].x[m_id, 2] = 0.0  # ready_ops_proc_time_mean
                self.hetero_data['machine'].x[m_id, 3] = 0.0  # ready_ops_proc_time_min
                self.hetero_data['machine'].x[m_id, 4] = 0.0  # ready_ops_proc_time_max
                self.hetero_data['machine'].x[m_id, 5] = 0.0  # ready_ops_proc_time_std
    
    def _update_job_features(self, op_idx: int = None, start_time: float = None, completion_time: float = None, initialization: bool = False):
        """
        Update dynamic job node features
        
        Updates: num_remaining_ops, start_time, finished_time
        
        Args:
            op_idx: Operation index that was scheduled (optional)
            start_time: Start time of the operation (optional)
            completion_time: Completion time of the operation (optional)
            initialization: If True, skip operation-specific updates (used during initialization)
        """
        # Handle initialization case - skip operation-specific updates
        if not initialization and op_idx is not None:
            # Update job timing only for specific operations
            job_id, op_position = self.problem_data.get_operation_info(op_idx)
            job_operations = self.problem_data.get_job_operations(job_id)
            
            # Update job start time ONLY when first operation starts
            if op_position == 0 and self.job_start_times[job_id] == 0:  # First operation AND not started yet
                self.job_start_times[job_id] = start_time
            
            # Update job finished time ONLY when last operation finishes
            if op_position == len(job_operations) - 1:  # Last operation in job
                self.job_finished_times[job_id] = completion_time
        
        current_makespan = max(self.get_makespan(), 1.0)  # Use current makespan for normalization
        
        for job_id in range(self.num_jobs):
            job_operations = self.problem_data.get_job_operations(job_id)
            
            # Update dynamic features
            remaining_ops = sum(1 for op in job_operations if self.operation_status[op.operation_id] == 0)
            self.hetero_data['job'].x[job_id, 1] = remaining_ops / len(job_operations)  # num_remaining_ops (normalized)
            
            # Update job timing features (normalized by makespan)
            self.hetero_data['job'].x[job_id, 2] = self.job_start_times[job_id] / current_makespan  # start_time (normalized)
            self.hetero_data['job'].x[job_id, 3] = self.job_finished_times[job_id] / current_makespan  # finished_time (normalized)
    
    def get_observation(self) -> HeteroData:
        """
        Get the current graph observation.
        
        Returns:
            Current HeteroData object representing the state
        """
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
        self.operation_status.fill(0)  # all unscheduled
        self.operation_completion_times.fill(0)
        self.machine_available_times.fill(0)
        self.job_start_times.fill(0)  # Reset job timing
        self.job_finished_times.fill(0)  # Reset job timing
        self.ready_operations.clear()
        self.operation_machine_assignments.clear()
        self.last_op_on_machine = {m_id: None for m_id in range(self.num_machines)}
        
        # Rebuild the graph with fresh features
        self.hetero_data = self._build_initial_graph()
        self._update_ready_operations()
        # Update graph features to reflect the correct operation status after determining ready operations
        self._update_graph_features(initialization=True)
    
    def _update_dynamic_edges(self, scheduled_op_idx: int, machine_idx: int):
        """
        Dynamically add disjunctive arcs between operations on the same machine.
        
        Args:
            scheduled_op_idx: The operation that was just scheduled
            machine_idx: The machine it was assigned to
        """
        # Find the last operation that was scheduled on this machine
        last_op_on_this_machine = self.last_op_on_machine[machine_idx]
        
        # If there was a previous operation, add the disjunctive edge immediately
        if last_op_on_this_machine is not None:
            # Get processing time of the last operation (this is the temporal delay ΔT)
            last_operation = self.problem_data.get_operation(last_op_on_this_machine)
            last_op_assigned_machine = self.operation_machine_assignments.get(last_op_on_this_machine, machine_idx)
            last_op_processing_time = last_operation.get_processing_time(last_op_assigned_machine)
            
            # Add the disjunctive edge immediately
            self._add_machine_precedes_edge(
                last_op_on_this_machine, 
                scheduled_op_idx, 
                last_op_processing_time
            )
        
        # Update the tracker to remember that the current operation is now 
        # the last one scheduled on this machine
        self.last_op_on_machine[machine_idx] = scheduled_op_idx
        
        # Also add assignment edges for completeness
        self._add_assignment_edges(scheduled_op_idx, machine_idx)
    
    def _add_machine_precedes_edge(self, src_op: int, dst_op: int, processing_time: float):
        """
        Add a single machine_precedes edge immediately to the graph.
        
        Args:
            src_op: Source operation ID
            dst_op: Destination operation ID  
            processing_time: Processing time of source operation (edge attribute)
        """
        # Create new edge tensors
        new_edge = torch.tensor([[src_op], [dst_op]], dtype=torch.long, device=self.device)
        new_attr = torch.tensor([[processing_time]], dtype=torch.float32, device=self.device)
        
        # Add edge immediately (edge type already initialized)
        self.hetero_data['op', 'machine_precedes', 'op'].edge_index = torch.cat(
            [self.hetero_data['op', 'machine_precedes', 'op'].edge_index, new_edge], dim=1
        )
        self.hetero_data['op', 'machine_precedes', 'op'].edge_attr = torch.cat(
            [self.hetero_data['op', 'machine_precedes', 'op'].edge_attr, new_attr], dim=0
        )
    
    def _add_assignment_edges(self, scheduled_op_idx: int, machine_idx: int):
        """
        Add assignment edges between operation and machine.
        
        Args:
            scheduled_op_idx: The operation that was just scheduled
            machine_idx: The machine it was assigned to
        """
        # Add the new assignment edge (edge type already initialized)
        current_edges = self.hetero_data['op', 'assigned_to', 'machine'].edge_index
        new_edge = torch.tensor([[scheduled_op_idx], [machine_idx]], dtype=torch.long, device=self.device)
        updated_edges = torch.cat([current_edges, new_edge], dim=1)
        self.hetero_data['op', 'assigned_to', 'machine'].edge_index = updated_edges
        
        # Also add reverse edge for symmetry (edge type already initialized)
        current_reverse_edges = self.hetero_data['machine', 'processes', 'op'].edge_index
        new_reverse_edge = torch.tensor([[machine_idx], [scheduled_op_idx]], dtype=torch.long, device=self.device)
        updated_reverse_edges = torch.cat([current_reverse_edges, new_reverse_edge], dim=1)
        self.hetero_data['machine', 'processes', 'op'].edge_index = updated_reverse_edges
