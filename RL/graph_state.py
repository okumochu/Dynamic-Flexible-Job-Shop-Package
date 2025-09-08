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
    
    def __init__(self, problem_data: FlexibleJobShopDataHandler):
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
        
        # Build the heterogeneous graph
        self.hetero_data = self._build_initial_graph()
        self._update_ready_operations()
        # Update graph features to reflect the correct operation status after determining ready operations
        self._update_graph_features()
    
    def _build_initial_graph(self) -> HeteroData:
        """Build the initial heterogeneous graph structure."""
        data = HeteroData()
        
        # Build operation nodes and features
        op_features = self._build_operation_features()
        data['op'].x = torch.tensor(op_features, dtype=torch.float32)
        
        # Build machine nodes and features  
        machine_features = self._build_machine_features()
        data['machine'].x = torch.tensor(machine_features, dtype=torch.float32)
        
        # Build edges
        self._build_edges(data)
        
        return data
    
    def _build_operation_features(self) -> np.ndarray:
        """
        Build initial operation node features.
        
        Returns:
            Array of shape (num_operations, 9) with normalized features:
            [proc_time_mean, proc_time_std, proc_time_min, proc_time_max, 
             job_progress, status, completion_time, earliest_start_time,
             num_compatible_machines]
        """
        features = np.zeros((self.num_operations, 9))
        
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
            
            # Job progress
            job_id, op_position = self.problem_data.get_operation_info(op_id)
            total_ops_in_job = len(self.problem_data.get_job_operations(job_id))
            features[op_id, 4] = op_position / total_ops_in_job  # job_progress
            
            # Dynamic features (will be updated)
            features[op_id, 5] = self.operation_status[op_id]  # status (0: unscheduled, 1: scheduled)
            features[op_id, 6] = self.operation_completion_times[op_id] / max_due_date  # completion_time  
            features[op_id, 7] = self._get_earliest_start_time(op_id) / max_due_date  # earliest_start_time
            
            # Compatibility features
            features[op_id, 8] = len(compatible_machines) / self.num_machines  # num_compatible_machines (normalized)
            
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
        """Build all edge types in the heterogeneous graph."""
        
        # 1. Operation precedence edges ('op', 'precedes', 'op')
        precedence_edges = self._get_precedence_edges()
        if len(precedence_edges) > 0:
            data['op', 'precedes', 'op'].edge_index = torch.tensor(precedence_edges, dtype=torch.long).t()
        
        # 2. Operation-machine compatibility edges ('op', 'on_machine', 'machine') 
        # and ('machine', 'can_process', 'op')
        op_machine_edges, machine_op_edges = self._get_operation_machine_edges()
        
        if len(op_machine_edges) > 0:
            data['op', 'on_machine', 'machine'].edge_index = torch.tensor(op_machine_edges, dtype=torch.long).t()
        if len(machine_op_edges) > 0:
            data['machine', 'can_process', 'op'].edge_index = torch.tensor(machine_op_edges, dtype=torch.long).t()
    
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
        """Update dynamic features in the heterogeneous graph."""
        max_due_date = self.problem_data.get_max_due_date()
        max_workload = max(self.machine_workloads.max(), 1.0)
        
        # Update operation features (status, completion_time, earliest_start_time)
        for op_id in range(self.num_operations):
            self.hetero_data['op'].x[op_id, 5] = self.operation_status[op_id]  # status (0: unscheduled, 1: scheduled)
            self.hetero_data['op'].x[op_id, 6] = self.operation_completion_times[op_id] / max_due_date  # completion_time
            self.hetero_data['op'].x[op_id, 7] = self._get_earliest_start_time(op_id) / max_due_date  # earliest_start_time
        
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
        self.current_time = 0
        self.operation_status.fill(0)  # all unscheduled
        self.operation_completion_times.fill(0)
        self.machine_available_times.fill(0)
        self.machine_workloads.fill(0)
        self.ready_operations.clear()
        self.operation_machine_assignments.clear()
        self.last_op_on_machine = {m_id: None for m_id in range(self.num_machines)}
        
        # Rebuild the graph with fresh features
        self.hetero_data = self._build_initial_graph()
        self._update_ready_operations()
        # Update graph features to reflect the correct operation status after determining ready operations
        self._update_graph_features()
    
    def _update_dynamic_edges(self, scheduled_op_idx: int, machine_idx: int):
        """
        Dynamically add disjunctive arcs between operations on the same machine.
        
        This creates the critical sequential relationship between operations processed
        on the same machine, enabling the HGT to learn temporal dependencies.
        
        Args:
            scheduled_op_idx: The operation that was just scheduled
            machine_idx: The machine it was assigned to
        """
        # Find the last operation that was scheduled on this machine
        last_op_on_this_machine = self.last_op_on_machine[machine_idx]
        
        # If there was a previous operation, add a directed edge from it to the current one
        if last_op_on_this_machine is not None:
            edge_type = ('op', 'machine_precedes', 'op')
            
            # Initialize edge type if it doesn't exist
            if edge_type not in self.hetero_data.edge_types:
                self.hetero_data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
                # Initialize edge attributes for temporal information (processing times)
                self.hetero_data[edge_type].edge_attr = torch.empty((0, 1), dtype=torch.float32)
                
            # Get processing time of the last operation (this is the temporal delay ΔT)
            # Use the processing time on the machine where it was actually scheduled
            last_operation = self.problem_data.get_operation(last_op_on_this_machine)
            last_op_assigned_machine = self.operation_machine_assignments.get(last_op_on_this_machine, machine_idx)
            last_op_processing_time = last_operation.get_processing_time(last_op_assigned_machine)
            
            # Add the new disjunctive edge (last_op -> current_op)
            new_edge = torch.tensor([[last_op_on_this_machine], [scheduled_op_idx]], dtype=torch.long)
            new_edge_attr = torch.tensor([[last_op_processing_time]], dtype=torch.float32)
            
            self.hetero_data[edge_type].edge_index = torch.cat(
                [self.hetero_data[edge_type].edge_index, new_edge], dim=1
            )
            self.hetero_data[edge_type].edge_attr = torch.cat(
                [self.hetero_data[edge_type].edge_attr, new_edge_attr], dim=0
            )
        
        # CRITICAL: Update the tracker to remember that the current operation is now 
        # the last one scheduled on this machine
        self.last_op_on_machine[machine_idx] = scheduled_op_idx
        
        # Optional: Also add assignment edges for completeness
        # These are less critical now that we have the sequential relationships
        self._add_assignment_edges(scheduled_op_idx, machine_idx)
    
    def _add_assignment_edges(self, scheduled_op_idx: int, machine_idx: int):
        """
        Add assignment edges between operation and machine.
        
        Args:
            scheduled_op_idx: The operation that was just scheduled
            machine_idx: The machine it was assigned to
        """
        # Create assignment edge type if it doesn't exist
        if ('op', 'assigned_to', 'machine') not in self.hetero_data.edge_types:
            # Initialize empty assignment edges
            self.hetero_data['op', 'assigned_to', 'machine'].edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Add the new assignment edge
        current_edges = self.hetero_data['op', 'assigned_to', 'machine'].edge_index
        new_edge = torch.tensor([[scheduled_op_idx], [machine_idx]], dtype=torch.long)
        updated_edges = torch.cat([current_edges, new_edge], dim=1)
        self.hetero_data['op', 'assigned_to', 'machine'].edge_index = updated_edges
        
        # Also add reverse edge for symmetry
        if ('machine', 'processes', 'op') not in self.hetero_data.edge_types:
            self.hetero_data['machine', 'processes', 'op'].edge_index = torch.empty((2, 0), dtype=torch.long)
        
        current_reverse_edges = self.hetero_data['machine', 'processes', 'op'].edge_index
        new_reverse_edge = torch.tensor([[machine_idx], [scheduled_op_idx]], dtype=torch.long)
        updated_reverse_edges = torch.cat([current_reverse_edges, new_reverse_edge], dim=1)
        self.hetero_data['machine', 'processes', 'op'].edge_index = updated_reverse_edges
