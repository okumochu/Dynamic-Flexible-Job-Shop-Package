# Flexible Job Shop Data Handler

A comprehensive data handler for Flexible Job Shop Scheduling Problems (FJSP) that provides a unified interface for various optimization algorithms including Reinforcement Learning, MILP solvers, Genetic Algorithms, and more.

## Features

- **Multi-format Support**: Handles Brandimarte (mk01-mk15) and Hurink (abz, car, la, mt, orb) dataset formats
- **Unified Interface**: Single interface for RL, MILP, GA, and other algorithms
- **Rich Data Structures**: Efficient access to jobs, operations, and machine assignments
- **Statistics & Analysis**: Comprehensive problem statistics and lower bounds
- **Serialization**: JSON import/export capabilities
- **Batch Processing**: Load and compare multiple datasets

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from data_handler import FlexibleJobShopDataHandler

# Load a dataset
handler = FlexibleJobShopDataHandler("dataset/brandimarte/mk01.txt")

# Get basic information
print(f"Jobs: {handler.num_jobs}, Machines: {handler.num_machines}")
print(f"Operations: {handler.num_operations}")

# Get statistics
stats = handler.get_statistics()
print(f"Total processing time: {stats['total_processing_time']}")
print(f"Problem lower bound: {stats['problem_lower_bound']}")
```

## Dataset Formats

### Brandimarte Format (mk01-mk15)
```
10 6
6 2 0 5 2 4 3 4 3 2 5 1 1 2 2 4 5 2 3 5 5 1 6 0 1 1 2 1 3 5 6 2 6 3 3
5 1 1 6 1 2 1 1 0 2 2 1 6 3 6 3 5 5 1 6 0 1
...
```

### Hurink Format (abz, car, la, mt, orb series)
```
10 10
10 1 4 88 1 8 68 1 6 94 1 5 99 2 1 67 7 67 1 2 89 1 9 77 1 7 99 1 0 86 1 3 92
10 1 5 72 1 3 50 2 6 69 5 69 1 4 75 1 2 94 1 8 66 2 0 92 5 92 1 1 82 1 7 94 1 9 63
...
```

## Core Classes

### Operation
Represents a single operation in a job.

```python
@dataclass
class Operation:
    operation_id: int
    job_id: int
    machine_id: int
    processing_time: int
```

### Job
Represents a job with multiple operations.

```python
@dataclass
class Job:
    job_id: int
    operations: List[Operation]
```

### FlexibleJobShopDataHandler
Main class for handling FJSP datasets.

## Usage Examples

### For Reinforcement Learning

```python
# Get state representations
matrix_state = handler.get_state_representation("matrix")  # numpy array
vector_state = handler.get_state_representation("vector")  # flattened array
dict_state = handler.get_state_representation("dict")      # dictionary

# Access operations for decision making
job_operations = handler.get_job_operations(job_id=1)
machine_operations = handler.get_machine_operations(machine_id=1)

# Get processing times
processing_time = handler.get_processing_time(operation_id=0, machine_id=1)
```

### For MILP Solvers

```python
# Get MILP-compatible data
milp_data = handler.get_milp_data()

# Access data structures
num_jobs = milp_data["num_jobs"]
num_machines = milp_data["num_machines"]
processing_times = milp_data["processing_times"]  # 2D list
job_operations = milp_data["job_operations"]      # List of operation IDs per job
```

### For Genetic Algorithms

```python
# Get GA-compatible data
ga_data = handler.get_ga_data()

# Access data structures
job_sequences = ga_data["job_sequences"]          # Operation sequences per job
processing_times = ga_data["processing_times"]    # Processing time matrix
machine_assignments = ga_data["machine_assignments"]  # Machine assignments per job
```

### For Analysis and Statistics

```python
# Get comprehensive statistics
stats = handler.get_statistics()

# Access specific metrics
total_time = stats["total_processing_time"]
avg_ops_per_job = stats["avg_operations_per_job"]
machine_loads = stats["machine_loads"]
job_times = stats["job_processing_times"]
lower_bound = stats["problem_lower_bound"]

# Get theoretical bounds
job_lower_bound = handler.get_job_makespan_lower_bound(job_id=1)
problem_lower_bound = handler.get_problem_lower_bound()
```

### Batch Processing

```python
from data_handler import load_multiple_datasets, compare_datasets

# Load multiple datasets
datasets = load_multiple_datasets("dataset/brandimarte", "mk*.txt")

# Compare datasets
comparison = compare_datasets(datasets)

# Access individual datasets
mk01_handler = datasets["mk01.txt"]
mk02_handler = datasets["mk02.txt"]
```

### Serialization

```python
# Save to JSON
handler.save_to_json("problem_instance.json")

# Load from JSON
new_handler = FlexibleJobShopDataHandler()
new_handler.load_from_json("problem_instance.json")
```

## API Reference

### FlexibleJobShopDataHandler Methods

#### Core Methods
- `load_dataset(dataset_path: str)` - Load dataset from file
- `get_job_operations(job_id: int)` - Get operations for a job
- `get_machine_operations(machine_id: int)` - Get operations for a machine
- `get_operation(operation_id: int)` - Get specific operation
- `get_processing_time(operation_id: int, machine_id: int)` - Get processing time

#### Algorithm Interfaces
- `get_state_representation(format_type: str)` - Get RL state representation
- `get_milp_data()` - Get MILP-compatible data
- `get_ga_data()` - Get GA-compatible data

#### Analysis Methods
- `get_statistics()` - Get comprehensive statistics
- `get_total_processing_time()` - Get total processing time
- `get_machine_load(machine_id: int)` - Get machine load
- `get_problem_lower_bound()` - Get theoretical lower bound

#### Utility Methods
- `save_to_json(filepath: str)` - Save to JSON format
- `load_from_json(filepath: str)` - Load from JSON format
- `validate_solution(schedule: List[Tuple])` - Validate solution (placeholder)

### Utility Functions
- `load_multiple_datasets(dataset_dir: str, pattern: str)` - Load multiple datasets
- `compare_datasets(datasets: Dict)` - Compare multiple datasets

## Testing

Run the test suite to verify functionality:

```bash
cd benchmark
python test_data_handler.py
```

## Example Workflows

### RL Algorithm Integration
```python
# Initialize environment
handler = FlexibleJobShopDataHandler("dataset/brandimarte/mk01.txt")

# Get initial state
state = handler.get_state_representation("matrix")

# During training loop
for episode in range(num_episodes):
    # Get available operations
    available_ops = handler.get_job_operations(current_job)
    
    # Make decision
    action = agent.select_action(state, available_ops)
    
    # Get next state
    next_state = handler.get_state_representation("matrix")
    
    # Update state
    state = next_state
```

### MILP Solver Integration
```python
# Get problem data
handler = FlexibleJobShopDataHandler("dataset/brandimarte/mk01.txt")
milp_data = handler.get_milp_data()

# Use with solver (e.g., PuLP, Gurobi, CPLEX)
import pulp

# Create variables
x = pulp.LpVariable.dicts("x", 
    [(i, j, k) for i in range(milp_data["num_jobs"]) 
               for j in range(len(milp_data["job_operations"][i]))
               for k in range(milp_data["num_machines"])], 
    cat='Binary')

# Add constraints using milp_data
# ... solver implementation
```

### GA Integration
```python
# Get GA data
handler = FlexibleJobShopDataHandler("dataset/brandimarte/mk01.txt")
ga_data = handler.get_ga_data()

# Initialize population
population = []
for _ in range(population_size):
    # Create chromosome using job_sequences and machine_assignments
    chromosome = create_chromosome(ga_data["job_sequences"], 
                                 ga_data["machine_assignments"])
    population.append(chromosome)

# Fitness evaluation using processing_times
# ... GA implementation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 