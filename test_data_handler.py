#!/usr/bin/env python3
"""
Test script for FlexibleJobShopDataHandler
Demonstrates usage with the mk01 dataset
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmark.data_handler import FlexibleJobShopDataHandler, load_multiple_datasets, compare_datasets
from two_stage_GA.env import get_ga_data
from MILP.env import get_milp_data
from hierarchical_rl.env import get_state_representation
from utils.io_utils import save_to_json, load_from_json


def test_basic_functionality():
    """Test basic functionality with mk01 dataset."""
    print("=== Testing Basic Functionality ===")
    
    # Load the mk01 dataset
    dataset_path = "benchmark/datasets/brandimarte/mk01.txt"
    handler = FlexibleJobShopDataHandler(dataset_path)
    
    print(f"Dataset loaded: {handler}")
    print(f"Dataset path: {handler.dataset_path}")
    
    # Print basic statistics
    stats = handler.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total processing time: {stats['total_processing_time']}")
    print(f"  Average operations per job: {stats['avg_operations_per_job']:.2f}")
    print(f"  Average processing time: {stats['avg_processing_time']:.2f}")
    print(f"  Problem lower bound: {stats['problem_lower_bound']}")
    
    # Print machine loads
    print(f"\nMachine loads:")
    for machine_id, load in stats['machine_loads'].items():
        print(f"  {machine_id}: {load}")
    
    # Print job processing times
    print(f"\nJob processing times:")
    for job_id, time in stats['job_processing_times'].items():
        print(f"  {job_id}: {time}")


def test_job_and_operation_access():
    """Test accessing jobs and operations."""
    print("\n=== Testing Job and Operation Access ===")
    
    handler = FlexibleJobShopDataHandler("benchmark/datasets/brandimarte/mk01.txt")
    
    # Test job access
    print(f"Number of jobs: {handler.num_jobs}")
    for job in handler.jobs[:3]:  # Show first 3 jobs
        print(f"  {job}")
        for op in job.operations:
            print(f"    {op}")
    
    # Test operation access
    print(f"\nNumber of operations: {handler.num_operations}")
    for i in range(min(5, handler.num_operations)):  # Show first 5 operations
        op = handler.get_operation(i)
        print(f"  Operation {i}: {op}")


def test_algorithm_interfaces():
    """Test interfaces for different algorithms."""
    print("\n=== Testing Algorithm Interfaces ===")
    
    handler = FlexibleJobShopDataHandler("benchmark/datasets/brandimarte/mk01.txt")
    
    # Test RL interface
    print("RL State Representations:")
    matrix_state = get_state_representation(handler, "matrix")
    print(f"  Matrix shape: {matrix_state.shape}")
    print(f"  Matrix sample: {matrix_state[:3, :3]}")
    vector_state = get_state_representation(handler, "vector")
    print(f"  Vector length: {len(vector_state)}")
    dict_state = get_state_representation(handler, "dict")
    print(f"  Dict keys: {list(dict_state.keys())}")
    
    # Test MILP interface
    print("\nMILP Data:")
    milp_data = get_milp_data(handler)
    print(f"  Keys: {list(milp_data.keys())}")
    print(f"  Processing times shape: {len(milp_data['processing_times'])} x {len(milp_data['processing_times'][0])}")
    
    # Test GA interface
    print("\nGA Data:")
    ga_data = get_ga_data(handler)
    print(f"  Keys: {list(ga_data.keys())}")
    print(f"  Job sequences: {ga_data['job_sequences'][:2]}")  # Show first 2 jobs


def test_multiple_datasets():
    """Test loading multiple datasets."""
    print("\n=== Testing Multiple Datasets ===")
    
    # Load all Brandimarte datasets
    brandimarte_dir = "benchmark/datasets/brandimarte"
    if os.path.exists(brandimarte_dir):
        datasets = load_multiple_datasets(brandimarte_dir, "mk*.txt")
        print(f"Loaded {len(datasets)} Brandimarte datasets")
        
        # Compare datasets
        comparison = compare_datasets(datasets)
        
        print("\nDataset Comparison:")
        for name, stats in comparison.items():
            print(f"  {name}: {stats['num_jobs']} jobs, {stats['num_machines']} machines, "
                  f"{stats['num_operations']} operations, LB={stats['problem_lower_bound']}")
    
    # Load some Hurink datasets
    hurink_dir = "benchmark/datasets/hurink/edata"
    if os.path.exists(hurink_dir):
        datasets = load_multiple_datasets(hurink_dir, "*.txt")
        print(f"\nLoaded {len(datasets)} Hurink datasets")
        
        # Show first few
        for i, (name, handler) in enumerate(datasets.items()):
            if i < 3:  # Show first 3
                print(f"  {name}: {handler}")


def test_json_serialization():
    """Test JSON serialization and deserialization."""
    print("\n=== Testing JSON Serialization ===")
    
    handler = FlexibleJobShopDataHandler("benchmark/datasets/brandimarte/mk01.txt")
    
    # Save to JSON
    json_path = "test_mk01.json"
    save_to_json(handler, json_path)
    print(f"Saved to {json_path}")
    
    # Load from JSON
    new_handler = load_from_json(json_path)
    print(f"Loaded from {json_path}")
    
    # Compare
    original_stats = handler.get_statistics()
    new_stats = new_handler.get_statistics()
    
    print(f"Original total processing time: {original_stats['total_processing_time']}")
    print(f"New total processing time: {new_stats['total_processing_time']}")
    print(f"Match: {original_stats['total_processing_time'] == new_stats['total_processing_time']}")
    
    # Clean up
    if os.path.exists(json_path):
        os.remove(json_path)


def main():
    """Run all tests."""
    print("Flexible Job Shop Data Handler Test Suite")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_job_and_operation_access()
        test_algorithm_interfaces()
        test_multiple_datasets()
        test_json_serialization()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 