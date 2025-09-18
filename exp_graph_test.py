import argparse
import os
import time
import torch
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

from RL.graph_state import GraphState
from benchmarks.data_handler import FlexibleJobShopDataHandler
from config import config


def visualize_and_save(data_path: str, data_type: str = 'simulation', out_dir: str = 'result/graph_viz'):
    # Build problem and graph state
    if data_type == 'simulation':
        sim_config = config.simulation_params
        problem_data = FlexibleJobShopDataHandler(data_source=sim_config, data_type='simulation')
    else:
        problem_data = FlexibleJobShopDataHandler(data_source=data_path, data_type='dataset')

    state = GraphState(problem_data)
    hetero_data = state.get_observation()

    # Optional validation (no prints)
    try:
        hetero_data.validate(raise_on_error=True)
    except Exception:
        pass

    # Convert to homogeneous for visualization
    homogeneous_data = hetero_data.to_homogeneous()

    # Build NetworkX graph
    G = to_networkx(homogeneous_data, to_undirected=False)

    # Color by node type
    node_colors = []
    node_type_map = homogeneous_data.node_type.cpu().numpy() if hasattr(homogeneous_data, 'node_type') else None
    color_map = {0: 'skyblue', 1: 'lightgreen', 2: 'salmon'}

    for node_idx in sorted(G.nodes()):
        if node_type_map is not None:
            node_colors.append(color_map.get(int(node_type_map[node_idx]), 'gray'))
        else:
            node_colors.append('gray')

    # Prepare output path
    os.makedirs(out_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, f'graph_{data_type}_{timestamp}.png')

    # Draw and save
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos=pos, with_labels=True, node_color=node_colors, node_size=500, font_size=8, arrows=True)
    plt.title('Homogeneous view of HeteroData')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

    # Print saved path only
    print(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save graph visualization to result/')
    parser.add_argument('--data-path', type=str, required=False, help='Path to dataset (required for dataset)')
    parser.add_argument('--data-type', type=str, default='simulation', choices=['dataset', 'simulation'])
    parser.add_argument('--out-dir', type=str, default='result/graph_viz', help='Output directory for images')
    args = parser.parse_args()

    if args.data_type == 'dataset' and not args.data_path:
        raise SystemExit('Error: --data-path is required when --data-type=dataset')

    visualize_and_save(args.data_path, args.data_type, args.out_dir)


