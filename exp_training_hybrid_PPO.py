from typing import Optional

from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
from config import config
from RL.rl_env_continuius_idleness import RLEnvContinuousIdleness
from RL.flat_rl_trainer import HybridFlatRLTrainer
from RL.hierarchical_rl_trainer import HybridHierarchicalRLTrainer


def run_hybrid_training(project_name: Optional[str] = None, run_prefix: str = "hybrid"):
    sim = config.simulation_params
    # Get both flat and hierarchical RL parameters for hybrid training
    flat_config = config.get_flat_rl_config()
    hierarchical_config = config.get_hierarchical_rl_config()
    rl_flat = flat_config['rl_params']
    rl_hierarchical = hierarchical_config['rl_params']
    project = project_name or "exp_hrl_and_flarl"

    data_handler = FlexibleJobShopDataHandler(
        data_source={
            'num_jobs': sim['num_jobs'],
            'num_machines': sim['num_machines'],
            'operation_lb': sim['operation_lb'],
            'operation_ub': sim['operation_ub'],
            'processing_time_lb': sim['processing_time_lb'],
            'processing_time_ub': sim['processing_time_ub'],
            'compatible_machines_lb': sim['compatible_machines_lb'],
            'compatible_machines_ub': sim['compatible_machines_ub'],
            'TF': sim['TF'],
            'RDD': sim['RDD'],
            'seed': sim['seed'],
        },
        data_type='simulation',
        TF=sim['TF'],
        RDD=sim['RDD'],
        seed=sim['seed'],
    )

    env = RLEnvContinuousIdleness(
        data_handler,
        alpha=rl_flat['alpha'],  # Use common parameter from flat config
        use_reward_shaping=rl_flat['use_reward_shaping'],
    )

    flat_trainer = HybridFlatRLTrainer(
        env=env,
        epochs=rl_flat['epochs'],
        episodes_per_epoch=rl_flat['episodes_per_epoch'],
        train_per_episode=rl_flat['train_per_episode'],
        pi_lr=rl_flat['pi_lr'],
        v_lr=rl_flat['v_lr'],
        gamma=rl_flat['gamma'],
        gae_lambda=rl_flat['gae_lambda'],
        clip_ratio=rl_flat['clip_ratio'],
        entropy_coef=rl_flat['entropy_coef'],
        project_name=project,
        run_name=f"{run_prefix}_flat",
        device=rl_flat['device'],
        target_kl=rl_flat.get('target_kl', 0.01),
        max_grad_norm=rl_flat.get('max_grad_norm', 0.5),
        seed=rl_flat.get('seed', 42),
    )
    flat_results = flat_trainer.train()

    hrl_trainer = HybridHierarchicalRLTrainer(
        env=env,
        epochs=rl_hierarchical['epochs'],
        episodes_per_epoch=rl_hierarchical['episodes_per_epoch'],
        goal_duration_ratio=rl_hierarchical['goal_duration_ratio'],
        latent_dim=rl_hierarchical['latent_dim'],
        goal_dim=rl_hierarchical['goal_dim'],
        manager_lr=rl_hierarchical['manager_lr'],
        worker_lr=rl_hierarchical['worker_lr'],
        gamma_manager=rl_hierarchical['gamma_manager'],
        gamma_worker=rl_hierarchical['gamma_worker'],
        clip_ratio=rl_hierarchical['clip_ratio'],
        entropy_coef=rl_hierarchical['entropy_coef'],
        gae_lambda=rl_hierarchical['gae_lambda'],
        train_per_episode=rl_hierarchical['train_per_episode'],
        intrinsic_reward_scale=rl_hierarchical['intrinsic_reward_scale'],
        project_name=project,
        run_name=f"{run_prefix}_hrl",
        device=rl_hierarchical['device'],
        target_kl=rl_hierarchical.get('target_kl', 0.01),
        max_grad_norm=rl_hierarchical.get('max_grad_norm', 0.5),
        seed=rl_hierarchical.get('seed', 42),
    )
    hrl_results = hrl_trainer.train()

    return flat_results, hrl_results


def main():
    print("Remember to activate conda environment: conda activate dfjs")
    run_hybrid_training()


if __name__ == '__main__':
    main()
