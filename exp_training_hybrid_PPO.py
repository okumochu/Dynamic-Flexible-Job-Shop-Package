from typing import Optional

from benchmarks.static_benchmark.data_handler import FlexibleJobShopDataHandler
from config import config
from RL.rl_env_continuius_idleness import RLEnvContinuousIdleness
from RL.flat_rl_trainer import HybridFlatRLTrainer
from RL.hierarchical_rl_trainer import HybridHierarchicalRLTrainer


def run_hybrid_training(project_name: Optional[str] = None, run_prefix: str = "hybrid"):
    sim = config.simulation_params
    rl = config.rl_params
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
        alpha=rl['alpha'],
        use_reward_shaping=rl['use_reward_shaping'],
    )

    flat_trainer = HybridFlatRLTrainer(
        env=env,
        epochs=rl['epochs'],
        episodes_per_epoch=rl['episodes_per_epoch'],
        train_per_episode=rl['train_per_episode'],
        pi_lr=rl['pi_lr'],
        v_lr=rl['v_lr'],
        gamma=rl['gamma'],
        gae_lambda=rl['gae_lambda'],
        clip_ratio=rl['clip_ratio'],
        entropy_coef=rl['entropy_coef'],
        project_name=project,
        run_name=f"{run_prefix}_flat",
        device=rl['device'],
        target_kl=rl.get('target_kl', 0.01),
        max_grad_norm=rl.get('max_grad_norm', 0.5),
        seed=rl.get('seed', 42),
    )
    flat_results = flat_trainer.train()

    hrl_trainer = HybridHierarchicalRLTrainer(
        env=env,
        epochs=rl['epochs'],
        episodes_per_epoch=rl['episodes_per_epoch'],
        goal_duration_ratio=rl['goal_duration_ratio'],
        latent_dim=rl['latent_dim'],
        goal_dim=rl['goal_dim'],
        manager_lr=rl['manager_lr'],
        worker_lr=rl['worker_lr'],
        gamma_manager=rl['gamma_manager'],
        gamma_worker=rl['gamma_worker'],
        clip_ratio=rl['clip_ratio'],
        entropy_coef=rl['entropy_coef'],
        gae_lambda=rl['gae_lambda'],
        train_per_episode=rl['train_per_episode'],
        intrinsic_reward_scale=rl['intrinsic_reward_scale'],
        project_name=project,
        run_name=f"{run_prefix}_hrl",
        device=rl['device'],
        target_kl=rl.get('target_kl', 0.01),
        max_grad_norm=rl.get('max_grad_norm', 0.5),
        seed=rl.get('seed', 42),
    )
    hrl_results = hrl_trainer.train()

    return flat_results, hrl_results


def main():
    print("Remember to activate conda environment: conda activate dfjs")
    run_hybrid_training()


if __name__ == '__main__':
    main()
