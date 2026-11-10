from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from explorer.envs.explore_env import ExploreEnv
from explorer.envs.wrappers import ObservationMemoryWrapper, TrainMapSampler
from explorer.train import (
    DungeonEvalCallback,
    check_checkpoint_compatibility,
    make_vec_env_dungeon,
    train_multirun,
    train_one,
)
from explorer.utils.config import get_split, validate_seed_splits
from explorer.utils.seeding import (
    atomic_save_json,
    atomic_save_model,
    build_run_rngs,
    compute_config_hash,
    get_run_seed_int,
    make_run_seeds,
)


class TestSeedingArchitecture:
    def test_make_run_seeds_prefix_property_and_determinism(self) -> None:
        """Requirement 2: make_run_seeds(g, 3) must be exact prefix of make_run_seeds(g, 5)."""
        global_seed = 2026
        seeds_3 = make_run_seeds(global_seed, 3)
        seeds_5 = make_run_seeds(global_seed, 5)

        int_seeds_3 = [get_run_seed_int(s) for s in seeds_3]
        int_seeds_5 = [get_run_seed_int(s) for s in seeds_5]

        assert len(int_seeds_3) == 3
        assert len(int_seeds_5) == 5
        assert int_seeds_5[:3] == int_seeds_3

        # Determinism across multiple calls
        seeds_3_bis = make_run_seeds(global_seed, 3)
        assert [get_run_seed_int(s) for s in seeds_3_bis] == int_seeds_3

        # Different global seed yields different run seeds
        seeds_other = make_run_seeds(9999, 3)
        assert [get_run_seed_int(s) for s in seeds_other] != int_seeds_3

    def test_build_run_rngs_stateless(self) -> None:
        """Requirement 3: build_run_rngs must be stateless and reproducible across calls."""
        run_ss = make_run_seeds(2026, 1)[0]
        n_envs = 4

        # Calling twice with same arguments returns identical outputs
        algo1, map_sss1 = build_run_rngs(run_ss, n_envs, resume_count=0)
        algo2, map_sss2 = build_run_rngs(run_ss, n_envs, resume_count=0)

        assert algo1 == algo2
        assert len(map_sss1) == n_envs
        assert [get_run_seed_int(s) for s in map_sss1] == [
            get_run_seed_int(s) for s in map_sss2
        ]

        # Different resume_count yields different algorithmic and map seeds
        algo_res1, map_res1 = build_run_rngs(run_ss, n_envs, resume_count=1)
        assert algo_res1 != algo1
        assert [get_run_seed_int(s) for s in map_res1] != [
            get_run_seed_int(s) for s in map_sss1
        ]


class TestTrainMapSampler:
    def test_sampler_boundary_invariants(self) -> None:
        """Requirement 5: pool == range(0, 100000), pool.stop <= min(val, test, dev), all drawn in [0, 99999]."""
        train_pool = get_split("train")
        val_pool = get_split("val")
        test_pool = get_split("test")
        dev_pool = get_split("dev")

        assert train_pool == range(0, 100000)
        assert train_pool.stop <= val_pool.start
        assert train_pool.stop <= test_pool.start
        assert train_pool.stop <= dev_pool.start

    def test_sampler_ignores_external_seed_and_logs_audit(self, tmp_path: Path) -> None:
        """TrainMapSampler ignores caller seeds, samples from pool, and writes JSONL audit."""
        train_pool = get_split("train")
        ss = np.random.SeedSequence(42)
        rng = np.random.default_rng(ss)
        log_file = tmp_path / "worker_0.jsonl"

        base_env = ExploreEnv(difficulty="easy")
        wrapped = TrainMapSampler(
            base_env,
            seed_pool=train_pool,
            rng=rng,
            log_file=log_file,
            resume_count=0,
        )

        sampled_seeds = []
        for _ in range(10):
            # Pass external seed that should be ignored
            obs, info = wrapped.reset(seed=9999999)
            assert "map_seed" in info
            map_seed = info["map_seed"]
            assert map_seed in train_pool
            assert map_seed != 9999999
            sampled_seeds.append(map_seed)

        wrapped.close()

        # Verify audit file
        assert log_file.exists()
        with open(log_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 10
        for i, entry in enumerate(lines, start=1):
            assert entry["episode"] == i
            assert entry["map_seed"] in train_pool
            assert entry["resume_count"] == 0

    def test_sampler_rejects_out_of_pool_seed(self) -> None:
        """Requirement 5: Enforce boundary check if map_seed not in seed_pool."""
        base_env = ExploreEnv(difficulty="easy")

        class MockRNG:
            def integers(self, low: int, high: int) -> int:
                return 9999999  # Out of pool

        wrapped = TrainMapSampler(
            base_env,
            seed_pool=range(0, 100),
            rng=MockRNG(),  # type: ignore
        )

        with pytest.raises(ValueError, match="outside authorized seed_pool"):
            wrapped.reset()

        wrapped.close()


class TestSubprocVecEnvAuditTrail:
    def test_subproc_worker_audit_files(self, tmp_path: Path) -> None:
        """Requirement B: Verify map_seeds/worker_{k}.jsonl generated correctly in parallel."""
        run_ss = make_run_seeds(12345, 1)[0]
        n_envs = 2
        algo_seed, map_sss = build_run_rngs(run_ss, n_envs, resume_count=0)
        log_dir = tmp_path / "map_seeds"

        venv = make_vec_env_dungeon(
            difficulty="easy",
            n_envs=n_envs,
            map_sss=map_sss,
            log_dir=log_dir,
            resume_count=0,
            vec_env_type="subproc",
        )

        venv.reset()
        for _ in range(20):
            venv.step([1] * n_envs)
        venv.close()

        # Both workers should have created log files
        w0_log = log_dir / "worker_0.jsonl"
        w1_log = log_dir / "worker_1.jsonl"
        assert w0_log.exists()
        assert w1_log.exists()

        with open(w0_log, "r", encoding="utf-8") as f:
            w0_lines = [json.loads(line) for line in f]
        assert len(w0_lines) >= 1
        assert all(line["resume_count"] == 0 for line in w0_lines)
        assert all(line["map_seed"] in get_split("train") for line in w0_lines)


class TestValidationGuards:
    def test_eval_callback_guard_exceeds_val_seeds(self) -> None:
        """Guard n_eval_episodes <= len(val_seeds) must raise ValueError."""
        eval_env = ExploreEnv(difficulty="easy")
        with pytest.raises(ValueError, match="exceeds available validation seeds"):
            DungeonEvalCallback(
                eval_env=eval_env,
                eval_freq=10,
                n_eval_episodes=50,
                val_seeds=[1, 2, 3],  # Only 3 seeds provided for 50 episodes
            )


class TestResumeAndCheckpointGuards:
    def test_run_manifest_allows_seed_count_expansion(self, tmp_path: Path) -> None:
        """Requirement 2: Manifest guard excludes seed_count from equality and allows expansion."""
        exp_dir = tmp_path / "checkpoints" / "test_exp"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 1. Run with seed_count=1
        train_multirun(
            algo="ppo",
            difficulty="easy",
            seed=2026,
            seed_count=1,
            total_timesteps=16,
            n_envs=1,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            eval_freq=16,
            n_eval_episodes=1,
            exp_name="test_exp",
            output_dir=tmp_path,
            verbose=0,
        )

        manifest_file = exp_dir / "run_manifest.json"
        assert manifest_file.exists()
        with open(manifest_file, "r") as f:
            man1 = json.load(f)
        assert man1["seed_count"] == 1
        assert len(man1["seeds"]) == 1

        # 2. Expand to seed_count=2 without error
        train_multirun(
            algo="ppo",
            difficulty="easy",
            seed=2026,
            seed_count=2,
            total_timesteps=16,
            n_envs=1,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            eval_freq=16,
            n_eval_episodes=1,
            exp_name="test_exp",
            output_dir=tmp_path,
            verbose=0,
        )

        with open(manifest_file, "r") as f:
            man2 = json.load(f)
        assert man2["seed_count"] == 2
        assert len(man2["seeds"]) == 2
        assert man2["seeds"][:1] == man1["seeds"]

    def test_resume_preserves_best_score_and_step_accounting(self, tmp_path: Path) -> None:
        """Requirements 1 & 9: Interrupt training via exception, resume, verify step count and best_score preservation."""
        run_dir = tmp_path / "checkpoints" / "interrupt_exp" / "seed_0"
        exp_dir = tmp_path / "checkpoints" / "interrupt_exp"
        run_ss = make_run_seeds(2026, 1)[0]

        class InterruptCallback(BaseCallback):
            def __init__(self, interrupt_step: int = 32):
                super().__init__()
                self.interrupt_step = interrupt_step

            def _on_step(self) -> bool:
                if self.num_timesteps >= self.interrupt_step:
                    raise KeyboardInterrupt("Simulated training interruption")
                return True

        # Phase 1: Train to 32 steps and interrupt
        try:
            train_one(
                run_ss=run_ss,
                run_idx=0,
                run_dir=run_dir,
                exp_dir=exp_dir,
                total_timesteps=64,
                n_envs=1,
                n_steps=16,
                batch_size=16,
                n_epochs=1,
                eval_freq=16,
                n_eval_episodes=1,
                extra_callbacks=[InterruptCallback(32)],
                verbose=0,
            )
        except (KeyboardInterrupt, Exception):
            pass

        assert not (run_dir / "DONE").exists()
        assert (run_dir / "latest_model.zip").exists()
        assert (run_dir / "run_state.json").exists()

        # Intentionally record a high best_score in run_state.json to test preservation
        with open(run_dir / "run_state.json", "r") as f:
            st = json.load(f)
        st["best_score"] = 0.99
        atomic_save_json(run_dir / "run_state.json", st)

        # Touch best_model.zip with sentinel marker
        sentinel_best = run_dir / "best_model.zip"
        sentinel_mtime = sentinel_best.stat().st_mtime if sentinel_best.exists() else None

        # Phase 2: Resume training to complete all 64 steps
        model, cov, history = train_one(
            run_ss=run_ss,
            run_idx=0,
            run_dir=run_dir,
            exp_dir=exp_dir,
            total_timesteps=64,
            n_envs=1,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            eval_freq=16,
            n_eval_episodes=1,
            resume=True,
            verbose=0,
        )

        assert (run_dir / "DONE").exists()
        assert model.num_timesteps == 64

        # Verify best_score (0.99) was preserved in run_state and not overwritten by a worse eval
        with open(run_dir / "run_state.json", "r") as f:
            final_st = json.load(f)
        assert final_st["best_score"] >= 0.99

    def test_resume_empty_dir_trains_from_scratch(self, tmp_path: Path) -> None:
        """Requirement 6: --resume on a directory without checkpoints trains from step 0 fresh."""
        run_dir = tmp_path / "checkpoints" / "empty_exp" / "seed_0"
        exp_dir = tmp_path / "checkpoints" / "empty_exp"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_ss = make_run_seeds(2026, 1)[0]

        model, cov, history = train_one(
            run_ss=run_ss,
            run_idx=0,
            run_dir=run_dir,
            exp_dir=exp_dir,
            total_timesteps=32,
            n_envs=1,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            eval_freq=16,
            n_eval_episodes=1,
            resume=True,
            verbose=0,
        )

        assert (run_dir / "DONE").exists()
        assert model.num_timesteps == 32

    def test_restart_archives_incomplete_dir(self, tmp_path: Path) -> None:
        """--restart archives incomplete run dir to seed_i.incomplete_<timestamp> and starts fresh."""
        run_dir = tmp_path / "checkpoints" / "restart_exp" / "seed_0"
        exp_dir = tmp_path / "checkpoints" / "restart_exp"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "dummy.txt").write_text("partial data")
        run_ss = make_run_seeds(2026, 1)[0]

        model, cov, history = train_one(
            run_ss=run_ss,
            run_idx=0,
            run_dir=run_dir,
            exp_dir=exp_dir,
            total_timesteps=32,
            n_envs=1,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
            eval_freq=16,
            n_eval_episodes=1,
            restart=True,
            verbose=0,
        )

        assert (run_dir / "DONE").exists()
        # Verify an archived directory was created
        archived = list(exp_dir.glob("seed_0.incomplete_*"))
        assert len(archived) == 1
        assert (archived[0] / "dummy.txt").exists()

    def test_init_from_algo_compatibility(self, tmp_path: Path) -> None:
        """Requirement 10: --init-from validates algorithm class compatibility."""
        env = ExploreEnv(difficulty="easy")
        w_env = ObservationMemoryWrapper(env)
        p_model = PPO("MultiInputPolicy", w_env, n_steps=16, batch_size=16, n_epochs=1)
        p_path = tmp_path / "ppo_model.zip"
        p_model.save(str(p_path))

        # PPO to PPO is compatible
        check_checkpoint_compatibility(p_path, expected_algo="ppo")

        # PPO to RecurrentPPO must raise ValueError
        with pytest.raises(ValueError, match="Algorithm mismatch"):
            check_checkpoint_compatibility(p_path, expected_algo="recurrent_ppo")

        w_env.close()
