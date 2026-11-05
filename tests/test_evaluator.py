from __future__ import annotations

from pathlib import Path

from explorer.agents.random_agent import RandomAgent
from explorer.eval.evaluator import (
    EvalResult,
    detect_use_memory,
    evaluate_single_agent,
    format_results_markdown,
    format_results_terminal,
    save_evaluation_report,
)


def test_eval_result_to_dict():
    """Verify EvalResult serializes to dictionary cleanly."""
    res = EvalResult(
        agent_name="TestAgent",
        difficulty="easy",
        num_episodes=5,
        coverage_mean=85.0,
        coverage_std=5.0,
        coverage_median=86.0,
        coverage_min=75.0,
        coverage_max=92.0,
        coverage_q25=80.0,
        coverage_q75=90.0,
        completion_rate=60.0,
        death_rate=0.0,
        timeout_rate=40.0,
        visited_path_mean=20.0,
        poi_discovery_mean=100.0,
        mean_ep_length=150.0,
        mean_reward=1.5,
        reward_breakdown={"reveal": 1.2, "poi": 0.5},
        end_reasons={"coverage": 3, "timeout": 2},
        fps=1200.0,
        latency_ms=0.83,
        raw_coverages=[75.0, 80.0, 86.0, 90.0, 92.0],
        raw_rewards=[1.0, 1.2, 1.5, 1.8, 2.0],
        raw_lengths=[120, 140, 150, 160, 180],
    )
    d = res.to_dict()
    assert d["agent_name"] == "TestAgent"
    assert d["coverage_mean"] == 85.0
    assert len(d["raw_coverages"]) == 5


def test_evaluate_single_agent_random():
    """Verify evaluate_single_agent runs end-to-end on RandomAgent with full metrics and non-zero reward."""
    res = evaluate_single_agent(
        agent_or_factory=lambda env, s: RandomAgent(seed=s),
        agent_name="RandomTest",
        difficulty="easy",
        seeds=[100, 101],
        verbose=False,
    )
    assert res.num_episodes == 2
    assert 0.0 <= res.coverage_mean <= 100.0
    assert len(res.raw_coverages) == 2
    assert res.fps > 0.0
    # Verify reward calculation is non-zero
    assert res.mean_reward != 0.0
    assert len(res.raw_rewards) == 2
    # Verify new Ather Labs required metrics are aggregated
    assert res.wall_collisions_mean >= 0.0
    assert 0.0 <= res.bump_rate <= 100.0
    assert 0.0 <= res.loop_rate <= 100.0
    assert res.revealed_efficiency >= 0.0
    assert res.visited_efficiency >= 0.0
    assert res.monster_encounters_mean >= 0.0
    assert "25%" in res.coverage_milestones
    assert "100%" in res.coverage_milestones


def test_format_and_save_report(tmp_path: Path):
    """Verify markdown and terminal formatting and file export."""
    res = EvalResult(
        agent_name="AgentA",
        difficulty="medium",
        num_episodes=2,
        coverage_mean=88.5,
        coverage_std=2.1,
        coverage_median=88.5,
        coverage_min=86.4,
        coverage_max=90.6,
        coverage_q25=87.0,
        coverage_q75=90.0,
        completion_rate=50.0,
        death_rate=0.0,
        timeout_rate=50.0,
        visited_path_mean=25.0,
        poi_discovery_mean=50.0,
        mean_ep_length=220.0,
        mean_reward=2.5,
        reward_breakdown={},
        end_reasons={"coverage": 1, "timeout": 1},
        fps=800.0,
        latency_ms=1.25,
        raw_coverages=[86.4, 90.6],
        raw_rewards=[2.0, 3.0],
        raw_lengths=[200, 240],
    )
    md_str = format_results_markdown([res])
    assert "| **AgentA** |" in md_str
    assert "88.50 ± 2.10" in md_str

    term_str = format_results_terminal([res])
    assert "AgentA" in term_str

    j_path, m_path = save_evaluation_report([res], tmp_path, "test_prefix")
    assert j_path.exists()
    assert m_path.exists()
    assert m_path.stat().st_size > 50


def test_detect_use_memory():
    """Verify detect_use_memory default behavior."""
    rnd = RandomAgent()
    assert detect_use_memory(rnd) is True

