from src.sample_data import SSA_N8NN_SOURCE_URL, generate_sample_bpo_kpis


def test_sample_data_includes_public_benchmark_context_and_known_incidents(tmp_path):
    frame = generate_sample_bpo_kpis(tmp_path / "sample.csv")

    assert len(frame) == 75 * 48
    assert SSA_N8NN_SOURCE_URL.startswith("https://www.ssa.gov/")
    assert {"ssa_n8nn_month", "ssa_n8nn_calls_offered", "ssa_n8nn_busy_rate_pct"}.issubset(frame.columns)
    assert frame["agent_id"].nunique() == 48
    assert frame["team"].nunique() == 4

    early_aht = frame.loc[(frame["team"] == "Team Manila A") & (frame["date"] < "2026-06-06"), "aht"].mean()
    late_aht = frame.loc[(frame["team"] == "Team Manila A") & (frame["date"] >= "2026-06-06"), "aht"].mean()
    assert late_aht > early_aht

    agent_qa_before = frame.loc[(frame["agent_id"] == "A007") & (frame["date"] < "2026-06-08"), "qa"].mean()
    agent_qa_after = frame.loc[(frame["agent_id"] == "A007") & (frame["date"] >= "2026-06-08"), "qa"].mean()
    assert agent_qa_after < agent_qa_before
