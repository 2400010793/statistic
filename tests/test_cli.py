import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from virtual_embryo_benchmark.cli import run_config
from virtual_embryo_benchmark import cli


def write_snapshot(path, shift=0, stage=8.5):
    data = ad.AnnData(
        np.asarray([[1, 2], [2, 3], [4, 1], [5, 2]], dtype=np.float32) + shift,
        obs=pd.DataFrame({"celltype": ["A", "A", "B", "B"], "stage": stage}),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    data.write_h5ad(path)


def test_run_writes_clean_h5ad_and_provenance(tmp_path):
    write_snapshot(tmp_path / "early.h5ad")
    write_snapshot(tmp_path / "late.h5ad", 1, 9.5)
    config = dict(
        task="T1", model="pseudobulk_shift", source="late.h5ad", history="early.h5ad",
        source_stage=9.5, history_stage=8.5, target_stage=10.5,
        output="artifacts/pred.h5ad", seed=5, n_cells=3,
    )
    record = run_config(config, root=tmp_path)
    prediction = ad.read_h5ad(tmp_path / config["output"])
    assert prediction.shape == (3, 2)
    assert "celltype" not in prediction.obs
    assert len(record["inputs"]) == 2
    assert len(record["output_sha256"]) == 64
    assert json.loads((tmp_path / "artifacts/pred.run.json").read_text())["config"] == config


def test_run_respects_board_panel_order_and_cell_limit(tmp_path):
    write_snapshot(tmp_path / "early.h5ad")
    panel = tmp_path / "data/panels"
    panel.mkdir(parents=True)
    (panel / "genes.txt").write_text("g2\ng1\n")
    (panel / "index.json").write_text(json.dumps({"demo": {"task": "T1", "genes_file": "genes.txt", "min_cells": 3, "max_cells": 4}}))
    config = dict(task="T1", model="copy_last", source="early.h5ad", source_stage=8.5, target_stage=9.5, output="artifacts/pred.h5ad", board="demo", n_cells=3)
    run_config(config, root=tmp_path)
    assert list(ad.read_h5ad(tmp_path / config["output"]).var_names) == ["g2", "g1"]
    config["n_cells"] = 2
    with pytest.raises(ValueError, match="at least 3"):
        run_config(config, root=tmp_path)


def test_run_does_not_tile_tiny_samples(tmp_path):
    write_snapshot(tmp_path / "early.h5ad")
    config = dict(task="T1", model="copy_last", source="early.h5ad", source_stage=8.5, target_stage=9.5, output="pred.h5ad", n_cells=1000)
    with pytest.raises(ValueError, match="do not tile"):
        run_config(config, root=tmp_path)


def test_demo_never_scores_future_prediction_against_training_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "download_all", lambda **kwargs: {"public_complete": True})
    monkeypatch.setattr(cli, "install_model", lambda *args: None)
    monkeypatch.setattr(cli, "run_config", lambda config, **kwargs: {"n_cells": 150})
    scored = []

    def score(**kwargs):
        scored.append(kwargs)
        return {"metrics": {"mmd_u": 0.1}}

    monkeypatch.setattr(cli, "score_files", score)
    summary = cli.demo(tmp_path, include_flow=True, include_papers=True)
    assert len(summary["results"]) == 11
    assert len(scored) == 6
    t1_scored = [item for item in scored if item["task"] == "T1"]
    assert len(t1_scored) == 1
    assert t1_scored[0]["input_path"].name == "T1-copy_last.h5ad"
    for result in summary["results"]:
        if result["task"] == "T1" and result["model"] != "copy_last":
            assert result["metrics"] is None
            metrics = json.loads((tmp_path / "artifacts/demo" / f"T1-{result['model']}.metrics.json").read_text())
            assert metrics["status"] == "target_unavailable"
