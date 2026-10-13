"""One CLI for acquisition, reproducible baselines and official raw metrics."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .acquisition import download_all, install_model, load_catalog, model_registry, relative_path
from .io import check_data, clean_prediction, read_data, sha256, subset_panel, write_json


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def finite_json(value):
    """Represent scorer NaNs explicitly, never silently drop metrics."""
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def versions():
    names = ["virtual-embryo-benchmark", "numpy", "scipy", "anndata", "scikit-learn", "veckit", "torch", "geomloss", "POT"]
    result = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def run_config(config, *, root="."):
    from .models import predict_t1, predict_t2, predict_t3

    task = config["task"]
    model = config["model"]
    supported = model_registry()["local"]
    if model not in supported or task not in supported[model]:
        raise ValueError(f"Model {model} does not support task {task}")
    seed = int(config.get("seed", 0))
    n_cells = config.get("n_cells", 2000)
    max_train = int(config.get("max_train_cells", 2000))
    output_path = relative_path(root, config["output"])
    source_path = Path(root) / config["source"]
    # Estimate shifts from up to max_train rows; do not estimate type means on
    # just the output subset. Deterministic and explicit training/output budgets.
    source = read_data(str(source_path), max_cells=max_train, seed=seed)
    board = config.get("board")
    contract = None
    panel = None
    if board:
        index_path = Path(root) / "data/panels/index.json"
        if not index_path.exists():
            raise ValueError("Missing official panel index; run download --public-only")
        index = json.loads(index_path.read_text())
        if board not in index:
            raise ValueError("Unknown board; download the current official panel index")
        contract = index[board]
        if contract["task"] != task:
            raise ValueError("Board task differs from run task")
        panel = (index_path.parent / contract["genes_file"]).read_text().splitlines()
        source = subset_panel(source, panel)
    histories = {}
    provenance = [{"role": "source", "path": str(source_path), "sha256": sha256(source_path)}]
    for role in ["history", "observed_wt", "observed_perturbed"]:
        if config.get(role):
            path = Path(root) / config[role]
            data = read_data(str(path), max_cells=max_train, seed=seed)
            histories[role] = subset_panel(data, panel) if panel else data
            provenance.append({"role": role, "path": str(path), "sha256": sha256(path)})
    started = time.monotonic()
    checkpoints = []
    if model in {"prescient", "wot_linear"}:
        if "history" not in histories:
            raise ValueError(f"{model} requires history")
        if model == "prescient":
            from .prescient_adapter import predict_prescient

            checkpoint = output_path.with_suffix(".model.pt")
            prescient_keys = ["hidden_dim", "batch_size", "dt", "noise_sd", "learning_rate", "sinkhorn_blur", "sinkhorn_scaling"]
            prediction = predict_prescient(
                source, histories["history"], root=root,
                source_stage=config["source_stage"], history_stage=config["history_stage"],
                target_stage=config["target_stage"], seed=seed,
                epochs=int(config.get("epochs", 20)), n_components=int(config.get("n_components", 30)),
                device=config.get("device", "cpu"),
                checkpoint_path=checkpoint,
                **{key: config[key] for key in prescient_keys if key in config},
            )
        else:
            from .wot_adapter import predict_wot

            checkpoint = output_path.with_suffix(".transport.h5ad")
            prediction = predict_wot(
                source, histories["history"], root=root,
                source_stage=config["source_stage"], history_stage=config["history_stage"],
                target_stage=config["target_stage"], seed=seed,
                n_components=int(config.get("n_components", 30)),
                epsilon=float(config.get("epsilon", 0.1)), n_iter=int(config.get("n_iter", 300)),
                checkpoint_path=checkpoint,
            )
        checkpoints.append({"path": str(checkpoint), "sha256": sha256(checkpoint)})
    elif model in {"ot_linear", "ot_flow"}:
        from .dynamics import predict_ot

        if "history" not in histories:
            raise ValueError(f"{model} requires history")
        dynamics_keys = ["n_components", "epsilon", "sinkhorn_iterations", "epochs", "hidden_dim", "batch_size", "integration_steps", "device"]
        prediction = predict_ot(
            source, histories["history"], model=model, task=task,
            source_stage=config["source_stage"], history_stage=config["history_stage"],
            target_stage=config["target_stage"], seed=seed,
            **{key: config[key] for key in dynamics_keys if key in config},
        )
    elif task in {"T1", "T2"}:
        predictor = predict_t1 if task == "T1" else predict_t2
        prediction = predictor(
            source, method=model, history=histories.get("history"),
            source_stage=config["source_stage"], history_stage=config.get("history_stage"),
            target_stage=config["target_stage"], celltype_key=config.get("celltype_key", "celltype"),
            min_cells=int(config.get("min_cells_per_type", 1)),
            scale_by_time=bool(config.get("scale_by_time", False)),
        )
    else:
        prediction = predict_t3(
            source, method=model, observed_wt=histories.get("observed_wt"),
            observed_perturbed=histories.get("observed_perturbed"),
            stage=config["target_stage"], observed_stage=config.get("observed_stage"),
            ko_genes=config.get("ko_genes"),
        )
    if n_cells is not None:
        import numpy as np

        count = int(n_cells)
        if count < 1:
            raise ValueError("n_cells must be positive")
        # No synthetic increase in sample size: reject oversampling rather than
        # tile a small demo into apparent competition-sized evidence.
        if count > prediction.n_obs:
            raise ValueError("n_cells exceeds available model output; do not tile demo cells")
        if count < prediction.n_obs:
            selected = np.sort(np.random.default_rng(seed).choice(prediction.n_obs, count, replace=False))
            prediction = prediction[selected].copy()
    check_data(
        prediction, spatial=task in {"T2", "T3"}, panel=panel,
        min_cells=contract["min_cells"] if contract else None,
        max_cells=contract["max_cells"] if contract else None,
    )
    prediction = clean_prediction(prediction, task=task)
    prediction.uns["run"] = {"task": task, "model": model, "seed": seed}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_h5ad(output_path, compression="gzip")
    record = {
        "task": task, "model": model, "config": config,
        "inputs": provenance, "output": str(output_path),
        "output_sha256": sha256(output_path), "n_cells": prediction.n_obs,
        "n_genes": prediction.n_vars, "runtime_seconds": round(time.monotonic() - started, 3),
        "versions": versions(), "status": "prediction_written_not_officially_submitted",
        "checkpoints": checkpoints,
    }
    write_json(output_path.with_suffix(".run.json"), record)
    return record


def score_files(*, task, input_path, target, reference=None, wt=None, setting="heart", seed=0, max_cells=2000, out=None):
    from tempfile import TemporaryDirectory
    from veckit import score

    if max_cells < 31:
        raise ValueError("Official 30-PC metrics require max_cells >=31")
    if task in {"T1", "T2"} and not reference:
        raise ValueError("Explicit --reference is required for meaningful change metrics")
    if task == "T3" and not wt:
        raise ValueError("T3 requires the matched same-stage --wt")
    original = {"input": input_path, "target": target}
    if reference:
        original["reference"] = reference
    if wt:
        original["wt"] = wt
    start = time.monotonic()
    with TemporaryDirectory(prefix="embryo-score-") as directory:
        arguments = {"task": task, "setting": setting, "seed": seed}
        for role, path in original.items():
            # Public scorer densifies whole files; bound memory before invoking.
            data = read_data(str(path), max_cells=max_cells, seed=seed)
            if data.n_obs < 31:
                raise ValueError("At least 31 observed cells required for this scorer")
            if role == "target":
                if "celltype" not in data.obs or data.obs["celltype"].isna().any():
                    raise ValueError("Local official scorer needs complete target obs['celltype'] labels")
                if data.obs["celltype"].nunique() < 2:
                    raise ValueError("Frozen target classifier requires at least two observed cell types")
            check_data(data, spatial=task != "T1")
            temporary = Path(directory) / (role + ".h5ad")
            data.write_h5ad(temporary)
            arguments[role] = str(temporary)
        result = finite_json(score(**arguments))
    result["benchmark"] = {
        "meaning": "Local raw metrics on user-supplied target; not official leaderboard score",
        "max_cells": max_cells, "seed": seed, "paths": {k: str(v) for k,v in original.items()},
        "file_sha256": {k: sha256(Path(v)) for k,v in original.items()},
        "runtime_seconds": round(time.monotonic() - start, 3), "versions": versions(),
    }
    # Temporary paths are not useful provenance once the context closes.
    result["meta"].update({key: str(value) for key, value in original.items()})
    result["meta"]["target_source"] = str(target)
    if out:
        write_json(Path(out), result)
    return result


def demo(root, *, include_flow=False, include_papers=False, epochs=20, max_cells=150):
    report = download_all(root=root, public_only=True)
    if not report["public_complete"]:
        raise ValueError("Public downloads incomplete; rerun download before demo")
    dest = "artifacts/demo"
    sample = "data/samples/"
    sample_first = sample + "sample_8.5.h5ad"
    sample_last = sample + "sample_9.5.h5ad"
    cases = [
        {"task":"T1","model":"copy_last","source":sample_first,"source_stage":8.5,"target_stage":9.5},
        {"task":"T1","model":"pseudobulk_shift","source":sample_last,"history":sample_first,"source_stage":9.5,"history_stage":8.5,"target_stage":10.5},
        {"task":"T1","model":"ot_linear","source":sample_last,"history":sample_first,"source_stage":9.5,"history_stage":8.5,"target_stage":10.5},
        {"task":"T2","model":"copy_last","source":sample+"sample_heart_9.25.h5ad","source_stage":9.25,"target_stage":9.5},
        {"task":"T3","model":"wt_identity","source":sample+"sample_wt.h5ad","target_stage":9.5},
        {"task":"T3","model":"shift_transfer","source":sample+"sample_wt.h5ad","target_stage":9.5,"observed_stage":9.5,"observed_wt":sample+"sample_wt.h5ad","observed_perturbed":sample+"sample_mab21l2_ko.h5ad"},
        {"task":"T3","model":"gene_ko","source":sample+"sample_wt.h5ad","target_stage":9.5,"observed_stage":9.5,"observed_wt":sample+"sample_wt.h5ad","observed_perturbed":sample+"sample_mab21l2_ko.h5ad","ko_genes":["Mab21l2"]},
        {"task":"T3","model":"gene_zero","source":sample+"sample_wt.h5ad","target_stage":9.5,"ko_genes":["Mab21l2"]},
    ]
    if include_flow:
        cases.append({"task":"T1","model":"ot_flow","source":sample_last,"history":sample_first,"source_stage":9.5,"history_stage":8.5,"target_stage":10.5,"epochs":epochs})
    if include_papers:
        for model in ["wot_linear", "prescient"]:
            install_model("wot" if model == "wot_linear" else model, root)
            cases.append({"task":"T1","model":model,"source":sample_last,"history":sample_first,"source_stage":9.5,"history_stage":8.5,"target_stage":10.5,"epochs":epochs})
    results = []
    for case in cases:
        case.update(seed=0,n_cells=max_cells,max_train_cells=max_cells)
        case["output"] = f"{dest}/{case['task']}-{case['model']}.h5ad"
        result = run_config(case, root=root)
        output = Path(root) / case["output"]
        if case["task"] == "T1":
            if case["model"] == "copy_last":
                metric = score_files(task="T1",input_path=output,target=Path(root)/sample_last,reference=Path(root)/sample_first,max_cells=max_cells)
                meaning = "time-forward copy baseline on demo"
            else:
                meaning = "forecast file checked; E10.5 target unavailable, no future metrics"
                metric = {"metrics": None, "status": "target_unavailable"}
        elif case["task"] == "T2":
            metric = score_files(task="T2",input_path=output,target=Path(root)/(sample+"sample_heart_9.5.h5ad"),reference=Path(root)/(sample+"sample_heart_9.25.h5ad"),max_cells=max_cells)
            meaning="demo-only snapshots; sample E9.25 is not in current official full training split"
        else:
            metric = score_files(task="T3",input_path=output,target=Path(root)/(sample+"sample_mab21l2_ko.h5ad"),wt=Path(root)/(sample+"sample_wt.h5ad"),max_cells=max_cells)
            meaning="seen-KO pipeline diagnostic; not unseen-perturbation validation"
        metric["demo_meaning"] = meaning
        write_json(output.with_suffix(".metrics.json"), metric)
        results.append({"task":case["task"],"model":case["model"],"n_cells":result["n_cells"],"output":str(output),"meaning":meaning,"metrics":metric["metrics"]})
        print(f"demo complete: {case['task']} {case['model']}",flush=True)
    summary = {"note":"150-cell demo samples: engineering smoke checks only, no scientific or official competition score","results":results}
    write_json(Path(root)/dest/"summary.json",summary)
    return summary


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    sub = main.add_subparsers(dest="command",required=True)
    sub.add_parser("catalog",help="List public and registered training resources")
    download = sub.add_parser("download",help="Download public files and authorized released training")
    download.add_argument("--all",action="store_true",help="Process the complete selected catalog")
    download.add_argument("--public-only",action="store_true")
    download.add_argument("--manifest",help="Private local JSON containing authorized URL/local_path")
    download.add_argument("--task",choices=["T1","T2","T3"])
    download.add_argument("--root",default=".")
    models = sub.add_parser("models")
    action = models.add_subparsers(dest="action",required=True)
    action.add_parser("list")
    install = action.add_parser("install",help="Acquire locked author source, without training or installing dependencies")
    install.add_argument("name",nargs="?")
    install.add_argument("--all",action="store_true")
    install.add_argument("--root",default=".")
    run = sub.add_parser("run",help="Produce predictions using local adapters")
    run.add_argument("--config",required=True)
    run.add_argument("--root",default=".")
    score = sub.add_parser("score",help="Compute official raw metrics against a supplied local target")
    score.add_argument("--task",choices=["T1","T2","T3"],required=True)
    score.add_argument("--input",required=True)
    score.add_argument("--target",required=True)
    score.add_argument("--reference")
    score.add_argument("--wt")
    score.add_argument("--setting",choices=["heart","embryo"],default="heart")
    score.add_argument("--seed",type=int,default=0)
    score.add_argument("--max-cells",type=int,default=2000)
    score.add_argument("--out")
    smoke = sub.add_parser("demo",help="Run engineering smoke checks on 150-cell official samples")
    smoke.add_argument("--root",default=".")
    smoke.add_argument("--include-flow",action="store_true")
    smoke.add_argument("--include-papers",action="store_true",help="Run real WOT coupling and PRESCIENT author-model adapters")
    smoke.add_argument("--epochs",type=int,default=20)
    smoke.add_argument("--max-cells",type=int,default=150)
    return main


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=="catalog":
            emit(load_catalog())
        elif args.command=="download":
            report=download_all(root=args.root,manifest=args.manifest,task=args.task,public_only=args.public_only)
            print("Report: artifacts/download-report.json")
            return 0 if report["selected_resources_complete"] else 2
        elif args.command=="models":
            if args.action=="list":
                emit(model_registry())
            else:
                if not args.all and not args.name:
                    raise ValueError("Supply a model name or --all")
                names=list(model_registry()["external"]) if args.all else [args.name]
                records=[]
                for name in names:
                    records.append(install_model(name,args.root))
                    write_json(Path(args.root)/"artifacts/model-source-report.json",records)
                    print(f"Source acquired: {name}",flush=True)
                emit(records)
        elif args.command=="run":
            emit(run_config(json.loads(Path(args.config).read_text()),root=args.root))
        elif args.command=="score":
            emit(score_files(task=args.task,input_path=args.input,target=args.target,reference=args.reference,wt=args.wt,setting=args.setting,seed=args.seed,max_cells=args.max_cells,out=args.out))
        elif args.command=="demo":
            result=demo(args.root,include_flow=args.include_flow,include_papers=args.include_papers,epochs=args.epochs,max_cells=args.max_cells)
            print(f"Demo models: {len(result['results'])}. Summary: artifacts/demo/summary.json")
        return 0
    except (ValueError,RuntimeError,OSError,KeyError,ImportError,subprocess.SubprocessError) as exc:
        print(f"Error: {exc}",file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
