#!/usr/bin/env python3
"""Benchmark GNNDelete / SCGU / SCGUMoE(+two-loss) on KG datasets.

统计内容：
1) 训练时间：按指定模型、数据集、GNN(rgcn/rgat)、seed 调用 delete_gnn.py 跑 N 个 epoch
   （默认 10），记录子进程 wall time、trainer_log 里的最后一次 train_time，以及 nvidia-smi
   观察到的峰值显存。
2) 现有模型推理时间：按项目原有 checkpoint 命名规则加载 model_best.pt/model_final.pt，
   复用项目 KGTrainer.eval 做 val/test 推理计时，记录 PyTorch CUDA peak allocated/reserved。

默认三个模型配置：
- gnndelete: 原始 GNNDelete deletion_operator=original，embedding_mse 随机性 loss。
- scgu: SCGU low-rank deletion_operator=scgu，embedding_mse 随机性 loss。
- scgumoe_two_loss: SCGU-MoE deletion_operator=scgu_moe，loss_l + bce_rank_l2 两个目标。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch_geometric.seed import seed_everything
from torch_geometric.utils import is_undirected, k_hop_subgraph

from framework import get_model, get_trainer
from framework.training_args import parse_args as project_parse_args


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    unlearning_model: str
    deletion_operator: str
    loss_fct: str = "mse_mean"
    loss_type: str = "both_layerwise"
    alpha: float = 0.5
    neg_sample_random: str = "non_connected"
    loss_r_type: str = "embedding_mse"
    loss_r_margin: float = 0.0
    loss_r_beta: float = 1.0
    scgu_rank: int = 16
    scgu_init: str = "random"
    del_moe_num_experts: int = 4
    del_gate_emb_dim: int = 16
    del_gate_mode: str = "soft"


MODEL_SPECS: Dict[str, ModelSpec] = {
    "gnndelete": ModelSpec(
        name="gnndelete",
        unlearning_model="gnndelete",
        deletion_operator="original",
    ),
    "scgu": ModelSpec(
        name="scgu",
        unlearning_model="gnndelete",
        deletion_operator="scgu",
    ),
    "scgumoe_two_loss": ModelSpec(
        name="scgumoe_two_loss",
        unlearning_model="gnndelete_nodeemb",
        deletion_operator="scgu_moe",
        loss_r_type="bce_rank_l2",
    ),
}


def split_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def spec_to_project_cli(spec: ModelSpec) -> List[str]:
    args = [
        "--unlearning_model", spec.unlearning_model,
        "--loss_fct", spec.loss_fct,
        "--loss_type", spec.loss_type,
        "--alpha", str(spec.alpha),
        "--neg_sample_random", spec.neg_sample_random,
        "--loss_r_type", spec.loss_r_type,
        "--loss_r_margin", str(spec.loss_r_margin),
        "--loss_r_beta", str(spec.loss_r_beta),
        "--deletion_operator", spec.deletion_operator,
    ]
    if spec.deletion_operator in {"scgu", "scgu_lowrank", "lowrank", "scgu_moe"}:
        args += ["--scgu_rank", str(spec.scgu_rank), "--scgu_init", spec.scgu_init]
    if spec.deletion_operator == "scgu_moe":
        args += [
            "--del_moe_num_experts", str(spec.del_moe_num_experts),
            "--del_gate_emb_dim", str(spec.del_gate_emb_dim),
            "--del_gate_mode", spec.del_gate_mode,
        ]
    return args


def make_project_args(extra_cli: List[str]):
    """Use the project's own parse_args so defaults/path tags match delete_gnn.py."""
    old_argv = sys.argv[:]
    try:
        sys.argv = ["benchmark_unlearning_models.py"] + extra_cli
        return project_parse_args()
    finally:
        sys.argv = old_argv


def build_checkpoint_dir(args) -> str:
    """Exactly mirror delete_gnn.py checkpoint_dir construction."""
    if "gnndelete" in args.unlearning_model:
        loss_tag: List[Any] = [args.loss_fct, args.loss_type, args.alpha, args.neg_sample_random]
        if getattr(args, "loss_r_type", "embedding_mse") != "embedding_mse":
            loss_tag.extend([args.loss_r_type, args.loss_r_margin, args.loss_r_beta])
        if getattr(args, "deletion_operator", "original") != "original":
            loss_tag.append(args.deletion_operator)
            if args.deletion_operator in ["scgu", "scgu_lowrank", "lowrank"]:
                loss_tag.extend([args.scgu_rank, args.scgu_init])
            elif args.deletion_operator == "scgu_moe":
                loss_tag.extend([
                    args.scgu_rank,
                    args.scgu_init,
                    args.del_moe_num_experts,
                    args.del_gate_emb_dim,
                    args.del_gate_mode,
                ])
            else:
                loss_tag.extend([
                    args.del_moe_num_experts,
                    args.del_lora_rank,
                    args.del_gate_emb_dim,
                    args.del_lora_alpha,
                    args.del_lora_dropout,
                    args.del_gate_mode,
                    args.del_gate_temperature,
                ])
        return os.path.join(
            args.checkpoint_dir,
            args.dataset,
            args.gnn,
            args.unlearning_model,
            "-".join(str(i) for i in loss_tag),
            "-".join(str(i) for i in [args.df, args.df_size, args.random_seed]),
        )
    return os.path.join(
        args.checkpoint_dir,
        args.dataset,
        args.gnn,
        args.unlearning_model,
        "-".join(str(i) for i in [args.df, args.df_size, args.random_seed]),
    )


class NvidiaSmiMonitor:
    def __init__(self, gpu_index: int = 0, interval: float = 0.2):
        self.gpu_index = gpu_index
        self.interval = interval
        self.max_mb: Optional[int] = None
        self.samples: List[int] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.available = True

    def _sample_once(self) -> Optional[int]:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            ).strip().splitlines()[0]
            return int(out.strip())
        except Exception:
            self.available = False
            return None

    def _run(self):
        while not self._stop.is_set():
            value = self._sample_once()
            if value is not None:
                self.samples.append(value)
                self.max_mb = value if self.max_mb is None else max(self.max_mb, value)
            time.sleep(self.interval)

    def start(self):
        first = self._sample_once()
        if first is None:
            return self
        self.samples.append(first)
        self.max_mb = first
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> Optional[int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.max_mb


def load_trainer_log_stats(ckpt_dir: str) -> Dict[str, Any]:
    path = os.path.join(ckpt_dir, "trainer_log.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            log = json.load(f)
    except Exception as exc:
        return {"trainer_log_error": repr(exc)}

    train_times = [x.get("train_time") for x in log.get("log", []) if isinstance(x, dict) and "train_time" in x]
    out: Dict[str, Any] = {}
    if train_times:
        out["logged_train_time_last_sec"] = train_times[-1]
        out["logged_train_time_sum_sec"] = sum(float(x) for x in train_times)
        out["logged_train_time_count"] = len(train_times)
    for key in ["dt_auc", "dt_aup", "df_auc", "df_aup", "last_loss_total", "last_loss_l", "last_loss_r", "mi_sucrate_all_before", "mi_sucrate_sub_before", "mi_sucrate_all_before_current", "mi_sucrate_sub_before_current", "mi_sucrate_all_after", "mi_sucrate_sub_after", "mi_ratio_all", "mi_ratio_sub"]:
        if key in log:
            out[key] = log[key]
    return out


def prepare_delete_data(args):
    """Prepare Df/Dr/S_Df exactly like delete_gnn.py for KG rgcn/rgat."""
    data_path = os.path.join(args.data_dir, args.dataset, f"d_{args.random_seed}.pkl")
    with open(data_path, "rb") as f:
        dataset, data = pickle.load(f)

    if args.df == "none":
        raise ValueError("--df cannot be none for deletion/inference benchmark")

    if args.df_size >= 100:
        df_size = int(args.df_size)
    else:
        df_size = int(args.df_size / 100 * data.train_pos_edge_index.shape[1])

    df_masks = torch.load(os.path.join(args.data_dir, args.dataset, f"df_{args.random_seed}.pt"), map_location="cpu")
    if args.df in df_masks:
        df_mask_all = df_masks[args.df]
    elif args.df == "random":
        df_mask_all = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    else:
        raise KeyError(f"Unknown --df {args.df!r}; available={sorted(list(df_masks.keys()) + ['random'])}")

    df_nonzero = df_mask_all.nonzero().squeeze()
    idx = torch.randperm(df_nonzero.shape[0])[:df_size]
    df_global_idx = df_nonzero[idx]

    dr_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    dr_mask[df_global_idx] = False
    df_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    df_mask[df_global_idx] = True

    data.directed_df_edge_index = data.train_pos_edge_index[:, df_mask]
    data.directed_df_edge_type = data.train_edge_type[df_mask]

    _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(),
        2,
        data.train_pos_edge_index,
        num_nodes=data.num_nodes,
    )
    _, one_hop_edge, _, _ = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(),
        1,
        data.train_pos_edge_index,
        num_nodes=data.num_nodes,
    )
    sdf_node_1hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_2hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_1hop[one_hop_edge.flatten().unique()] = True
    sdf_node_2hop[two_hop_edge.flatten().unique()] = True
    data.sdf_node_1hop_mask = sdf_node_1hop
    data.sdf_node_2hop_mask = sdf_node_2hop

    if args.gnn not in ["rgcn", "rgat"]:
        raise ValueError("This benchmark script is intended for KG rgcn/rgat only.")
    if is_undirected(data.train_pos_edge_index):
        raise AssertionError("Expected directed KG train_pos_edge_index before reverse-edge expansion.")

    r, c = data.train_pos_edge_index
    rev_edge_index = torch.stack([c, r], dim=0)
    rev_edge_type = data.train_edge_type + args.num_edge_type
    data.edge_index = torch.cat((data.train_pos_edge_index, rev_edge_index), dim=1)
    data.edge_type = torch.cat([data.train_edge_type, rev_edge_type], dim=0)
    if hasattr(data, "train_mask"):
        data.train_mask = data.train_mask.repeat(2).view(-1)

    data.sdf_mask = two_hop_mask.repeat(2).view(-1)
    data.df_mask = df_mask.repeat(2).view(-1)
    data.dr_mask = dr_mask.repeat(2).view(-1)

    if not is_undirected(data.edge_index):
        raise AssertionError("Reverse-edge expanded KG edge_index should be undirected.")

    return dataset, data, sdf_node_1hop, sdf_node_2hop


def find_checkpoint(ckpt_dir: str, preferred: str = "best") -> Optional[str]:
    names = ["model_best.pt", "model_final.pt"] if preferred == "best" else ["model_final.pt", "model_best.pt"]
    for name in names:
        path = os.path.join(ckpt_dir, name)
        if os.path.exists(path):
            return path
    return None


def run_training_case(py: str, delete_script: str, project_cli: List[str], gpu_index: int, monitor_interval: float) -> Dict[str, Any]:
    ckpt_args = make_project_args(project_cli)
    ckpt_dir = build_checkpoint_dir(ckpt_args)
    cmd = [py, delete_script] + project_cli

    monitor = NvidiaSmiMonitor(gpu_index=gpu_index, interval=monitor_interval).start()
    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - start
    peak_mb = monitor.stop()

    out_tail = "\n".join(proc.stdout.splitlines()[-80:]) if proc.stdout else ""
    result: Dict[str, Any] = {
        "train_returncode": proc.returncode,
        "train_wall_time_sec": wall,
        "train_peak_nvidia_smi_mb": peak_mb,
        "checkpoint_dir": ckpt_dir,
    }
    result.update(load_trainer_log_stats(ckpt_dir))
    if proc.returncode != 0:
        result["train_error_tail"] = out_tail
    return result


def run_inference_case(
    project_cli: List[str],
    stage: str,
    repeats: int,
    ckpt_preferred: str,
    pred_all: bool,
    gpu_index: int,
    monitor_interval: float,
) -> Dict[str, Any]:
    args = make_project_args(project_cli)
    seed_everything(args.random_seed)
    ckpt_dir = build_checkpoint_dir(args)
    ckpt_path = find_checkpoint(ckpt_dir, preferred=ckpt_preferred)
    if ckpt_path is None:
        return {
            "checkpoint_dir": ckpt_dir,
            "checkpoint_path": None,
            "inference_status": "missing_checkpoint",
        }

    _, data, mask_1hop, mask_2hop = prepare_delete_data(args)
    model = get_model(args, mask_1hop, mask_2hop, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    trainer = get_trainer(args)

    # Warmup once to avoid first-call kernel/cache overhead in timed repeats.
    model.eval()
    with torch.no_grad():
        _ = trainer.eval(model, data, stage=stage, pred_all=False)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times: List[float] = []
    last_log: Dict[str, Any] = {}
    monitor = NvidiaSmiMonitor(gpu_index=gpu_index, interval=monitor_interval).start()
    for _ in range(repeats):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            *_, log = trainer.eval(model, data, stage=stage, pred_all=pred_all)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        last_log = log
    nvidia_peak_mb = monitor.stop()

    result: Dict[str, Any] = {
        "checkpoint_dir": ckpt_dir,
        "checkpoint_path": ckpt_path,
        "inference_status": "ok",
        "inference_stage": stage,
        "inference_repeats": repeats,
        "inference_time_mean_sec": sum(times) / len(times),
        "inference_time_min_sec": min(times),
        "inference_time_max_sec": max(times),
        "load_missing_keys": len(missing),
        "load_unexpected_keys": len(unexpected),
        "infer_peak_nvidia_smi_mb": nvidia_peak_mb,
    }
    if DEVICE.type == "cuda":
        result.update({
            "infer_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "infer_peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
            "infer_end_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "infer_end_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        })
    for k, v in last_log.items():
        if isinstance(v, (int, float, str)):
            result[k] = v
    return result


def write_outputs(rows: List[Dict[str, Any]], output_prefix: str):
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix + ".json"
    csv_path = output_prefix + ".csv"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    keys = sorted({k for row in rows for k in row.keys()})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


def main():
    p = argparse.ArgumentParser(description="Benchmark GNNDelete/SCGU/SCGUMoE training and inference.")
    p.add_argument("--datasets", default="WordNet18", help="Comma-separated datasets, e.g. WordNet18,FB15k-237")
    p.add_argument("--gnns", default="rgcn,rgat", help="Comma-separated: rgcn,rgat")
    p.add_argument("--models", default="gnndelete,scgu,scgumoe_two_loss", help=f"Comma-separated from {list(MODEL_SPECS)}")
    p.add_argument("--seeds", default="42", help="Comma-separated seeds")
    p.add_argument("--df", default="out", help="Deletion set: in/out/random; must exist in data/<dataset>/df_<seed>.pt")
    p.add_argument("--df_size", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=10, help="Training epochs for timing; default 10.")
    p.add_argument("--valid_freq", type=int, default=None, help="Default = epochs, so validation runs once at the end.")
    p.add_argument("--batch_size", type=int, default=None, help="Optional override; otherwise project defaults apply.")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--checkpoint_dir", default="./checkpoint")
    p.add_argument("--delete_script", default="delete_gnn.py")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--no_train", action="store_true", help="Skip 10-epoch training benchmark.")
    p.add_argument("--no_inference", action="store_true", help="Skip existing-checkpoint inference benchmark.")
    p.add_argument("--inference_stage", default="test", choices=["val", "test"])
    p.add_argument("--inference_repeats", type=int, default=3)
    p.add_argument("--ckpt", default="best", choices=["best", "final"])
    p.add_argument("--pred_all", action="store_true", help="Also compute all-pair logits in eval; much more memory.")
    p.add_argument("--gpu_index", type=int, default=0)
    p.add_argument("--monitor_interval", type=float, default=0.2)
    p.add_argument("--output_prefix", default=None)
    args = p.parse_args()

    datasets = split_csv(args.datasets)
    gnns = split_csv(args.gnns)
    model_names = split_csv(args.models)
    seeds = [int(x) for x in split_csv(args.seeds)]
    valid_freq = args.valid_freq or args.epochs

    unknown = [m for m in model_names if m not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}; valid={list(MODEL_SPECS)}")
    for gnn in gnns:
        if gnn not in {"rgcn", "rgat"}:
            raise ValueError("This script only supports --gnns rgcn,rgat for KG datasets.")

    rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        for gnn in gnns:
            for seed in seeds:
                for model_name in model_names:
                    spec = MODEL_SPECS[model_name]
                    project_cli = [
                        "--dataset", dataset,
                        "--gnn", gnn,
                        "--random_seed", str(seed),
                        "--df", args.df,
                        "--df_size", str(args.df_size),
                        "--data_dir", args.data_dir,
                        "--checkpoint_dir", args.checkpoint_dir,
                    ] + spec_to_project_cli(spec)
                    if args.batch_size is not None:
                        project_cli += ["--batch_size", str(args.batch_size)]

                    base_row = {
                        "dataset": dataset,
                        "gnn": gnn,
                        "seed": seed,
                        "df": args.df,
                        "df_size": args.df_size,
                        "model": model_name,
                        "unlearning_model": spec.unlearning_model,
                        "deletion_operator": spec.deletion_operator,
                        "loss_r_type": spec.loss_r_type,
                    }

                    if not args.no_train:
                        train_cli = project_cli + ["--epochs", str(args.epochs), "--valid_freq", str(valid_freq)]
                        print(f"[train] {dataset} {gnn} seed={seed} model={model_name}")
                        train_res = run_training_case(args.python, args.delete_script, train_cli, args.gpu_index, args.monitor_interval)
                        rows.append({**base_row, "phase": "train", "epochs": args.epochs, **train_res})

                    if not args.no_inference:
                        print(f"[infer] {dataset} {gnn} seed={seed} model={model_name}")
                        infer_res = run_inference_case(
                            project_cli,
                            args.inference_stage,
                            args.inference_repeats,
                            args.ckpt,
                            args.pred_all,
                            args.gpu_index,
                            args.monitor_interval,
                        )
                        rows.append({**base_row, "phase": "inference", **infer_res})

    if args.output_prefix is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_prefix = f"benchmark_results/unlearning_benchmark_{stamp}"
    write_outputs(rows, args.output_prefix)


if __name__ == "__main__":
    main()
