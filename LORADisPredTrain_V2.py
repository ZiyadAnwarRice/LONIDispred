#!/usr/bin/env python3
"""
LORADisPredTrain.py


Run (example, DDP):
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  LORADisPredTrain.py \
  --run_name "esm2_msa_$(date +%Y%m%d-%H%M%S)" \
  --model_name esm2_t33_650M_UR50D \
  --max_sequence_length_train 1500 \
  --max_sequence_length_test 5000 \
  --numberofepochs 3 \
  --numberofbatch 2 \
  --learning_rate 8e-5 \
  --scheduler cosine \
  --warmup_ratio 0.12 \
  --weight_decay 0.05 \
  --gradient_clip 1.0 \
  --grad_accum_steps 4 \
  --rank 16 --lora_alpha 32 --lora_dropout 0.10 \
  --head crf --head_dropout 0.15 \
  --loss ce --layer_pooling token_weighted \
  --amp bf16 \
  --data_backend local \
  --train_sequences "/work/$USER/LONIDispred/data/train_sequences.pkl" \
  --train_labels    "/work/$USER/LONIDispred/data/train_labels.pkl" \
  --val_sequences   "/work/$USER/LONIDispred/data/testPDB_sequences.pkl" \
  --val_labels      "/work/$USER/LONIDispred/data/testPDB_labels.pkl" \
  --testNox_sequences "/work/$USER/LONIDispred/data/Seq_CAID3NOX.pkl" \
  --testNox_labels    "/work/$USER/LONIDispred/data/target_CAID3NOX.pkl" \
  --testPDB_sequences "/work/$USER/LONIDispred/data/Seq_CAID3PDB.pkl" \
  --testPDB_labels    "/work/$USER/LONIDispred/data/target_CAID3PDB.pkl" \
  --use_msa \
  --msa_dim 23 \
  --msa_fuse concat \
  --msa_train_pkl   "/work/$USER/LONIDispred/MSA_list/msa_feat_F23_train_LIST.pkl" \
  --msa_val_pkl     "/work/$USER/LONIDispred/MSA_list/msa_feat_F23_ValPDB_LIST.pkl" \
  --msa_testNox_pkl "/work/$USER/LONIDispred/MSA_list/msa_feat_F23_testNOX_LIST.pkl" \
  --msa_testPDB_pkl "/work/$USER/LONIDispred/MSA_list/msa_feat_F23_testPDB_LIST.pkl" \
  --output_dir "/work/$USER/LONIDispred/outputs"


"""

import os
import re
import json
import time
import shutil
import random
import warnings
import argparse
import pickle
import getpass
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, log_loss

import transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from datasets import Dataset

import peft
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

import mlflow
import wandb

import fsspec
import s3fs

# ----------------- GPU pinning (DDP) -----------------

def pin_gpu_to_local_rank():
    if not torch.cuda.is_available():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    print(
        f"[rank {os.environ.get('RANK','?')}] LOCAL_RANK={local_rank} "
        f"current_device={torch.cuda.current_device()} "
        f"name={torch.cuda.get_device_name(torch.cuda.current_device())}",
        flush=True
    )

# ----------------- HF cache dirs (cluster-friendly) -----------------

user = getpass.getuser()
base_work_dir = f"/work/{user}/huggingface"
os.environ["HF_HOME"] = base_work_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(base_work_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(base_work_dir, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(base_work_dir, "hub")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# ----------------- Reproducibility -----------------

SEED = 2515
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

# Quiet progress output
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("WANDB_SILENT", "true")

warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)

# ----------------- Output & tracking helpers -----------------

def slugify(x: str) -> str:
    x = x.strip().lower()
    x = re.sub(r"[^a-z0-9\-_. ]+", "", x)
    x = re.sub(r"\s+", "_", x)
    return x[:80] or "run"

def prepare_run_dirs(run_name: str, *, output_root: str) -> Dict[str, str]:
    os.makedirs(output_root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"{ts}__{slugify(run_name)}")
    sub = {
        "root": run_dir,
        "checkpoints": os.path.join(run_dir, "checkpoints"),
        "model": os.path.join(run_dir, "model"),
        "results": os.path.join(run_dir, "results"),
        "config": os.path.join(run_dir, "config"),
        "logs": os.path.join(run_dir, "logs"),
        "wandb_shared": os.path.join(output_root, "wandb"),  # shared across runs
        "mlruns": os.path.join(output_root, "mlruns"),
    }
    for k, d in sub.items():
        if k not in ("wandb_shared", "mlruns"):
            os.makedirs(d, exist_ok=True)
    os.makedirs(sub["wandb_shared"], exist_ok=True)
    os.makedirs(sub["mlruns"], exist_ok=True)

    # Optional symlink to latest
    try:
        latest_link = os.path.join(output_root, "latest")
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            if os.path.islink(latest_link):
                os.unlink(latest_link)
        os.symlink(run_dir, latest_link)
    except Exception:
        pass

    # Snapshot script
    try:
        src_path = os.path.abspath(__file__)
        shutil.copy(src_path, os.path.join(sub["config"], os.path.basename(src_path)))
    except Exception:
        pass

    return sub

def set_tracking(output_root: str, experiment_name: str):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_uri = os.path.join(output_root, "mlruns")
        os.makedirs(tracking_uri, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)

    try:
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name)
    except Exception:
        pass
    exp = mlflow.set_experiment(experiment_name)
    return exp

# ----------------- Backend helpers (local / s3) -----------------

def has_minio_conf() -> bool:
    return bool(os.getenv("MINIO_ENDPOINT") or os.getenv("AWS_ACCESS_KEY_ID"))

def is_s3_like_path(p: str) -> bool:
    return isinstance(p, str) and p.startswith("s3://")

def _s3_key(path: str) -> str:
    return path[5:] if path.startswith("s3://") else path.lstrip("/")

def _is_s3fs(fs) -> bool:
    try:
        import s3fs as _s3fs_mod
        return isinstance(fs, _s3fs_mod.S3FileSystem)
    except Exception:
        return False

@contextmanager
def open_binary(path: str, *, fs=None, backend: str = "auto"):
    if backend not in {"auto", "local", "s3"}:
        raise ValueError("backend must be 'auto', 'local', or 's3'")

    if is_s3_like_path(path):
        with fsspec.open(path, "rb") as f:
            yield f
        return

    if backend in {"auto", "local"} and os.path.exists(path):
        with open(path, "rb") as f:
            yield f
        return

    if backend in {"auto", "s3"} and fs is not None:
        norm = _s3_key(path) if _is_s3fs(fs) else path
        with fs.open(norm, "rb") as f:
            yield f
        return

    with fsspec.open(path, "rb") as f:
        yield f

def load_pickle(path: str, *, fs=None, backend: str = "auto"):
    with open_binary(path, fs=fs, backend=backend) as f:
        return pickle.load(f)

def make_s3fs() -> s3fs.S3FileSystem:
    endpoint = os.getenv("MINIO_ENDPOINT")
    protocol = os.getenv("MINIO_PROTOCOL", "https")
    verify_path = os.getenv("MINIO_CERT_FILE_PATH")
    key = os.getenv("MINIO_ACCESS_KEY")
    secret = os.getenv("MINIO_SECRET_KEY")

    if endpoint:
        fsspec.config.conf = {
            "s3": {
                "key": key,
                "secret": secret,
                "client_kwargs": {
                    "endpoint_url": f"{protocol}://{endpoint}",
                    **({"verify": verify_path} if verify_path else {}),
                },
            }
        }
        return s3fs.S3FileSystem(anon=False)
    return s3fs.S3FileSystem(anon=False)

# ----------------- Metrics helpers -----------------

def truncate_labels(labels: List[List[int]], max_length: int) -> List[List[int]]:
    return [lab[:max_length] for lab in labels]

def softmax_last_dim_np(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=-1, keepdims=True)

def f1_max_calc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    denom = (precision + recall)
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom != 0)
    best_idx = int(np.argmax(f1))
    best_f1 = float(f1[best_idx])
    best_thr = float(thresholds[max(0, best_idx - 1)]) if thresholds.size else 0.5
    return best_f1, best_thr

def compute_metrics_eval(eval_pred) -> dict:
    # Works for Trainer eval
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    probs = softmax_last_dim_np(logits)[..., 1]
    mask = labels != -100
    y_true = labels[mask].astype(int)
    y_prob = probs[mask]

    out = {}
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auc"] = float("nan")
    try:
        out["Average Precision"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["Average Precision"] = float("nan")

    f1m, _ = f1_max_calc(y_true, y_prob)
    out["F1_max"] = float(f1m)

    try:
        out["NLL"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        out["NLL"] = float("nan")
    try:
        out["Brier"] = float(np.mean((y_prob - y_true) ** 2))
    except Exception:
        out["Brier"] = float("nan")
    return out

# ----------------- Optional: Calibration -----------------

def fit_temperature(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    valid = mask
    Z = logits[valid]
    y = labels[valid].astype(int)
    best_T, best_nll = 1.0, float("inf")
    for T in np.linspace(0.5, 3.0, 26):
        scaled = Z / T
        p = softmax_last_dim_np(scaled)[..., 1]
        try:
            nll = log_loss(y, p, labels=[0, 1])
        except Exception:
            continue
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T

# ----------------- Heads / pooling / model -----------------

try:
    import torchcrf  # optional
except Exception:
    torchcrf = None

@dataclass
class HeadConfig:
    head: str = "linear"            # linear|mlp|crf|conformer
    hidden: int = 0                 # MLP hidden dim
    dropout: float = 0.1
    layer_pooling: str = "last"     # last|last4|weighted|token_weighted
    lora_target: Optional[List[str]] = None

    conformer_heads: int = 4
    conformer_kernel: int = 9
    conformer_dropout: float = 0.1

    # MSA fusion
    use_msa: bool = False
    msa_dim: int = 23
    msa_fuse: str = "concat"        # concat|add
    msa_dropout: float = 0.1

class WeightedLayerPool(nn.Module):
    def __init__(self, n_layers: int, init_last_only: bool = True):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))
        if init_last_only:
            with torch.no_grad():
                self.weights[-1] = 1.0

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        W = torch.softmax(self.weights, dim=0)
        stacked = torch.stack(hidden_states, dim=0)  # [L, B, T, H]
        pooled = torch.einsum("l,lbth->bth", W, stacked)
        return pooled

class TokenWiseLayerPool(nn.Module):
    """Per-token layer mixing."""
    def __init__(self, n_layers: int, hidden: int):
        super().__init__()
        self.proj = nn.Linear(n_layers, hidden)
        self.gate = nn.Linear(hidden, n_layers)

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        x = torch.stack(hidden_states, dim=0).permute(1, 2, 0, 3)  # [B,T,L,H]
        scores = x.mean(dim=-1)  # [B,T,L]
        w = torch.softmax(self.gate(F.gelu(self.proj(scores))), dim=-1)  # [B,T,L]
        pooled = torch.einsum("btl,btlh->bth", w, x)
        return pooled

class ConformerLite(nn.Module):
    def __init__(self, d: int, n_heads: int = 4, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        self.dw = nn.Conv1d(d, d, kernel_size, padding=kernel_size // 2, groups=d)
        self.pw = nn.Linear(d, d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.dw(x.transpose(1, 2)).transpose(1, 2)
        y = self.pw(y)
        x = x + self.drop(self.n1(y))
        key_padding = (attention_mask == 0) if attention_mask is not None else None
        z, _ = self.attn(x, x, x, key_padding_mask=key_padding)
        x = x + self.drop(self.n2(z))
        return x

class ESMTokenClassifier(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        num_labels: int,
        head_cfg: HeadConfig,
        *,
        lora_r: int = 2,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0
    ):
        super().__init__()
        self.num_labels = num_labels
        self.head_cfg = head_cfg
        self.base_model_name = base_model_name

        base = AutoModel.from_pretrained(f"facebook/{base_model_name}")
        base.config.output_hidden_states = True

        TASK_TOKEN_CLS = getattr(TaskType, "TOKEN_CLASSIFICATION", TaskType.TOKEN_CLS)

        # LoRA target modules (autodetect if not provided)
        if head_cfg.lora_target and len(head_cfg.lora_target) > 0:
            target_modules = head_cfg.lora_target
        else:
            candidate = ["q_proj","k_proj","v_proj","o_proj","query","key","value","dense","fc_in","fc_out"]
            names = [n for n, _ in base.named_modules()]
            target_modules = [t for t in candidate if any(n.endswith(f".{t}") or n == t for n in names)]
            if not target_modules:
                target_modules = ["query","key","value","dense"]

        self.peft_config = LoraConfig(
            task_type=TASK_TOKEN_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none"
        )
        self.esm: PeftModel = get_peft_model(base, self.peft_config)

        hidden_size = (
            self.esm.base_model.model.config.hidden_size
            if hasattr(self.esm, "base_model") else base.config.hidden_size
        )

        # Layer pooling
        if head_cfg.layer_pooling == "last4":
            self.pool = lambda hs: torch.mean(torch.stack(hs[-4:], dim=0), dim=0)
            self.pool_module = None
        elif head_cfg.layer_pooling == "weighted":
            n_layers = getattr(base.config, "num_hidden_layers", 12)
            self.pool_module = WeightedLayerPool(n_layers=n_layers + 1)
            self.pool = lambda hs: self.pool_module(hs)
        elif head_cfg.layer_pooling == "token_weighted":
            n_layers = getattr(base.config, "num_hidden_layers", 12)
            self.pool_module = TokenWiseLayerPool(n_layers=n_layers + 1, hidden=hidden_size // 2)
            self.pool = lambda hs: self.pool_module(hs)
        else:
            self.pool = lambda hs: hs[-1]
            self.pool_module = None

        self.dropout = nn.Dropout(head_cfg.dropout)

        # ---- MSA fusion (optional)
        self.use_msa = bool(head_cfg.use_msa)
        self.msa_dim = int(head_cfg.msa_dim)
        # self.msa_norm = nn.LayerNorm(self.msa_dim) if self.use_msa else None
        self.msa_fuse = str(head_cfg.msa_fuse)
        self.msa_drop = nn.Dropout(float(head_cfg.msa_dropout)) if self.use_msa else None
        if self.use_msa:
            if self.msa_fuse == "add":
                self.msa_proj = nn.Linear(self.msa_dim, hidden_size)
            else:
                self.msa_proj = nn.Linear(hidden_size + self.msa_dim, hidden_size)
        else:
            self.msa_proj = None

        # Optional conformer-lite before classifier
        self.conformer = None
        if head_cfg.head == "conformer":
            self.conformer = ConformerLite(
                hidden_size,
                n_heads=head_cfg.conformer_heads,
                kernel_size=head_cfg.conformer_kernel,
                dropout=head_cfg.conformer_dropout,
            )

        # Head
        if head_cfg.head == "mlp" and head_cfg.hidden and head_cfg.hidden > 0:
            self.head_net = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, head_cfg.hidden),
                nn.GELU(),
                nn.Dropout(head_cfg.dropout),
                nn.Linear(head_cfg.hidden, num_labels),
            )
            self.emission = None
            self.crf = None
        elif head_cfg.head == "crf":
            self.emission = nn.Linear(hidden_size, num_labels)
            if torchcrf is None:
                raise ImportError("torchcrf is not installed. Try: pip install torchcrf")
            self.crf = torchcrf.CRF(num_tags=num_labels, batch_first=True)
            self.head_net = None
        else:
            self.emission = nn.Linear(hidden_size, num_labels)
            self.crf = None
            self.head_net = None

    def forward(self, input_ids=None, attention_mask=None, labels=None, msa_feat=None):
        out = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states
        x = self.pool(hs)        # [B,T,H]
        x = self.dropout(x)

        # --- fuse MSA features
        if self.use_msa and msa_feat is not None:

            msa_feat = msa_feat.to(x.device)
            # if self.msa_norm is not None:
            #     msa_feat = self.msa_norm(msa_feat)
            msa_feat = self.msa_drop(msa_feat) if self.msa_drop is not None else msa_feat
            if self.msa_fuse == "add":
                x = x + self.msa_proj(msa_feat)
            else:
                x = self.msa_proj(torch.cat([x, msa_feat], dim=-1))

        if self.conformer is not None:
            x = self.conformer(x, attention_mask=attention_mask)

        if self.head_net is not None:
            emissions = self.head_net(x)
        else:
            emissions = self.emission(x)

        outputs = {"logits": emissions}

        # CRF loss
        if self.crf is not None:
            mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(emissions[..., 0], dtype=torch.bool)
            if labels is not None:
                safe_labels = labels.clone()
                ignore = (safe_labels == -100)
                safe_labels = safe_labels.masked_fill(ignore, 0)
                mask = mask & (~ignore)

                # torchcrf requirement: first timestep must be on
                mask[:, 0] = True
                safe_labels[:, 0] = 0

                nll = -self.crf(emissions, safe_labels, mask=mask, reduction="mean")
                outputs["loss"] = nll
        else:
            if labels is not None:
                ce = nn.CrossEntropyLoss(ignore_index=-100)
                loss = ce(emissions.view(-1, emissions.size(-1)), labels.view(-1))
                outputs["loss"] = loss

        return outputs

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.esm.save_pretrained(save_directory)
        torch.save({
            "emission": (self.emission.state_dict() if self.emission is not None else None),
            "head_net": (self.head_net.state_dict() if self.head_net is not None else None),
            "pool_module": (self.pool_module.state_dict() if self.pool_module is not None else None),
            "head_cfg": self.head_cfg.__dict__,
            "base_model_name": self.base_model_name,
            "num_labels": self.num_labels,
        }, os.path.join(save_directory, "head.pt"))

    @classmethod
    def from_pretrained(cls, load_directory: str):
        ckpt = torch.load(os.path.join(load_directory, "head.pt"), map_location="cpu")
        head_cfg = HeadConfig(**ckpt["head_cfg"]) if isinstance(ckpt.get("head_cfg"), dict) else HeadConfig()
        m = cls(ckpt.get("base_model_name"), ckpt.get("num_labels", 2), head_cfg)
        base = AutoModel.from_pretrained(f"facebook/{m.base_model_name}")
        base.config.output_hidden_states = True
        m.esm = PeftModel.from_pretrained(base, load_directory)
        if ckpt.get("emission") is not None and m.emission is not None:
            m.emission.load_state_dict(ckpt["emission"])
        if ckpt.get("head_net") is not None and m.head_net is not None:
            m.head_net.load_state_dict(ckpt["head_net"])
        if m.pool_module is not None and ckpt.get("pool_module") is not None:
            m.pool_module.load_state_dict(ckpt["pool_module"])
        return m

# ----------------- Losses / Trainer -----------------

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        valid = target != self.ignore_index
        logits = logits[valid]
        target = target[valid]
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        logpt = F.log_softmax(logits, dim=-1)
        pt = torch.exp(logpt)
        logpt_g = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt_g = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        if self.alpha is not None:
            at = torch.full_like(pt_g, fill_value=self.alpha)
            at = torch.where(target == 1, at, 1 - at)
            loss = -at * (1 - pt_g) ** self.gamma * logpt_g
        else:
            loss = -(1 - pt_g) ** self.gamma * logpt_g
        return loss.mean()

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None,
                 loss_type: str = "ce", focal_gamma: float = 2.0, focal_alpha: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_type = loss_type
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, ignore_index=-100) if loss_type == "focal" else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels if hasattr(model, "crf") and model.crf is not None else None)
        if isinstance(outputs, dict) and outputs.get("loss") is not None:
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        if self.loss_type == "focal":
            loss = self.focal(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
                ignore_index=-100,
            )
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ----------------- Optional MSA collator -----------------

class DataCollatorWithMSA:
    """
    Pads msa_feat to [B,T,F] aligned with padded input_ids.
    Each example should have "msa_feat": array-like [Ti,F] (already truncated to max_seq_len).
    """
    def __init__(self, tokenizer, msa_dim: int):
        self.inner = DataCollatorForTokenClassification(tokenizer=tokenizer)
        self.msa_dim = int(msa_dim)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        msa = None
        if "msa_feat" in features[0]:
            msa = [f.pop("msa_feat") for f in features]
        batch = self.inner(features)
        if msa is None:
            return batch

        B, T = batch["input_ids"].shape
        msa_out = torch.zeros((B, T, self.msa_dim), dtype=torch.float32)
        for i in range(B):
            mi = msa[i]
            if isinstance(mi, torch.Tensor):
                mi_t = mi.float()
            else:
                mi_t = torch.tensor(np.asarray(mi, dtype=np.float32), dtype=torch.float32)
            Ti = min(T, mi_t.shape[0])
            msa_out[i, :Ti] = mi_t[:Ti]

        batch["msa_feat"] = msa_out
        return batch

# ----------------- MLflow callback for TRAIN logging -----------------

class MLflowLogCallback(TrainerCallback):
    """
    Logs Trainer 'on_log' to MLflow so you get training loss/lr at logging_steps.
    Also logs eval metrics from on_evaluate.
    """
    def __init__(self, prefix_train: str = "train", prefix_eval: str = "eval"):
        self.prefix_train = prefix_train
        self.prefix_eval = prefix_eval

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = int(state.global_step)
        metrics = {}
        for k, v in logs.items():
            if v is None:
                continue
            if isinstance(v, (int, float, np.floating)):
                # HF uses keys like "loss", "learning_rate"
                metrics[f"{self.prefix_train}_{k}"] = float(v)
        if metrics:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception:
                pass

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        step = int(state.global_step)
        m2 = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.floating)):
                # HF eval keys like "eval_loss", "eval_auc" etc
                m2[k] = float(v)
        if m2:
            try:
                mlflow.log_metrics(m2, step=step)
            except Exception:
                pass

# ----------------- TrainingArguments helper -----------------

from inspect import signature

def make_training_args(
    *,
    output_dir: str,
    learning_rate: float,
    numberofbatch: int,
    numberofepochs: int,
    seed: int,
    scheduler: str,
    warmup_ratio: float,
    weight_decay: float,
    gradient_clip: float,
    grad_accum_steps: int,
    amp: str,
    logging_steps: int,
):
    params = set(signature(TrainingArguments.__init__).parameters.keys())

    base = dict(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=numberofbatch,
        per_device_eval_batch_size=numberofbatch,
        num_train_epochs=numberofepochs,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        seed=seed,
        remove_unused_columns=False,
        disable_tqdm=True,

        # save/eval/best-model (must match if load_best_model_at_end=True)
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_auc",
        greater_is_better=True,
        save_total_limit=7,

        report_to=["wandb"],
        push_to_hub=False,
        lr_scheduler_type=scheduler,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=grad_accum_steps,
        max_grad_norm=gradient_clip,
    )

    # --- Transformers 5 uses eval_strategy; older versions use evaluation_strategy
    if "eval_strategy" in params:
        base["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        base["evaluation_strategy"] = "epoch"
    else:
        # If neither exists (very unlikely), disable best-model loading to avoid validation errors
        base["load_best_model_at_end"] = False

    allowed = {k: v for k, v in base.items() if k in params}

    # --- Safety: if load_best_model_at_end is on, enforce match between save/eval strategies
    if allowed.get("load_best_model_at_end", False) and "save_strategy" in allowed:
        if "eval_strategy" in allowed:
            allowed["eval_strategy"] = allowed["save_strategy"]
        if "evaluation_strategy" in allowed:
            allowed["evaluation_strategy"] = allowed["save_strategy"]

    # --- AMP selection
    if amp == "fp16" and "fp16" in params:
        allowed["fp16"] = True
        if "bf16" in params:
            allowed["bf16"] = False
    elif amp == "bf16" and "bf16" in params and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        allowed["bf16"] = True
        if "fp16" in params:
            allowed["fp16"] = False
    elif amp == "off":
        if "fp16" in params:
            allowed["fp16"] = False
        if "bf16" in params:
            allowed["bf16"] = False
    else:
        # auto
        try:
            if "bf16" in params and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                allowed["bf16"] = True
                if "fp16" in params:
                    allowed["fp16"] = False
            elif "fp16" in params:
                allowed["fp16"] = True
        except Exception:
            pass

    # --- (Optional) one-line debug so you can verify on the cluster
    ev = allowed.get("eval_strategy", allowed.get("evaluation_strategy", None))
    print(
        f"[TrainingArguments] save_strategy={allowed.get('save_strategy')} "
        f"eval_strategy={ev} load_best_model_at_end={allowed.get('load_best_model_at_end')}",
        flush=True
    )

    return TrainingArguments(**allowed)

# ----------------- Data I/O -----------------

def map_neg1_to_ignore(labels, ignore=-100):
    return [[(ignore if y == -1 else y) for y in seq] for seq in labels]

def _ensure_list_of_features(msa_obj, n_expected: int) -> List[Any]:
    """
    msa_obj may be list/tuple length N (order-aligned) or dict (not supported here).
    """
    if msa_obj is None:
        raise ValueError("MSA object is None but --use_msa enabled.")
    if isinstance(msa_obj, dict):
        raise ValueError("This script expects order-aligned MSA features (list/tuple). You provided a dict.")
    if not isinstance(msa_obj, (list, tuple)):
        raise ValueError("MSA PKL must contain a list/tuple of per-example features.")
    if len(msa_obj) != n_expected:
        raise ValueError(f"MSA features length mismatch: got {len(msa_obj)} expected {n_expected}")
    return list(msa_obj)

def _truncate_msa_to_token_lengths(msa_list, tokenized, max_seq_len: int, msa_dim: int):
    out = []
    for i in range(len(msa_list)):
        feat = msa_list[i]
        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().numpy()
        feat = np.asarray(feat, dtype=np.float32)

        feat = feat[:max_seq_len]

        Ti = len(tokenized["input_ids"][i])
        if feat.shape[0] < Ti:
            raise ValueError(f"MSA shorter than tokens at i={i}: msa={feat.shape[0]} tok={Ti}")
        feat = feat[:Ti]

        if feat.ndim != 2 or feat.shape[1] != msa_dim:
            raise ValueError(f"Bad msa_feat shape at i={i}: {feat.shape} expected [Ti,{msa_dim}]")

     
        out.append(feat.tolist())

    return out

def read_train_val_datasets(
    *,
    s3: Optional[s3fs.S3FileSystem],
    tokenizer,
    max_seq_len: int,
    train_seq_path: str,
    train_lbl_path: str,
    val_seq_path: Optional[str],
    val_lbl_path: Optional[str],
    val_split: float,
    random_state: int,
    data_backend: str,
    use_msa: bool,
    msa_dim: int,
    msa_train_pkl: str,
    msa_val_pkl: str,
):
    train_sequences = load_pickle(train_seq_path, fs=s3, backend=data_backend)
    train_labels    = load_pickle(train_lbl_path, fs=s3, backend=data_backend)

    use_explicit_val = bool(val_seq_path) or bool(val_lbl_path)
    if use_explicit_val:
        if not (val_seq_path and val_lbl_path):
            raise ValueError("If providing explicit val, you must provide BOTH --val_sequences and --val_labels.")
        val_sequences = load_pickle(val_seq_path, fs=s3, backend=data_backend)
        val_labels    = load_pickle(val_lbl_path, fs=s3, backend=data_backend)
        if len(val_sequences) != len(val_labels):
            raise ValueError(f"Val sequence/label mismatch: {len(val_sequences)} vs {len(val_labels)}")
    else:
        tr_seqs, va_seqs, tr_labs, va_labs = train_test_split(
            train_sequences, train_labels, test_size=val_split, random_state=random_state
        )
        train_sequences, train_labels = tr_seqs, tr_labs
        val_sequences,   val_labels   = va_seqs, va_labs

    # Tokenize (no special tokens => length == residues)
    tr_tok = tokenizer(train_sequences, padding=False, truncation=True, max_length=max_seq_len,
                       add_special_tokens=False, is_split_into_words=False)
    va_tok = tokenizer(val_sequences, padding=False, truncation=True, max_length=max_seq_len,
                       add_special_tokens=False, is_split_into_words=False)

    train_labels = truncate_labels(train_labels, max_seq_len)
    val_labels   = truncate_labels(val_labels,   max_seq_len)

    train_dataset = Dataset.from_dict({k: v for k, v in tr_tok.items()})
    train_dataset = train_dataset.add_column("labels", train_labels)

    val_dataset = Dataset.from_dict({k: v for k, v in va_tok.items()})
    val_dataset = val_dataset.add_column("labels", val_labels)

    # Optional MSA
    if use_msa:
        if not msa_train_pkl or not msa_val_pkl:
            raise ValueError("With --use_msa you must provide --msa_train_pkl and --msa_val_pkl.")
        msa_train_obj = load_pickle(msa_train_pkl, fs=s3, backend=data_backend)
        msa_val_obj   = load_pickle(msa_val_pkl,   fs=s3, backend=data_backend)

        msa_train_list = _ensure_list_of_features(msa_train_obj, len(train_sequences))
        msa_val_list   = _ensure_list_of_features(msa_val_obj,   len(val_sequences))

        msa_train_list = _truncate_msa_to_token_lengths(msa_train_list, tr_tok, max_seq_len, msa_dim)
        msa_val_list   = _truncate_msa_to_token_lengths(msa_val_list,   va_tok, max_seq_len, msa_dim)

        train_dataset = train_dataset.add_column("msa_feat", msa_train_list)
        val_dataset   = val_dataset.add_column("msa_feat", msa_val_list)

    # Class weights from TRAIN labels
    classes = np.array([0, 1])
    flat_tr = np.array([y for seq in train_labels for y in seq], dtype=int)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=flat_tr)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return train_dataset, val_dataset, class_weights

def read_test_datasets(
    *,
    s3: Optional[s3fs.S3FileSystem],
    tokenizer,
    max_seq_len: int,
    test_nox_seq_path: str,
    test_nox_lbl_path: str,
    test_pdb_seq_path: str,
    test_pdb_lbl_path: str,
    data_backend: str,
    use_msa: bool,
    msa_dim: int,
    msa_testNox_pkl: str,
    msa_testPDB_pkl: str,
):
    testNox_sequences = load_pickle(test_nox_seq_path, fs=s3, backend=data_backend)
    testNox_labels    = load_pickle(test_nox_lbl_path, fs=s3, backend=data_backend)
    testPDB_sequences = load_pickle(test_pdb_seq_path, fs=s3, backend=data_backend)
    testPDB_labels    = load_pickle(test_pdb_lbl_path, fs=s3, backend=data_backend)

    testNox_tok = tokenizer(testNox_sequences, padding=False, truncation=True, max_length=max_seq_len,
                            add_special_tokens=False, is_split_into_words=False)
    testPDB_tok = tokenizer(testPDB_sequences, padding=False, truncation=True, max_length=max_seq_len,
                            add_special_tokens=False, is_split_into_words=False)

    testNox_labels = map_neg1_to_ignore(testNox_labels, ignore=-100)
    testPDB_labels = map_neg1_to_ignore(testPDB_labels, ignore=-100)
    testNox_labels = truncate_labels(testNox_labels, max_seq_len)
    testPDB_labels = truncate_labels(testPDB_labels, max_seq_len)

    testNOX_dataset = Dataset.from_dict({k: v for k, v in testNox_tok.items()})
    testNOX_dataset = testNOX_dataset.add_column("labels", testNox_labels)

    testPDB_dataset = Dataset.from_dict({k: v for k, v in testPDB_tok.items()})
    testPDB_dataset = testPDB_dataset.add_column("labels", testPDB_labels)

    if use_msa:
        if not msa_testNox_pkl or not msa_testPDB_pkl:
            raise ValueError("With --use_msa you must provide --msa_testNox_pkl and --msa_testPDB_pkl.")
        msa_nox_obj = load_pickle(msa_testNox_pkl, fs=s3, backend=data_backend)
        msa_pdb_obj = load_pickle(msa_testPDB_pkl, fs=s3, backend=data_backend)

        msa_nox_list = _ensure_list_of_features(msa_nox_obj, len(testNox_sequences))
        msa_pdb_list = _ensure_list_of_features(msa_pdb_obj, len(testPDB_sequences))

        msa_nox_list = _truncate_msa_to_token_lengths(msa_nox_list, testNox_tok, max_seq_len, msa_dim)
        msa_pdb_list = _truncate_msa_to_token_lengths(msa_pdb_list, testPDB_tok, max_seq_len, msa_dim)

        testNOX_dataset = testNOX_dataset.add_column("msa_feat", msa_nox_list)
        testPDB_dataset = testPDB_dataset.add_column("msa_feat", msa_pdb_list)

    return testNOX_dataset, testPDB_dataset

# ----------------- Train / evaluate -----------------

def build_model(model_name: str, head_cfg: HeadConfig, *, lora_r: int, lora_alpha: float, lora_dropout: float) -> ESMTokenClassifier:
    return ESMTokenClassifier(
        model_name,
        num_labels=2,
        head_cfg=head_cfg,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

# def evaluate_model(model_path: str, testNOX_dataset: Dataset, testPDB_dataset: Dataset, tokenizer,
#                    run_dirs: Dict[str, str], use_msa: bool, msa_dim: int,
#                    temperature: Optional[float] = None, val_based_threshold: Optional[float] = None):
#     model = ESMTokenClassifier.from_pretrained(model_path)
#     collator = DataCollatorWithMSA(tokenizer, msa_dim) if use_msa else DataCollatorForTokenClassification(tokenizer=tokenizer)
#     trainer = Trainer(model=model, data_collator=collator)

#     def _eval(dataset: Dataset, tag: str):
#         pred_out = trainer.predict(test_dataset=dataset)
#         logits = pred_out.predictions
#         labels = pred_out.label_ids

#         probs_raw = softmax_last_dim_np(logits)[..., 1]
#         mask = labels != -100
#         y_true = labels[mask].astype(int)
#         y_score_raw = probs_raw[mask]

#         if temperature is not None:
#             probs_cal = softmax_last_dim_np(logits / temperature)[..., 1]
#             y_score_cal = probs_cal[mask]
#         else:
#             y_score_cal = y_score_raw

#     def metrics(y, p):
#         try:
#             auc = roc_auc_score(y, p)
#         except Exception:
#             auc = float("nan")
#         try:
#             aps = average_precision_score(y, p)
#         except Exception:
#             aps = float("nan")
#         try:
#             nll = log_loss(y, p, labels=[0, 1])
#         except Exception:
#             nll = float("nan")
#         brier = float(np.mean((p - y) ** 2))

#         # --- compute F1 at a chosen threshold
#         if val_based_threshold is None:
#             f1_best, thr_best = f1_max_calc(y, p)
#             f1_at_val = float("nan")
#             thr_used = thr_best
#         else:
#             thr_used = float(val_based_threshold)
#             yhat = (p >= thr_used).astype(int)
#             tp = np.sum((yhat == 1) & (y == 1))
#             fp = np.sum((yhat == 1) & (y == 0))
#             fn = np.sum((yhat == 0) & (y == 1))
#             prec = tp / (tp + fp + 1e-12)
#             rec  = tp / (tp + fn + 1e-12)
#             f1_at_val = float(2 * prec * rec / (prec + rec + 1e-12))
#             f1_best = float("nan")   # don’t tune on test
#             thr_best = float("nan")  # don’t tune on test

#         return {
#             "auc": float(auc),
#             "Average Precision": float(aps),
#             "F1_max": float(f1_best),          # only when no threshold is provided
#             "F1_thr_best": float(thr_best),    # only when no threshold is provided
#             "F1_at_valthr": float(f1_at_val),  # when threshold is provided
#             "thr_used": float(thr_used),
#             "NLL": float(nll),
#             "Brier": float(brier),
#         }

#         raw_metrics = metrics(y_true, y_score_raw)
#         cal_metrics = metrics(y_true, y_score_cal)

#         np.save(os.path.join(run_dirs["results"], f"{tag}_y_true.npy"), y_true)
#         np.save(os.path.join(run_dirs["results"], f"{tag}_y_score_raw.npy"), y_score_raw)
#         np.save(os.path.join(run_dirs["results"], f"{tag}_y_score_cal.npy"), y_score_cal)

#         with open(os.path.join(run_dirs["results"], f"{tag}_metrics_raw.json"), "w") as f:
#             json.dump(raw_metrics, f, indent=2)
#         with open(os.path.join(run_dirs["results"], f"{tag}_metrics_calibrated.json"), "w") as f:
#             json.dump(cal_metrics, f, indent=2)

#         # MLflow
#         try:
#             mlflow.log_metrics({f"{tag}_raw_{k}": v for k, v in raw_metrics.items() if isinstance(v, float)})
#             mlflow.log_metrics({f"{tag}_cal_{k}": v for k, v in cal_metrics.items() if isinstance(v, float)})
#             for fn in [
#                 f"{tag}_y_true.npy",
#                 f"{tag}_y_score_raw.npy",
#                 f"{tag}_y_score_cal.npy",
#                 f"{tag}_metrics_raw.json",
#                 f"{tag}_metrics_calibrated.json",
#             ]:
#                 mlflow.log_artifact(os.path.join(run_dirs["results"], fn))
#         except Exception:
#             pass

#     print("NOX Test results"); _eval(testNOX_dataset, "testNOX")
#     print("PDB Test results"); _eval(testPDB_dataset, "testPDB")



def evaluate_model(
    model_path: str,
    testNOX_dataset: Dataset,
    testPDB_dataset: Dataset,
    tokenizer,
    run_dirs: Dict[str, str],
    use_msa: bool,
    msa_dim: int,
    temperature: Optional[float] = None,
    val_based_threshold: Optional[float] = None,
):
    model = ESMTokenClassifier.from_pretrained(model_path)
    collator = DataCollatorWithMSA(tokenizer, msa_dim) if use_msa else DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = Trainer(model=model, data_collator=collator)

    def metrics(y, p):
        # y, p are 1D arrays (masked already)
        try:
            auc = float(roc_auc_score(y, p))
        except Exception:
            auc = float("nan")

        try:
            aps = float(average_precision_score(y, p))
        except Exception:
            aps = float("nan")

        try:
            nll = float(log_loss(y, p, labels=[0, 1]))
        except Exception:
            nll = float("nan")

        brier = float(np.mean((p - y) ** 2))

        if val_based_threshold is None:
            f1_best, thr_best = f1_max_calc(y, p)
            return {
                "auc": auc,
                "Average Precision": aps,
                "F1_max": float(f1_best),
                "F1_thr_best": float(thr_best),
                "F1_at_valthr": float("nan"),
                "thr_used": float(thr_best),
                "NLL": nll,
                "Brier": brier,
            }
        else:
            thr_used = float(val_based_threshold)
            yhat = (p >= thr_used).astype(int)
            tp = np.sum((yhat == 1) & (y == 1))
            fp = np.sum((yhat == 1) & (y == 0))
            fn = np.sum((yhat == 0) & (y == 1))
            prec = tp / (tp + fp + 1e-12)
            rec  = tp / (tp + fn + 1e-12)
            f1_at_val = float(2 * prec * rec / (prec + rec + 1e-12))
            return {
                "auc": auc,
                "Average Precision": aps,
                "F1_max": float("nan"),          # don't tune threshold on test
                "F1_thr_best": float("nan"),     # don't tune threshold on test
                "F1_at_valthr": f1_at_val,
                "thr_used": thr_used,
                "NLL": nll,
                "Brier": brier,
            }

    def _eval(dataset: Dataset, tag: str):
        pred_out = trainer.predict(test_dataset=dataset)
        logits = pred_out.predictions
        labels = pred_out.label_ids

        probs_raw = softmax_last_dim_np(logits)[..., 1]
        mask = labels != -100
        y_true = labels[mask].astype(int)
        y_score_raw = probs_raw[mask]

        if temperature is not None:
            probs_cal = softmax_last_dim_np(logits / temperature)[..., 1]
            y_score_cal = probs_cal[mask]
        else:
            y_score_cal = y_score_raw

        raw_metrics = metrics(y_true, y_score_raw)
        cal_metrics = metrics(y_true, y_score_cal)

        # Save arrays
        np.save(os.path.join(run_dirs["results"], f"{tag}_y_true.npy"), y_true)
        np.save(os.path.join(run_dirs["results"], f"{tag}_y_score_raw.npy"), y_score_raw)
        np.save(os.path.join(run_dirs["results"], f"{tag}_y_score_cal.npy"), y_score_cal)

        # Save JSON
        with open(os.path.join(run_dirs["results"], f"{tag}_metrics_raw.json"), "w") as f:
            json.dump(raw_metrics, f, indent=2)
        with open(os.path.join(run_dirs["results"], f"{tag}_metrics_calibrated.json"), "w") as f:
            json.dump(cal_metrics, f, indent=2)

        # MLflow
        try:
            mlflow.log_metrics({f"{tag}_raw_{k}": v for k, v in raw_metrics.items() if isinstance(v, float)})
            mlflow.log_metrics({f"{tag}_cal_{k}": v for k, v in cal_metrics.items() if isinstance(v, float)})
            mlflow.log_artifact(os.path.join(run_dirs["results"], f"{tag}_metrics_raw.json"))
            mlflow.log_artifact(os.path.join(run_dirs["results"], f"{tag}_metrics_calibrated.json"))
        except Exception:
            pass

        print(f"[{tag}] raw:", raw_metrics)
        print(f"[{tag}] cal:", cal_metrics)

    print("NOX Test results"); _eval(testNOX_dataset, "testNOX")
    print("PDB Test results"); _eval(testPDB_dataset, "testPDB")
    
def write_manifest(run_dirs: Dict[str, str], *, args: argparse.Namespace, extra: Dict[str, Any]):
    import platform
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "peft_version": peft.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "args": vars(args),
        **extra,
    }
    with open(os.path.join(run_dirs["config"], "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(run_dirs["config"], "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

# ----------------- Args -----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--run_name", default="ESM2_LoRA_Dispred")
    p.add_argument("--output_dir", default=os.getenv("OUTPUT_DIR", "outputs"))

    p.add_argument("--max_sequence_length_train", type=int, default=1000)
    p.add_argument("--max_sequence_length_test", type=int, default=5000)
    p.add_argument("--model_name", default="esm2_t12_35M_UR50D")

    p.add_argument("--numberofepochs", type=int, default=1)
    p.add_argument("--numberofbatch", type=int, default=12)
    p.add_argument("--learning_rate", type=float, default=5e-4)

    p.add_argument("--scheduler", default="cosine",
                   choices=["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"])
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.2)
    p.add_argument("--gradient_clip", type=float, default=0.5)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--amp", default="auto", choices=["auto","fp16","bf16","off"])
    p.add_argument("--logging_steps", type=int, default=200)

    # LoRA
    p.add_argument("--lora_alpha", type=float, default=1.0)
    p.add_argument("--lora_dropout", type=float, default=0.2)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--lora_target", default="",
                   help='Comma-separated modules to target (e.g. "q_proj,k_proj,v_proj,o_proj,fc_in,fc_out"). Empty => autodetect.')

    # Head/Loss/Pooling
    p.add_argument("--head", choices=["linear", "mlp", "crf", "conformer"], default="linear")
    p.add_argument("--head_hidden_dim", type=int, default=0)
    p.add_argument("--head_dropout", type=float, default=0.1)
    p.add_argument("--loss", choices=["ce", "focal"], default="ce")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--focal_alpha", type=float, default=None)
    p.add_argument("--layer_pooling", choices=["last", "last4", "weighted", "token_weighted"], default="last")

    # Conformer-lite knobs
    p.add_argument("--conformer_heads", type=int, default=4)
    p.add_argument("--conformer_kernel", type=int, default=9)
    p.add_argument("--conformer_dropout", type=float, default=0.1)

    # Data backend
    p.add_argument("--data_backend", choices=["auto", "local", "s3"], default="auto")

    # Dataset paths
    p.add_argument("--train_sequences", required=True)
    p.add_argument("--train_labels", required=True)
    p.add_argument("--val_sequences", default="")
    p.add_argument("--val_labels", default="")
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--val_seed", type=int, default=42)

    p.add_argument("--testNox_sequences", required=True)
    p.add_argument("--testNox_labels", required=True)
    p.add_argument("--testPDB_sequences", required=True)
    p.add_argument("--testPDB_labels", required=True)

    # ---- MSA optional ----
    p.add_argument("--use_msa", action="store_true", help="Enable MSA feature fusion")
    p.add_argument("--msa_dim", type=int, default=23)
    p.add_argument("--msa_fuse", choices=["concat", "add"], default="concat")
    p.add_argument("--msa_dropout", type=float, default=0.1)

    p.add_argument("--msa_train_pkl", default="", help="Order-aligned list of [L,F] for TRAIN")
    p.add_argument("--msa_val_pkl", default="", help="Order-aligned list of [L,F] for VAL")
    p.add_argument("--msa_testNox_pkl", default="", help="Order-aligned list of [L,F] for testNOX")
    p.add_argument("--msa_testPDB_pkl", default="", help="Order-aligned list of [L,F] for testPDB")

    # Tracking
    p.add_argument("--experiment_name", default="Evaluate_ESM2FinetuneV3")
    p.add_argument("--wandb_mode", default=os.getenv("WANDB_MODE", "offline"), choices=["online", "offline", "disabled"])

    return p.parse_args()

# ----------------- Main -----------------

def main():
    args = parse_args()
    pin_gpu_to_local_rank()

    run_dirs = prepare_run_dirs(args.run_name, output_root=args.output_dir)

    # Shared W&B dir across runs
    WANDB_DIR = os.path.join(args.output_dir, "wandb")
    os.makedirs(WANDB_DIR, exist_ok=True)
    os.environ["WANDB_DIR"] = WANDB_DIR
    os.environ["WANDB_MODE"] = args.wandb_mode
    if args.wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"

    print("\n" + "#" * 40 + " Starting the ML Pipeline " + "#" * 40 + "\n")
    print(f"Run dir: {run_dirs['root']}")
    print(f"Model: facebook/{args.model_name} | Head: {args.head} | Loss: {args.loss} | Pool: {args.layer_pooling}")
    print(f"MSA enabled: {args.use_msa} | msa_dim={args.msa_dim} | fuse={args.msa_fuse}")

    # MLflow
    exp = set_tracking(args.output_dir, args.experiment_name)
    try:
        mlflow.end_run()
    except Exception:
        pass
    mlflow.start_run(run_name=args.run_name)
    print(f"MLflow experiment_id: {exp.experiment_id} | tracking_uri: {mlflow.get_tracking_uri()}")

    # W&B
    wb_run = None
    if args.wandb_mode != "disabled":
        wb_run = wandb.init(
            project=args.experiment_name,
            name=args.run_name,
            notes="ESM2 LoRA token-classification with optional MSA fusion",
            dir=os.environ.get("WANDB_DIR"),
        )

    # Log params (MLflow)
    try:
        mlflow.log_params({
            "run_dir": run_dirs["root"],
            "model_name": args.model_name,
            "max_sequence_length_train": args.max_sequence_length_train,
            "max_sequence_length_test": args.max_sequence_length_test,
            "numberofepochs": args.numberofepochs,
            "numberofbatch": args.numberofbatch,
            "learning_rate": args.learning_rate,
            "scheduler": args.scheduler,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "grad_accum_steps": args.grad_accum_steps,
            "amp": args.amp,
            "head": args.head,
            "head_hidden_dim": args.head_hidden_dim,
            "head_dropout": args.head_dropout,
            "loss": args.loss,
            "focal_gamma": args.focal_gamma,
            "focal_alpha": args.focal_alpha,
            "layer_pooling": args.layer_pooling,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "rank": args.rank,
            "lora_target": args.lora_target,
            "use_msa": args.use_msa,
            "msa_dim": args.msa_dim,
            "msa_fuse": args.msa_fuse,
            "msa_dropout": args.msa_dropout,
            "data_backend": args.data_backend,
            "val_sequences": args.val_sequences,
            "val_labels": args.val_labels,
            "val_split": args.val_split,
            "val_seed": args.val_seed,
        })
    except Exception:
        pass

    # Data backends
    use_s3 = (args.data_backend == "s3") or (args.data_backend == "auto" and has_minio_conf())
    s3 = make_s3fs() if use_s3 else None

    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.model_name}")

    # Load datasets (+ optional MSA)
    t0 = time.time()
    train_dataset, val_dataset, class_weights = read_train_val_datasets(
        s3=s3,
        tokenizer=tokenizer,
        max_seq_len=args.max_sequence_length_train,
        train_seq_path=args.train_sequences,
        train_lbl_path=args.train_labels,
        val_seq_path=(args.val_sequences or None),
        val_lbl_path=(args.val_labels or None),
        val_split=args.val_split,
        random_state=args.val_seed,
        data_backend=args.data_backend,
        use_msa=args.use_msa,
        msa_dim=args.msa_dim,
        msa_train_pkl=args.msa_train_pkl,
        msa_val_pkl=args.msa_val_pkl,
    )

    testNOX_dataset, testPDB_dataset = read_test_datasets(
        s3=s3,
        tokenizer=tokenizer,
        max_seq_len=args.max_sequence_length_test,
        test_nox_seq_path=args.testNox_sequences,
        test_nox_lbl_path=args.testNox_labels,
        test_pdb_seq_path=args.testPDB_sequences,
        test_pdb_lbl_path=args.testPDB_labels,
        data_backend=args.data_backend,
        use_msa=args.use_msa,
        msa_dim=args.msa_dim,
        msa_testNox_pkl=args.msa_testNox_pkl,
        msa_testPDB_pkl=args.msa_testPDB_pkl,
    )
    data_read_time_min = (time.time() - t0) / 60.0
    print(f"Data loaded in {data_read_time_min:.2f} min")

    # Head config
    head_cfg = HeadConfig(
        head=args.head,
        hidden=args.head_hidden_dim,
        dropout=args.head_dropout,
        layer_pooling=args.layer_pooling,
        lora_target=[s.strip() for s in args.lora_target.split(",") if s.strip()] or None,
        conformer_heads=args.conformer_heads,
        conformer_kernel=args.conformer_kernel,
        conformer_dropout=args.conformer_dropout,
        use_msa=args.use_msa,
        msa_dim=args.msa_dim,
        msa_fuse=args.msa_fuse,
        msa_dropout=args.msa_dropout,
    )

    # Build model
    model = build_model(
        args.model_name,
        head_cfg,
        lora_r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Log trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    try:
        mlflow.log_metrics({
            "trainable_params": float(trainable),
            "total_params": float(total),
            "trainable_pct": float(100.0 * trainable / max(1, total)),
        })
    except Exception:
        pass
    print(f"Trainable params: {trainable:,} / {total:,} ({100.0*trainable/max(1,total):.2f}%)")

    # Training args
    training_args = make_training_args(
        output_dir=run_dirs["checkpoints"],
        learning_rate=args.learning_rate,
        numberofbatch=args.numberofbatch,
        numberofepochs=args.numberofepochs,
        seed=8893,
        scheduler=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        grad_accum_steps=args.grad_accum_steps,
        amp=args.amp,
        logging_steps=args.logging_steps,
    )

    # Collator
    collator = DataCollatorWithMSA(tokenizer, msa_dim=args.msa_dim) if args.use_msa \
        else DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Trainer (logs train loss to W&B automatically; MLflow via callback)
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics_eval,
        class_weights=class_weights,
        loss_type=args.loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        callbacks=[MLflowLogCallback()],
    )

    print("Training…")
    t1 = time.time()
    trainer.train()
    training_time_min = (time.time() - t1) / 60.0

    # Save model
    model_path = run_dirs["model"]
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved at: {model_path}")

    # Evaluate on validation (Trainer.evaluate returns eval_* metrics)
    eval_metrics = trainer.evaluate()
    print("Validation metrics:", eval_metrics)

    # Save val metrics
    with open(os.path.join(run_dirs["results"], "val_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float, np.floating))}, f, indent=2)

    # MLflow log val metrics + times
    try:
        mlflow.log_metrics({k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float, np.floating))})
        mlflow.log_metrics({"data_read_time_min": data_read_time_min, "training_time_min": training_time_min})
        mlflow.log_artifact(os.path.join(run_dirs["results"], "val_metrics.json"))
    except Exception:
        pass

    # Calibration on validation
    print("Calibrating on validation set…")
    ev_model = ESMTokenClassifier.from_pretrained(model_path)
    ev_trainer = Trainer(model=ev_model, data_collator=collator)
    val_pred = ev_trainer.predict(test_dataset=val_dataset)
    val_logits = val_pred.predictions
    val_labels = val_pred.label_ids
    val_mask = val_labels != -100

    T = fit_temperature(val_logits, val_labels, val_mask)
    probs_cal = softmax_last_dim_np(val_logits / T)[..., 1][val_mask]
    y_true_val = val_labels[val_mask].astype(int)
    _, thr_val = f1_max_calc(y_true_val, probs_cal)

    calibration_info = {"temperature": float(T), "threshold": float(thr_val)}
    with open(os.path.join(run_dirs["config"], "calibration.json"), "w") as f:
        json.dump(calibration_info, f, indent=2)

    try:
        mlflow.log_params({"cal_temperature": float(T), "cal_threshold": float(thr_val)})
        mlflow.log_artifact(os.path.join(run_dirs["config"], "calibration.json"))
    except Exception:
        pass

    # Test evaluation (raw + calibrated)
    print("Evaluating (raw + calibrated)…")
    evaluate_model(
        model_path=model_path,
        testNOX_dataset=testNOX_dataset,
        testPDB_dataset=testPDB_dataset,
        tokenizer=tokenizer,
        run_dirs=run_dirs,
        use_msa=args.use_msa,
        msa_dim=args.msa_dim,
        temperature=T,
        # val_based_threshold=thr_val,
    )

    # Save reproducibility recipe + manifest
    reproduce = {
        "base_model_path": f"facebook/{args.model_name}",
        "model_path": run_dirs["model"],
        "datasets": {
            "train_sequences": args.train_sequences,
            "train_labels": args.train_labels,
            "val_sequences": args.val_sequences,
            "val_labels": args.val_labels,
            "testNox_sequences": args.testNox_sequences,
            "testNox_labels": args.testNox_labels,
            "testPDB_sequences": args.testPDB_sequences,
            "testPDB_labels": args.testPDB_labels,
            "msa_train_pkl": args.msa_train_pkl,
            "msa_val_pkl": args.msa_val_pkl,
            "msa_testNox_pkl": args.msa_testNox_pkl,
            "msa_testPDB_pkl": args.msa_testPDB_pkl,
        },
        "max_sequence_length_train": args.max_sequence_length_train,
        "max_sequence_length_test": args.max_sequence_length_test,
        "head": args.head,
        "loss": args.loss,
        "layer_pooling": args.layer_pooling,
        "use_msa": args.use_msa,
        "msa_dim": args.msa_dim,
        "msa_fuse": args.msa_fuse,
    }
    with open(os.path.join(run_dirs["config"], "reproduce.json"), "w") as f:
        json.dump(reproduce, f, indent=2)

    write_manifest(
        run_dirs,
        args=args,
        extra={
            "experiment_name": args.experiment_name,
            "mlflow_tracking_uri": mlflow.get_tracking_uri(),
        }
    )

    try:
        mlflow.log_artifact(os.path.join(run_dirs["config"], "reproduce.json"))
        mlflow.log_artifact(os.path.join(run_dirs["config"], "manifest.json"))
        mlflow.log_artifacts(run_dirs["root"], artifact_path="run")
    except Exception:
        pass

    # Finish W&B
    try:
        if wb_run is not None:
            wb_run.finish()
    except Exception:
        pass

    print("Done.")

if __name__ == "__main__":
    main()