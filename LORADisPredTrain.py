"""
ESM2 token-classification with LoRA, Hugging Face Trainer, W&B, MLflow — per-run
folders under ./outputs, shared W&B dir, robust MLflow, and SOTA-leaning options
(heads, losses, layer pooling, calibration).

Run examples:

# (A) Random split validation (default)
python train.py \
  --run_name "esm2_lora_test" \
  --model_name esm2_t33_650M_UR50D \
  --max_sequence_length_train 1022 \
  --max_sequence_length_test 1022 \
  --numberofepochs 8 --numberofbatch 12 --learning_rate 1e-4 \
  --scheduler cosine --warmup_ratio 0.1 --weight_decay 0.01 \
  --gradient_clip 1.0 --grad_accum_steps 4 --amp bf16 \
  --rank 16 --lora_alpha 32 --lora_dropout 0.15 \
  --lora_target "q_proj,k_proj,v_proj,o_proj,fc_in,fc_out" \
  --head mlp --head_hidden_dim 512 --head_dropout 0.2 \
  --loss ce --layer_pooling token_weighted \
  --data_backend auto \
  --train_sequences /path/train_sequences.pkl \
  --train_labels    /path/train_labels.pkl \
  --testNox_sequences /path/testNox_sequences.pkl \
  --testNox_labels    /path/testNox_labels.pkl \
  --testPDB_sequences /path/testPDB_sequences.pkl \
  --testPDB_labels    /path/testPDB_labels.pkl \
  --val_split 0.2 --val_seed 42

# (B) Explicit validation set
python train.py \
  --run_name "esm2_lora_with_explicit_val" \
  --model_name esm2_t33_650M_UR50D \
  --max_sequence_length_train 1022 \
  --max_sequence_length_test 1022 \
  --numberofepochs 8 --numberofbatch 12 --learning_rate 1e-4 \
  --scheduler cosine --warmup_ratio 0.1 --weight_decay 0.01 \
  --gradient_clip 1.0 --grad_accum_steps 4 --amp bf16 \
  --rank 16 --lora_alpha 32 --lora_dropout 0.15 \
  --lora_target "q_proj,k_proj,v_proj,o_proj,fc_in,fc_out" \
  --head conformer --head_dropout 0.2 \
  --conformer_heads 4 --conformer_kernel 9 --conformer_dropout 0.1 \
  --loss ce --layer_pooling token_weighted \
  --data_backend auto \
  --train_sequences /path/train_sequences.pkl \
  --train_labels    /path/train_labels.pkl \
  --val_sequences   /path/val_sequences.pkl \
  --val_labels      /path/val_labels.pkl \
  --testNox_sequences /path/testNox_sequences.pkl \
  --testNox_labels    /path/testNox_labels.pkl \
  --testPDB_sequences /path/testPDB_sequences.pkl \
  --testPDB_labels    /path/testPDB_labels.pkl
"""

import os
import re
import json
import shutil
import random
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import getpass

# Automatically gets the current cluster username (e.g., 'wasicse')
user = getpass.getuser()

# Define the base directory on the work volume
base_work_dir = f"/work/{user}/huggingface"

# Set environment variables
os.environ["HF_HOME"] = base_work_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(base_work_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(base_work_dir, "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(base_work_dir, "hub")

# Optional: Create the directories if they don't exist to prevent errors
os.makedirs(os.environ["HF_HOME"], exist_ok=True)



from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    log_loss,
)

from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

import mlflow
import wandb

import fsspec
import s3fs
import argparse
import time
from contextlib import contextmanager
import transformers, peft

try:
    import torchcrf  # optional
except Exception:
    torchcrf = None

# Optional external summary; safe import
try:
    from esm_results import ClassificationScore
except Exception:
    ClassificationScore = None

print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
warnings.filterwarnings("ignore")
np.set_printoptions(precision=3)

# ------------- Reproducibility -------------
SEED = 2515
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Determinism (may slow training)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

# MLflow system metrics (optional)
os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "true")

# Quiet progress output
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("WANDB_SILENT", "true")

# ------------- Output & Tracking Helpers -------------

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
        # shared W&B cache across runs (NOT inside run_dir)
        "wandb_shared": os.path.join(output_root, "wandb"),
    }
    for k, d in sub.items():
        if k != "wandb_shared":
            os.makedirs(d, exist_ok=True)
    os.makedirs(sub["wandb_shared"], exist_ok=True)

    # Optional convenience symlink to latest
    try:
        latest_link = os.path.join(output_root, "latest")
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            if os.path.islink(latest_link):
                os.unlink(latest_link)
        os.symlink(run_dir, latest_link)
    except Exception:
        pass

    # Snapshot this script into the run for reproducibility
    try:
        src_path = os.path.abspath(__file__)
        shutil.copy(src_path, os.path.join(sub["config"], os.path.basename(src_path)))
    except Exception:
        pass

    return sub

def set_tracking(output_root: str, experiment_name: str):
    """Use local MLflow file store under outputs/mlruns unless env overrides."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_uri = os.path.join(output_root, "mlruns")
        os.makedirs(tracking_uri, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)

    # Create experiment if missing
    try:
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name)
    except Exception:
        pass
    experiment = mlflow.set_experiment(experiment_name)
    return experiment

# ------------- Backend helpers -------------

def has_minio_conf() -> bool:
    return bool(os.getenv("MINIO_ENDPOINT") or os.getenv("AWS_ACCESS_KEY_ID"))

def is_s3_like_path(p: str) -> bool:
    return isinstance(p, str) and p.startswith("s3://")

def _s3_key(path: str) -> str:
    if path.startswith("s3://"):
        return path[5:]
    return path.lstrip("/")

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

# ------------- Layer pooling & Heads -------------

@dataclass
class HeadConfig:
    head: str = "linear"            # linear|mlp|crf|conformer
    hidden: int = 0                 # hidden dim for MLP (0 => linear)
    dropout: float = 0.1
    layer_pooling: str = "last"     # last|last4|weighted|token_weighted
    lora_target: Optional[List[str]] = None  # override target modules
    # Conformer-lite knobs (optional)
    conformer_heads: int = 4
    conformer_kernel: int = 9
    conformer_dropout: float = 0.1

class WeightedLayerPool(nn.Module):
    def __init__(self, n_layers: int, init_last_only: bool = True):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))
        if init_last_only:
            with torch.no_grad():
                self.weights[-1] = 1.0

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # hidden_states: list length L, each [B, T, H]
        W = torch.softmax(self.weights, dim=0)
        stacked = torch.stack(hidden_states, dim=0)  # [L, B, T, H]
        pooled = torch.einsum("l,lbth->bth", W, stacked)
        return pooled

# >>> NEW: per-token dynamic layer mixing
class TokenWiseLayerPool(nn.Module):
    """
    Learns a [B,T,L] softmax over layers per token, then mixes [L,B,T,H] -> [B,T,H].
    Cheap and effective when different residues benefit from different depths.
    """
    def __init__(self, n_layers: int, hidden: int):
        super().__init__()
        self.proj = nn.Linear(n_layers, hidden)
        self.gate = nn.Linear(hidden, n_layers)

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # hidden_states: list length L, each [B, T, H]
        # stack -> [L,B,T,H] -> [B,T,L,H]
        x = torch.stack(hidden_states, dim=0).permute(1, 2, 0, 3)
        # compress token features across H to score layers: [B,T,L]
        scores = x.mean(dim=-1)
        w = torch.softmax(self.gate(F.gelu(self.proj(scores))), dim=-1)  # [B,T,L]
        pooled = torch.einsum("btl,btlh->bth", w, x)  # [B,T,H]
        return pooled

# >>> NEW: lightweight conv + attention head block
class ConformerLite(nn.Module):
    """
    A tiny head: depthwise separable conv for local motifs + residual MHA for context.
    Keeps params small and works well on protein token tasks.
    """
    def __init__(self, d: int, n_heads: int = 4, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        self.dw = nn.Conv1d(d, d, kernel_size, padding=kernel_size // 2, groups=d)
        self.pw = nn.Linear(d, d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # DW-Conv (token axis) + pointwise
        y = self.dw(x.transpose(1, 2)).transpose(1, 2)
        y = self.pw(y)
        x = x + self.drop(self.n1(y))
        # Self-attn (respect padding mask if provided)
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

        # ---- LoRA targets
        TASK_TOKEN_CLS = getattr(TaskType, "TOKEN_CLASSIFICATION", TaskType.TOKEN_CLS)
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

        # ---- Layer pooling over transformer layers
        if head_cfg.layer_pooling == "last4":
            self.pool = lambda hs: torch.mean(torch.stack(hs[-4:], dim=0), dim=0)
            self.pool_module = None
        elif head_cfg.layer_pooling == "weighted":
            n_layers = getattr(base.config, 'num_hidden_layers', 12)
            self.pool_module = WeightedLayerPool(n_layers=n_layers + 1)  # include embeddings
            self.pool = lambda hs: self.pool_module(hs)
        elif head_cfg.layer_pooling == "token_weighted":
            n_layers = getattr(base.config, 'num_hidden_layers', 12)
            self.pool_module = TokenWiseLayerPool(n_layers=n_layers + 1, hidden=hidden_size // 2)
            self.pool = lambda hs: self.pool_module(hs)
        else:
            self.pool = lambda hs: hs[-1]
            self.pool_module = None

        self.dropout = nn.Dropout(head_cfg.dropout)

        # >>> NEW: optional Conformer-lite block before classifier
        self.conformer = None
        if self.head_cfg.head == "conformer":
            heads = getattr(head_cfg, "conformer_heads", 4)
            ksize = getattr(head_cfg, "conformer_kernel", 9)
            cdrop = getattr(head_cfg, "conformer_dropout", head_cfg.dropout)
            self.conformer = ConformerLite(hidden_size, n_heads=heads, kernel_size=ksize, dropout=cdrop)

        # ---- Head: linear | mlp | crf | conformer (classifier sits after optional conformer)
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
            # linear or conformer (classifier is simple linear after conformer)
            self.emission = nn.Linear(hidden_size, num_labels)
            self.crf = None
            self.head_net = None

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states
        x = self.pool(hs)
        x = self.dropout(x)

        # >>> NEW: run conformer-lite if enabled
        if self.conformer is not None:
            x = self.conformer(x, attention_mask=attention_mask)

        if self.head_net is not None:
            emissions = self.head_net(x)
        else:
            emissions = self.emission(x)

        outputs = {"logits": emissions}

        # if self.crf is not None:
            # mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(emissions[..., 0], dtype=torch.bool)
            # if labels is not None:
                # safe_labels = labels.clone()
                # if (safe_labels == -100).any():
                    # safe_labels = safe_labels.masked_fill(safe_labels == -100, 0)
                    # mask = mask & (labels != -100)
                # nll = -self.crf(emissions, safe_labels, mask=mask, reduction='mean')
                # outputs["loss"] = nll
                
        if self.crf is not None:
            mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(emissions[..., 0], dtype=torch.bool)

            if labels is not None:
                safe_labels = labels.clone()

                ignore = (safe_labels == -100)
                safe_labels = safe_labels.masked_fill(ignore, 0)

                mask = mask & (~ignore)

                # torchcrf requirement: first timestep must be on for all sequences
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

    # ---- save/load: include head/targets
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

# ------------- Losses -------------

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
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        if self.alpha is not None:
            at = torch.full_like(pt, fill_value=self.alpha)
            at = torch.where(target == 1, at, 1 - at)
            loss = -at * (1 - pt) ** self.gamma * logpt
        else:
            loss = - (1 - pt) ** self.gamma * logpt
        return loss.mean()

# ------------- TrainingArguments helper -------------

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
    amp: str
):
    base = dict(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=numberofbatch,
        per_device_eval_batch_size=numberofbatch,
        num_train_epochs=numberofepochs,
        weight_decay=weight_decay,
        logging_steps=200,
        seed=seed,
        remove_unused_columns=False,
        disable_tqdm=True,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        save_total_limit=7,
        report_to=["wandb"],
        push_to_hub=False,
        lr_scheduler_type=scheduler,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=grad_accum_steps,
        max_grad_norm=gradient_clip,
    )

    params = set(signature(TrainingArguments.__init__).parameters.keys())
    allowed = {k: v for k, v in base.items() if k in params}
    eval_key = "evaluation_strategy" if "evaluation_strategy" in params else ("eval_strategy" if "eval_strategy" in params else None)
    if eval_key:
        allowed[eval_key] = "epoch"

    # AMP selection
    if amp == "fp16" and "fp16" in params:
        allowed["fp16"] = True
        if "bf16" in params: allowed["bf16"] = False
    elif amp == "bf16" and "bf16" in params and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        allowed["bf16"] = True
        if "fp16" in params: allowed["fp16"] = False
    elif amp == "off":
        if "fp16" in params: allowed["fp16"] = False
        if "bf16" in params: allowed["bf16"] = False
    else:
        # auto: prefer bf16 if supported
        try:
            if "bf16" in params and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                allowed["bf16"] = True
                if "fp16" in params: allowed["fp16"] = False
            elif "fp16" in params:
                allowed["fp16"] = True
        except Exception:
            pass

    args = TrainingArguments(**allowed)
    if eval_key is None and "evaluate_during_training" in params:
        setattr(args, "evaluate_during_training", True)
        if "eval_steps" in params and "logging_steps" in base:
            setattr(args, "eval_steps", base["logging_steps"])
    return args

# ------------- Metrics helpers -------------

def truncate_labels(labels: List[List[int]], max_length: int) -> List[List[int]]:
    return [lab[:max_length] for lab in labels]

def softmax_last_dim(x: np.ndarray) -> np.ndarray:
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
    # Supports HF Trainer call signature and our manual (logits, labels)
    if isinstance(eval_pred, tuple):
        logits, labels = eval_pred
    else:
        logits = eval_pred.predictions
        labels = eval_pred.label_ids

    probs = softmax_last_dim(logits)[..., 1]
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

    # Calibration-sensitive
    try:
        out["NLL"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        out["NLL"] = float("nan")
    try:
        brier = np.mean((y_prob - y_true) ** 2)
        out["Brier"] = float(brier)
    except Exception:
        out["Brier"] = float("nan")

    return out

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, loss_type: str = "ce", focal_gamma: float = 2.0, focal_alpha: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_type = loss_type
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, ignore_index=-100) if loss_type == "focal" else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # For CRF, let the model compute its loss. For others, compute here.
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

# ------------- Data I/O (S3 / MinIO optional) -------------

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
    else:
        return s3fs.S3FileSystem(anon=False)

def read_train_val_datasets(
    s3: s3fs.S3FileSystem | None,
    tokenizer,
    max_seq_len: int,
    train_seq_path: str,
    train_lbl_path: str,
    *,
    val_seq_path: Optional[str] = None,
    val_lbl_path: Optional[str] = None,
    val_split: float = 0.2,
    random_state: int = 42,
    data_backend: str = "auto",
):
    # Load training
    train_sequences = load_pickle(train_seq_path, fs=s3, backend=data_backend)
    train_labels    = load_pickle(train_lbl_path,  fs=s3, backend=data_backend)

    # Decide validation source
    use_explicit_val = bool(val_seq_path) or bool(val_lbl_path)
    if use_explicit_val:
        if not (val_seq_path and val_lbl_path):
            raise ValueError(
                "When specifying a separate validation set, you must provide BOTH --val_sequences and --val_labels."
            )
        val_sequences = load_pickle(val_seq_path, fs=s3, backend=data_backend)
        val_labels    = load_pickle(val_lbl_path,  fs=s3, backend=data_backend)
        if len(val_sequences) != len(val_labels):
            raise ValueError(f"Validation sequences and labels length mismatch: {len(val_sequences)} vs {len(val_labels)}")
    else:
        tr_seqs, val_seqs, tr_labs, val_labs = train_test_split(
            train_sequences, train_labels, test_size=val_split, random_state=random_state
        )
        train_sequences, train_labels = tr_seqs, tr_labs
        val_sequences,   val_labels   = val_seqs, val_labs

    # Tokenize
    tr_tok  = tokenizer(train_sequences, padding=False, truncation=True, max_length=max_seq_len,
                        add_special_tokens=False, is_split_into_words=False)
    val_tok = tokenizer(val_sequences,  padding=False, truncation=True, max_length=max_seq_len,
                        add_special_tokens=False, is_split_into_words=False)

    # Truncate labels to match max length
    train_labels = truncate_labels(train_labels, max_seq_len)
    val_labels   = truncate_labels(val_labels,   max_seq_len)

    # Build datasets
    train_dataset = Dataset.from_dict({k: v for k, v in tr_tok.items()})
    train_dataset = train_dataset.add_column("labels", train_labels)

    val_dataset = Dataset.from_dict({k: v for k, v in val_tok.items()})
    val_dataset = val_dataset.add_column("labels", val_labels)

    # Class weights from *training* labels only
    classes = np.array([0, 1])
    flat_tr = np.array([y for seq in train_labels for y in seq], dtype=int)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=flat_tr)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return train_dataset, val_dataset, class_weights

def map_neg1_to_ignore(labels, ignore=-100):
    # labels: List[List[int]]
    return [[(ignore if y == -1 else y) for y in seq] for seq in labels]

def read_test_datasets(s3: s3fs.S3FileSystem | None, tokenizer, max_seq_len: int, test_nox_seq_path: str, test_nox_lbl_path: str, test_pdb_seq_path: str, test_pdb_lbl_path: str, data_backend: str = "auto"):
    testNox_sequences = load_pickle(test_nox_seq_path, fs=s3, backend=data_backend)
    testNox_labels    = load_pickle(test_nox_lbl_path,  fs=s3, backend=data_backend)
    testPDB_sequences = load_pickle(test_pdb_seq_path, fs=s3, backend=data_backend)
    testPDB_labels    = load_pickle(test_pdb_lbl_path,  fs=s3, backend=data_backend)

    testNox_tok = tokenizer(testNox_sequences, padding=False, truncation=True, max_length=max_seq_len, add_special_tokens=False, is_split_into_words=False)
    testPDB_tok = tokenizer(testPDB_sequences, padding=False, truncation=True, max_length=max_seq_len, add_special_tokens=False, is_split_into_words=False)
    testNox_labels = map_neg1_to_ignore(testNox_labels, ignore=-100)
    testPDB_labels = map_neg1_to_ignore(testPDB_labels, ignore=-100)

    testNox_labels = truncate_labels(testNox_labels, max_seq_len)
    testPDB_labels = truncate_labels(testPDB_labels, max_seq_len)

    testNOX_dataset = Dataset.from_dict({k: v for k, v in testNox_tok.items()})
    testNOX_dataset = testNOX_dataset.add_column("labels", testNox_labels)

    testPDB_dataset = Dataset.from_dict({k: v for k, v in testPDB_tok.items()})
    testPDB_dataset = testPDB_dataset.add_column("labels", testPDB_labels)

    return testNOX_dataset, testPDB_dataset

# ------------- Calibration -------------

def softmax_last_dim_torch(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x, dim=-1)

def fit_temperature(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    valid = mask
    Z = logits[valid]
    y = labels[valid].astype(int)
    best_T, best_nll = 1.0, float('inf')
    for T in np.linspace(0.5, 3.0, 26):
        scaled = Z / T
        p = softmax_last_dim(scaled)[..., 1]
        try:
            nll = log_loss(y, p, labels=[0, 1])
        except Exception:
            continue
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T

# ------------- Train / Evaluate -------------

def build_model(
    model_name: str,
    head_cfg: HeadConfig,
    *,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float
) -> ESMTokenClassifier:
    return ESMTokenClassifier(
        model_name, num_labels=2, head_cfg=head_cfg,
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
    )

def make_trainer_args(args: argparse.Namespace, run_dirs: Dict[str, str]) -> TrainingArguments:
    return make_training_args(
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
    )

def train_lora(*, args: argparse.Namespace, model_name: str, train_dataset: Dataset, eval_dataset: Dataset,
               tokenizer, class_weights: torch.Tensor, run_dirs: Dict[str, str], head_cfg: HeadConfig) -> Tuple[str, Dict[str, float]]:
    print(f"epochs: {args.numberofepochs} | batch: {args.numberofbatch} | model: {model_name}")
    print(f"head: {head_cfg.head} | pooling: {head_cfg.layer_pooling} | loss: {args.loss}")

    # Build model with LoRA params applied at construction time
    model = build_model(
        model_name, head_cfg,
        lora_r=args.rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout
    )

    # >>> NEW: log trainable parameter count (PEFT + head)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    mlflow.log_metrics({"trainable_params": float(trainable), "total_params": float(total),
                        "trainable_pct": float(100.0 * trainable / max(1,total))})

    training_args = make_trainer_args(args, run_dirs)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics_eval,
        class_weights=class_weights,
        loss_type=args.loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
    )


    trainer.train()

    # save model (adapters + head)
    save_path = run_dirs["model"]
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    eval_metrics = trainer.evaluate()

    mlflow.log_metrics({k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))})
    try:
        if getattr(trainer.state, "best_metric", None) is not None:
            mlflow.log_metric("best_metric", float(trainer.state.best_metric))
        if getattr(trainer.state, "best_model_checkpoint", None):
            mlflow.log_param("best_model_checkpoint", trainer.state.best_model_checkpoint)
    except Exception:
        pass

    with open(os.path.join(run_dirs["results"], "val_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in eval_metrics.items()}, f, indent=2)

    return save_path, {k: float(v) for k, v in eval_metrics.items()}

def evaluate_model(model_path: str, testNOX_dataset: Dataset, testPDB_dataset: Dataset, tokenizer, run_dirs: Dict[str, str], temperature: Optional[float] = None, val_based_threshold: Optional[float] = None):
    model = ESMTokenClassifier.from_pretrained(model_path)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = Trainer(model=model, data_collator=data_collator)

    def _eval(dataset: Dataset, tag: str):
        pred_out = trainer.predict(test_dataset=dataset)
        logits = pred_out.predictions
        labels = pred_out.label_ids

        probs_raw = softmax_last_dim(logits)[..., 1]
        mask = labels != -100
        y_true = labels[mask].astype(int)
        y_score_raw = probs_raw[mask]

        # calibrated
        if temperature is not None:
            probs_cal = softmax_last_dim(logits / temperature)[..., 1]
            y_score_cal = probs_cal[mask]
        else:
            y_score_cal = y_score_raw

        def metrics(y, p):
            try:
                auc = roc_auc_score(y, p)
            except Exception:
                auc = float("nan")
            try:
                aps = average_precision_score(y, p)
            except Exception:
                aps = float("nan")
            f1m, thr = f1_max_calc(y, p) if val_based_threshold is None else (None, val_based_threshold)
            try:
                nll = log_loss(y, p, labels=[0,1])
            except Exception:
                nll = float("nan")
            brier = float(np.mean((p - y) ** 2))
            return {"auc": float(auc), "Average Precision": float(aps), "F1_max": float(f1m) if f1m is not None else float("nan"), "F1_thr": float(thr),
                    "NLL": float(nll), "Brier": brier}

        raw_metrics = metrics(y_true, y_score_raw)
        cal_metrics = metrics(y_true, y_score_cal)

        # Save arrays & metrics
        np.save(os.path.join(run_dirs["results"], f"{tag}_y_true.npy"), y_true)
        np.save(os.path.join(run_dirs["results"], f"{tag}_y_score_raw.npy"), y_score_raw)
        np.save(os.path.join(run_dirs["results"], f"{tag}_y_score_cal.npy"), y_score_cal)
        with open(os.path.join(run_dirs["results"], f"{tag}_metrics_raw.json"), "w") as f:
            json.dump(raw_metrics, f, indent=2)
        with open(os.path.join(run_dirs["results"], f"{tag}_metrics_calibrated.json"), "w") as f:
            json.dump(cal_metrics, f, indent=2)

        mlflow.log_metrics({f"{tag}_raw_{k}": v for k, v in raw_metrics.items() if isinstance(v, float)})
        mlflow.log_metrics({f"{tag}_cal_{k}": v for k, v in cal_metrics.items() if isinstance(v, float)})
        for fn in [f"{tag}_y_true.npy", f"{tag}_y_score_raw.npy", f"{tag}_y_score_cal.npy", f"{tag}_metrics_raw.json", f"{tag}_metrics_calibrated.json"]:
            mlflow.log_artifact(os.path.join(run_dirs["results"], fn))

    print("NOX Test results"); _eval(testNOX_dataset, "testNOX")
    print("PDB Test results"); _eval(testPDB_dataset, "testPDB")

# ------------- Optional: Download local ESM weights from MinIO -------------

def download_minio_esm_weights(s3: s3fs.S3FileSystem | None):
    if os.getenv("DOWNLOAD_LOCAL_ESM_WEIGHTS", "0") != "1" or s3 is None:
        return
    os.makedirs("facebook", exist_ok=True)
    targets = [
        ("/hoque-research-bucket/LORADispredData/facebook/esm2_t12_35M_UR50D.pt", "/dev/shm/facebook/esm2_t12_35M_UR50D.pt", "facebook/esm2_t12_35M_UR50D.pt"),
        ("/hoque-research-bucket/LORADispredData/facebook/esm2_t33_650M_UR50D.pt", "/dev/shm/facebook/esm2_t33_650M_UR50D.pt", "facebook/esm2_t33_650M_UR50D.pt"),
    ]
    for s3_src, shm_dst, link_path in targets:
        os.makedirs(os.path.dirname(shm_dst), exist_ok=True)
        if not os.path.exists(shm_dst):
            print(f"Downloading {s3_src} -> {shm_dst}")
            s3.get(_s3_key(s3_src), shm_dst)
        if not os.path.exists(link_path):
            os.symlink(shm_dst, link_path)

# ------------- Manifest & Config Persistence -------------

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

# ------------- Main -------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_name", default="ESM2_LoRA_Dispred", help="Run name")
    parser.add_argument("-p", "--test", default="True", help="Whether this is a test run (True/False)")
    parser.add_argument("-e", "--max_sequence_length_train", default=1000, type=int)
    parser.add_argument("-l", "--max_sequence_length_test", default=5000, type=int)
    parser.add_argument("-m", "--model_name", default="esm2_t12_35M_UR50D")
    parser.add_argument("-n", "--numberofepochs", default=1, type=int)
    parser.add_argument("-b", "--numberofbatch", default=12, type=int)
    parser.add_argument("-a", "--lora_alpha", default=1.0, type=float)
    parser.add_argument("-d", "--lora_dropout", default=0.2, type=float)
    parser.add_argument("-k", "--rank", default=2, type=int)
    parser.add_argument("-t", "--learning_rate", default=5.701568055793089e-04, type=float)

    # Training knobs
    parser.add_argument("--scheduler", default="cosine",
                        choices=["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"])
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.2)
    parser.add_argument("--gradient_clip", type=float, default=0.5)  # Trainer uses max_grad_norm
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--amp", default="auto", choices=["auto","fp16","bf16","off"], help="Mixed precision mode")

    # LoRA targets override (comma-separated)
    parser.add_argument("--lora_target", default="",
                        help='Comma-separated modules to target (e.g. "q_proj,k_proj,v_proj,o_proj,fc_in,fc_out"). Empty uses autodetect.')

    # Heads / Loss / Pooling
    parser.add_argument("--head", choices=["linear", "mlp", "crf", "conformer"], default="linear")  # <<< NEW option
    parser.add_argument("--head_hidden_dim", type=int, default=0, help="Hidden dim for MLP head; 0 uses linear")
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=None)
    parser.add_argument("--layer_pooling", choices=["last", "last4", "weighted", "token_weighted"], default="last")  # <<< NEW option

    # Conformer-lite knobs
    parser.add_argument("--conformer_heads", type=int, default=4)
    parser.add_argument("--conformer_kernel", type=int, default=9)
    parser.add_argument("--conformer_dropout", type=float, default=0.1)

    parser.add_argument("--data_backend", choices=["auto", "local", "s3"], default="auto", help="Dataset backend")
    parser.add_argument("--output_dir", default=os.getenv("OUTPUT_DIR", "outputs"), help="Root folder for all runs")

    parser.add_argument("--train_sequences", default="/hoque-research-bucket/LORADispredData/Dataset/train_sequences.pkl")
    parser.add_argument("--train_labels",    default="/hoque-research-bucket/LORADispredData/Dataset/train_labels.pkl")

    # Explicit validation support
    parser.add_argument("--val_sequences", default="", help="Optional: path to validation sequences .pkl")
    parser.add_argument("--val_labels",    default="", help="Optional: path to validation labels .pkl")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction for random train/val split when no explicit validation set is provided")
    parser.add_argument("--val_seed", type=int, default=42,
                        help="Random state for train/val split when no explicit validation set is provided")

    parser.add_argument("--testNox_sequences", default="/hoque-research-bucket/LORADispred/testNox_sequences.pkl")
    parser.add_argument("--testNox_labels",    default="/hoque-research-bucket/LORADispred/testNox_labels.pkl")
    parser.add_argument("--testPDB_sequences", default="/hoque-research-bucket/LORADispredData/testPDB_sequences_Spot.pkl")
    parser.add_argument("--testPDB_labels",    default="/hoque-research-bucket/LORADispredData/testPDB_labels_Spot.pkl")

    return parser.parse_args()

def main():
    args = parse_args()

    # Prepare run directories and shared W&B dir
    run_dirs = prepare_run_dirs(args.run_name, output_root=args.output_dir)
    WANDB_DIR = os.path.join(args.output_dir, "wandb")
    os.makedirs(WANDB_DIR, exist_ok=True)
    os.environ["WANDB_DIR"] = WANDB_DIR          # shared across runs
    os.environ.setdefault("WANDB_MODE", "offline")

    print("\n" + "#" * 40 + " Starting the ML Pipeline " + "#" * 40 + "\n")
    print(f"Run dir: {run_dirs['root']}")
    print(f"Model: facebook/{args.model_name} | Head: {args.head} | Loss: {args.loss} | Pool: {args.layer_pooling}")

    # MLflow
    experiment_name = "Evaluate_ESM2FinetuneV3"
    set_tracking(args.output_dir, experiment_name)
    mlflow.end_run()
    experiment = mlflow.set_experiment(experiment_name)
    print(f"Experiment_id: {experiment.experiment_id}")
    mlflow.start_run(run_name=args.run_name)

    # W&B
    wandb_run = wandb.init(
        project=experiment_name,
        name=args.run_name,
        notes="ESM2 LoRA token-classification (DisPred) with heads/loss/pooling + calibration",
        dir=os.environ.get("WANDB_DIR"),
    )

    # Log params
    mlflow.log_params({
        **{k: getattr(args, k) for k in [
            "max_sequence_length_train","max_sequence_length_test","model_name",
            "numberofepochs","numberofbatch","lora_alpha","lora_dropout","rank",
            "learning_rate","data_backend","head","loss","focal_gamma","focal_alpha",
            "layer_pooling","scheduler","warmup_ratio","weight_decay","gradient_clip",
            "grad_accum_steps","amp","head_hidden_dim","head_dropout","lora_target",
            # NEW
            "val_sequences","val_labels","val_split","val_seed",
            "conformer_heads","conformer_kernel","conformer_dropout"
        ]},
        "run_dir": run_dirs["root"],
    })

    # Data backends
    use_s3 = (args.data_backend == "s3") or (args.data_backend == "auto" and has_minio_conf())
    s3 = make_s3fs() if use_s3 else None
    if use_s3:
        download_minio_esm_weights(s3)

    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{args.model_name}")

    t0 = time.time()
    train_dataset, val_dataset, class_weights = read_train_val_datasets(
        s3=s3, tokenizer=tokenizer, max_seq_len=args.max_sequence_length_train,
        train_seq_path=args.train_sequences, train_lbl_path=args.train_labels,
        val_seq_path=(args.val_sequences or None),
        val_lbl_path=(args.val_labels or None),
        val_split=args.val_split, random_state=args.val_seed,
        data_backend=args.data_backend,
    )
    testNOX_dataset, testPDB_dataset = read_test_datasets(
        s3=s3, tokenizer=tokenizer, max_seq_len=args.max_sequence_length_test,
        test_nox_seq_path=args.testNox_sequences, test_nox_lbl_path=args.testNox_labels,
        test_pdb_seq_path=args.testPDB_sequences, test_pdb_lbl_path=args.testPDB_labels,
        data_backend=args.data_backend,
    )
    data_read_time_min = (time.time() - t0) / 60.0

    head_cfg = HeadConfig(
        head=args.head,
        hidden=args.head_hidden_dim,
        dropout=args.head_dropout,
        layer_pooling=args.layer_pooling,
        lora_target=[s.strip() for s in args.lora_target.split(",") if s.strip()] or None,
        conformer_heads=args.conformer_heads,
        conformer_kernel=args.conformer_kernel,
        conformer_dropout=args.conformer_dropout,
    )

    print("Training…")
    t1 = time.time()
    model_path, val_metrics = train_lora(
        args=args,
        model_name=args.model_name,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        class_weights=class_weights,
        run_dirs=run_dirs,
        head_cfg=head_cfg,
    )
    training_time_min = (time.time() - t1) / 60.0
    print(f"Model saved at: {model_path}")

    # ---- Calibration on validation ----
    print("Calibrating on validation set…")
    model_cal = ESMTokenClassifier.from_pretrained(model_path)
    ev_trainer = Trainer(
    model=model_cal,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
)
    val_pred = ev_trainer.predict(test_dataset=val_dataset)
    val_logits = val_pred.predictions
    val_labels = val_pred.label_ids
    val_mask = val_labels != -100

    T = fit_temperature(val_logits, val_labels, val_mask)
    probs_cal = softmax_last_dim(val_logits / T)[..., 1][val_mask]
    y_true_val = val_labels[val_mask].astype(int)
    _, thr_val = f1_max_calc(y_true_val, probs_cal)

    calibration_info = {"temperature": float(T), "threshold": float(thr_val)}
    with open(os.path.join(run_dirs["config"], "calibration.json"), "w") as f:
        json.dump(calibration_info, f, indent=2)
    mlflow.log_params({"cal_temperature": float(T), "cal_threshold": float(thr_val)})
    mlflow.log_artifact(os.path.join(run_dirs["config"], "calibration.json"))

    print("Evaluating (raw + calibrated)…")
    evaluate_model(
        model_path=model_path,
        testNOX_dataset=testNOX_dataset,
        testPDB_dataset=testPDB_dataset,
        tokenizer=tokenizer,
        run_dirs=run_dirs,
        temperature=T,
        val_based_threshold=thr_val,
    )

    mlflow.log_metrics({"data_read_time_min": data_read_time_min, "training_time_min": training_time_min})

    # ---- Repro recipe & manifest ----
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
        },
        "max_sequence_length_train": args.max_sequence_length_train,
        "max_sequence_length_test": args.max_sequence_length_test,
        "tokenizer_name": f"facebook/{args.model_name}",
        "head": args.head,
        "loss": args.loss,
        "layer_pooling": args.layer_pooling,
    }
    with open(os.path.join(run_dirs["config"], "reproduce_test.json"), "w") as f:
        json.dump(reproduce, f, indent=2)

    write_manifest(run_dirs, args=args, extra={"experiment_name": experiment_name, "mlflow_tracking_uri": mlflow.get_tracking_uri()})

    # Optional external summary table
    if ClassificationScore is not None:
        try:
            y_nox = np.load(os.path.join(run_dirs["results"], "testNOX_y_true.npy"))
            y_proba_nox = np.load(os.path.join(run_dirs["results"], "testNOX_y_score_raw.npy"))
            y_pdb = np.load(os.path.join(run_dirs["results"], "testPDB_y_true.npy"))
            y_proba_pdb = np.load(os.path.join(run_dirs["results"], "testPDB_y_score_raw.npy"))
            ClassificationScore(args.run_name, SEED, y_nox, y_proba_nox, y_pdb, y_proba_pdb, data_read_time_min, training_time_min)
        except Exception:
            pass

    try:
        mlflow.log_artifacts(run_dirs["root"], artifact_path="run")
    except Exception:
        pass

    print("Done.")

if __name__ == "__main__":
    main()
