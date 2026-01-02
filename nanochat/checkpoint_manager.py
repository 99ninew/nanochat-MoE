"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import re
import glob
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    rank = int(os.environ.get('RANK', 0))
    # Default: only rank0 logs to avoid spam.
    # Set NANOCHAT_LOG_ALL_RANKS=1 to log on every rank (useful when a non-rank0 fails early).
    log_all = os.environ.get("NANOCHAT_LOG_ALL_RANKS", "0") == "1"
    if log_all or rank == 0:
        logger.info(message)


def resolve_checkpoint_dir_and_step(source, device=None, phase=None, model_tag=None, step=None):
    """Resolve the checkpoint directory + step that would be loaded.

    This is a pure resolution helper (no model construction), useful for debugging.
    """
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)

    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)

    if step is None:
        step = find_last_step(checkpoint_dir)
    return checkpoint_dir, step

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]

    # Compatibility: some runs pad vocab_size (e.g. 50257 -> 50304) for efficiency.
    # Occasionally meta_data may store the unpadded vocab (e.g. 50266) while the checkpoint
    # weights are saved with a padded embedding table (e.g. 50304). In that case, prefer the
    # checkpoint tensor shape so we can load strictly.
    try:
        ckpt_vocab = int(model_data.get("transformer.wte.weight").shape[0])
        cfg_vocab = int(model_config_kwargs.get("vocab_size"))
        if ckpt_vocab != cfg_vocab:
            log0(
                f"Vocab size mismatch (config={cfg_vocab}, checkpoint={ckpt_vocab}); "
                "overriding config vocab_size to match checkpoint."
            )
            model_config_kwargs = dict(model_config_kwargs)
            model_config_kwargs["vocab_size"] = ckpt_vocab
    except Exception:
        # If keys are missing or shapes unavailable, fall back to meta config.
        pass
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    # Load the model state
    model.to_empty(device=device)
    # Weights are already initialized in GPT.__init__ via self.apply(self._init_weights)
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # Load the Tokenizer
    # NOTE: nanoMoE-style training typically uses GPT-2 tokens (50257) and may:
    # - pad the model vocab_size up to 50304 for efficiency, and/or
    # - add nanochat chat special tokens (len(SPECIAL_TOKENS)=9) => 50257+9=50266.
    # In all of these cases we must use GPT-2 tokenization (with nanochat special tokens).
    vocab_size = int(model_config_kwargs.get("vocab_size"))
    use_tiktoken_gpt2 = 50257 <= vocab_size <= 50304
    tokenizer = get_tokenizer(use_tiktoken_gpt2=use_tiktoken_gpt2)
    # Basic compatibility check: tokenizer ids must fit in the model embedding table.
    tok_vocab = int(tokenizer.get_vocab_size())
    assert tok_vocab <= vocab_size, (
        f"Tokenizer vocab_size ({tok_vocab}) exceeds model vocab_size ({vocab_size}). "
        "This will produce out-of-bounds token ids. "
        "Fix by using GPT-2 tokenizer (50257, padded model vocab 50304) or resizing the model vocab."
    )
    return model, tokenizer, meta_data


def find_largest_model(checkpoint_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    # 1) Prefer model tags that encode depth as d<number> (e.g. d6, d20, d6_lr1e-4).
    # Choose the largest depth, and if multiple tags share the same depth, pick the most recently updated.
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            tag_path = os.path.join(checkpoint_dir, model_tag)
            try:
                mtime = os.path.getmtime(tag_path)
            except OSError:
                mtime = 0.0
            candidates.append((model_depth, mtime, model_tag))
    if candidates:
        # depth desc, mtime desc
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]

    # 2) If depth can't be parsed from the tag name, fall back to most recently updated directory.
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
