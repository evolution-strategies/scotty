from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import laya.train as lt
from laya.train import (
    TrainConfig,
    encode_item,
    items_from_rows,
    load_checkpoint,
    read_jsonl,
    save_checkpoint,
    soft_ce_loss,
    uses_parallel_layout,
)
from laya.common import collate_items


def bf16_forward(model, batch, device, amp: bool, detach_encoder: bool):
    args = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["marker_pos"].to(device),
        batch["marker_mask"].to(device),
        batch["qtype"].to(device),
    )
    kwargs = {"detach_encoder": detach_encoder}
    if "option_ids" in batch:
        kwargs.update(position_ids=batch["position_ids"].to(device), option_ids=batch["option_ids"].to(device))
    if amp:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _act = model(*args, **kwargs)
    else:
        logits, _act = model(*args, **kwargs)
    return logits.float()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    device = torch.device("cuda", 0)
    model, tok, cfg = load_checkpoint(args.base)
    if uses_parallel_layout(cfg):
        raise ValueError("released root checkpoint unexpectedly uses parallel option layout")
    if cfg.get("amp_dtype") != "bf16":
        raise ValueError(f"root checkpoint amp_dtype is {cfg.get('amp_dtype')!r}, expected 'bf16'")
    max_len, head_max_len = 1024, 256

    train_rows = read_jsonl(args.train)
    validation_rows = read_jsonl(args.validation)
    train_items, train_skipped = items_from_rows(tok, train_rows, max_len, head_max_len, label_smoothing=0.0)
    validation_items, validation_skipped = items_from_rows(tok, validation_rows, max_len, head_max_len, label_smoothing=0.0)
    if len(train_items) != 10411 or len(validation_items) != 1590 or train_skipped or validation_skipped:
        raise ValueError({
            "train_items": len(train_items), "validation_items": len(validation_items),
            "train_skipped": train_skipped, "validation_skipped": validation_skipped,
        })
    for item in train_items + validation_items:
        target = item["target"]
        if abs(sum(target) - 1.0) > 1e-8 or sum(value == 1.0 for value in target) != 1 or any(value not in (0.0, 1.0) for value in target):
            raise ValueError("a converted target is not exactly one-hot")

    picked = []
    for qtype in ("choice", "noul", "score"):
        picked.extend([item for item in train_items if item["q"]["t"] == qtype][:2])
    if len(picked) != 6:
        raise ValueError("smoke batch did not cover all three Laya question types")

    lt._forward = bf16_forward
    model.to(device).train()
    encoder_params = [p for name, p in model.named_parameters() if name.startswith("encoder.") and p.requires_grad]
    head_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.") and p.requires_grad]
    if not encoder_params or not head_params:
        raise ValueError("encoder and head must both be unfrozen")
    encoder_before = encoder_params[0].detach().float().cpu().clone()
    head_before = head_params[0].detach().float().cpu().clone()
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": 2.5e-5},
        {"params": head_params, "lr": 1e-4},
    ], weight_decay=0.01)
    batch = collate_items([[encode_item(tok, item, max_len, head_max_len, parallel=False) for item in picked]], tok.pad_token_id)
    logits = bf16_forward(model, batch, device, amp=True, detach_encoder=False)
    loss = soft_ce_loss(logits, batch["target"].to(device), batch["marker_mask"].to(device))
    loss.backward()
    encoder_grad = torch.sqrt(sum((p.grad.detach().float() ** 2).sum() for p in encoder_params if p.grad is not None)).item()
    head_grad = torch.sqrt(sum((p.grad.detach().float() ** 2).sum() for p in head_params if p.grad is not None)).item()
    if not (encoder_grad > 0 and head_grad > 0 and torch.isfinite(loss)):
        raise ValueError("non-finite loss or missing encoder/head gradient")
    optimizer.step()
    encoder_delta = (encoder_params[0].detach().float().cpu() - encoder_before).abs().max().item()
    head_delta = (head_params[0].detach().float().cpu() - head_before).abs().max().item()
    if not (encoder_delta > 0 and head_delta > 0):
        raise ValueError("encoder or head did not update")

    checkpoint_dir = out_dir / "checkpoint"
    smoke_cfg = dict(cfg, max_len=max_len, head_max_len=head_max_len)
    smoke_cfg["beam_one_smoke"] = {"precision": "bf16 autocast", "loss": "hard cross-entropy via one-hot soft-ce"}
    save_checkpoint(model, tok, smoke_cfg, str(checkpoint_dir))
    del model
    torch.cuda.empty_cache()
    reloaded, reloaded_tok, reloaded_cfg = load_checkpoint(str(checkpoint_dir))
    if reloaded_cfg.get("beam_one_smoke", {}).get("precision") != "bf16 autocast":
        raise ValueError("save/reload lost smoke metadata")
    if reloaded_tok.mask_token != tok.mask_token:
        raise ValueError("save/reload changed the shared MASK token")
    if any(not torch.isfinite(value).all() for value in reloaded.state_dict().values()):
        raise ValueError("save/reload produced a non-finite tensor")

    report = {
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "precision": "torch.autocast(cuda, bfloat16)",
        "loss": float(loss.item()),
        "encoder_gradient_l2": encoder_grad,
        "head_gradient_l2": head_grad,
        "encoder_update_max_abs": encoder_delta,
        "head_update_max_abs": head_delta,
        "train_items": len(train_items),
        "validation_items": len(validation_items),
        "skipped": {"train": train_skipped, "validation": validation_skipped},
        "question_types_in_batch": [item["q"]["t"] for item in picked],
        "option_layout": "sequential",
        "mask_token": tok.mask_token,
        "full_encoder_and_head_unfrozen": True,
        "save_reload_finite": True,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
