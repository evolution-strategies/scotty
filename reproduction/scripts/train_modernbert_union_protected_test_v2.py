from __future__ import annotations

import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import laya.train as lt
from beam_one_common import prediction_metrics, read_jsonl as read_audit_jsonl, sha256_file
from laya.common import QTYPE_NAMES, collate_items
from laya.train import (TrainConfig, calibration_records, draw_option_order, encode_item, items_from_rows,
                        load_checkpoint, read_jsonl, save_checkpoint, soft_ce_loss)
from smoke_beam_one import bf16_forward
from train_laya_stage_compare import latency, metrics, records_json


ROOT = Path(os.environ["SCOTTY_REPRO_ROOT"]).expanduser().resolve()
BASE = ROOT / "checkpoints/fresh-modernbert-45bb465"
DATA = ROOT / "analysis/plain-modernbert-beam-typed-union-protected-test-v2"
OUT = ROOT / "runs/plain-modernbert-beam-typed-union-protected-test-v2-attempt2"
TYPED_VAL = ROOT / "analysis/beam-one-laya-stage-v1/validation.jsonl"
EXPECTED_BASE_SHA = "6f0fe404e2b83acd0fd7a949f2ceff0543cf56711d13f4d9c27f7c06d98b1002"
EXPECTED_ROOT_LAYA_SHA = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
MAX_LEN, HEAD_MAX_LEN = 1024, 256


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def eval_sets(model, tok, typed_items, beam_items, beam_map):
    device = torch.device("cuda", 0)
    typed_records = calibration_records(model, tok, typed_items, device, MAX_LEN, HEAD_MAX_LEN, parallel=False)
    beam_records = calibration_records(model, tok, beam_items, device, MAX_LEN, HEAD_MAX_LEN, parallel=False)
    return typed_records, metrics(typed_records), beam_records, prediction_metrics(beam_records, beam_map)


def smoke(train_items):
    model, tok, _ = load_checkpoint(str(BASE))
    model.cuda().train()
    batch = collate_items([[encode_item(tok, item, MAX_LEN, HEAD_MAX_LEN) for item in train_items[:8]]], tok.pad_token_id)
    logits = bf16_forward(model, batch, torch.device("cuda", 0), True, False)
    loss = soft_ce_loss(logits, batch["target"].cuda(), batch["marker_mask"].cuda())
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
    if not grads or any(not torch.isfinite(grad).all() for grad in grads):
        raise RuntimeError("smoke gradients invalid")
    result = {
        "loss": float(loss.detach()),
        "gradient_tensors": len(grads),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    del model
    torch.cuda.empty_cache()
    return result


def train_early(model, tok, train_items, typed_items, beam_items, beam_map, config, base_config):
    device = torch.device("cuda", 0)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.cuda().train()
    groups = [
        {"params": [parameter for name, parameter in model.named_parameters() if name.startswith("encoder.")],
         "lr": config.encoder_lr},
        {"params": [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.")],
         "lr": config.head_lr},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(train_items) / config.micro_batch)
    updates = math.ceil(steps_per_epoch / config.grad_accum) * config.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=config.min_lr)
    scaler = torch.amp.GradScaler("cuda")
    torch.manual_seed(config.seed)
    order_rng = random.Random(config.seed)
    params = [parameter for group in groups for parameter in group["params"]]
    history, best_key, best_epoch, stale = [], None, None, 0
    for epoch in range(config.epochs):
        epoch_items = list(train_items)
        random.Random(config.seed + epoch).shuffle(epoch_items)
        total, n_steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(epoch_items), config.micro_batch):
            chunk = [encode_item(tok, item, MAX_LEN, HEAD_MAX_LEN,
                                 draw_option_order(item, order_rng, config.shuffle_options), False)
                     for item in epoch_items[start:start + config.micro_batch]]
            batch = collate_items([chunk], tok.pad_token_id)
            logits = bf16_forward(model, batch, device, True, False)
            loss = soft_ce_loss(logits, batch["target"].to(device), batch["marker_mask"].to(device))
            window_start = (n_steps // config.grad_accum) * config.grad_accum
            window_size = min(config.grad_accum, steps_per_epoch - window_start)
            scaler.scale(loss / window_size).backward()
            n_steps += 1
            if n_steps % config.grad_accum == 0 or start + config.micro_batch >= len(epoch_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total += loss.item()
            if n_steps % 200 == 0:
                print(f"epoch {epoch + 1}/{config.epochs} step {n_steps} loss {loss.item():.4f}", flush=True)
        mean_loss = total / n_steps
        typed_records, typed_metrics, beam_records, beam_metrics = eval_sets(
            model, tok, typed_items, beam_items, beam_map)
        composite = (typed_metrics["soft_accuracy"] + beam_metrics["position_balanced_accuracy"]) / 2
        entry = {
            "epoch": epoch + 1,
            "train_mean_loss": mean_loss,
            "selection_composite": composite,
            "typed_validation": typed_metrics,
            "beam_validation": beam_metrics,
        }
        history.append(entry)
        atomic_json(OUT / "epoch_metrics.json", history)
        epoch_dir = OUT / f"checkpoint_epoch_{epoch + 1}"
        epoch_config = dict(base_config, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN)
        epoch_config.setdefault("training", {})["plain_modernbert_union_protected_test_v2"] = {
            "epoch": epoch + 1, "config": vars(config)}
        save_checkpoint(model, tok, epoch_config, str(epoch_dir))
        key = (composite, typed_metrics["soft_accuracy"], beam_metrics["position_balanced_accuracy"], -(epoch + 1))
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch + 1, 0
            save_checkpoint(model, tok, epoch_config, str(OUT / "selected_checkpoint"))
            atomic_json(OUT / "best_selection.json", {"epoch": best_epoch, "key": list(key), "metrics": entry})
        else:
            stale += 1
        print("EPOCH_VALIDATION " + json.dumps({
            "epoch": epoch + 1, "loss": mean_loss, "composite": composite,
            "typed_hard": typed_metrics["accuracy"], "typed_soft": typed_metrics["soft_accuracy"],
            "beam_raw": beam_metrics["accuracy"], "beam_balanced": beam_metrics["position_balanced_accuracy"],
            "best_epoch": best_epoch, "stale_epochs": stale,
        }, sort_keys=True), flush=True)
        model.train()
        if stale >= 1:
            print(f"EARLY_STOP epoch={epoch + 1} best_epoch={best_epoch}", flush=True)
            break
    return history, best_epoch


def records_from_json(rows):
    qtype_index = {name: index for index, name in enumerate(QTYPE_NAMES)}
    return [(qtype_index[row["qtype"]], row["logits"], row["target"], row["options"]) for row in rows]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "final_report.json").exists() or (OUT / "selected_checkpoint").exists():
        raise FileExistsError("protected-test union result artifacts already exist")
    if sha256_file(BASE / "model.safetensors") != EXPECTED_BASE_SHA:
        raise ValueError("fresh base hash mismatch")
    initialization = json.loads((BASE / "initialization_manifest.json").read_text())
    base_config = json.loads((BASE / "rl_agent_config.json").read_text())
    if initialization["modernbert_revision"] != "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13" or \
            base_config["initialization"]["root_laya_weights_loaded"] is not False:
        raise ValueError("fresh ModernBERT provenance invalid")

    manifest = json.loads((DATA / "split_manifest.json").read_text())
    if manifest["historical_final_test"]["evaluated_by_this_experiment"] or \
            manifest["relationship_grouping"]["cross_split_components"] != 0:
        raise ValueError("protected-test manifest invalid")
    expected_train = int(manifest["union"]["decisions"])
    expected_beam_val = int(manifest["beam"]["validation"]["rows"])

    model, tok, base_config = load_checkpoint(str(BASE))
    lt._forward = bf16_forward
    train_rows = read_jsonl(DATA / "train_union.jsonl")
    typed_rows = read_jsonl(TYPED_VAL)
    beam_rows = read_jsonl(DATA / "beam_validation_laya.jsonl")
    beam_map = read_audit_jsonl(DATA / "beam_validation_map.jsonl")
    train_items, skipped_train = items_from_rows(tok, train_rows, MAX_LEN, HEAD_MAX_LEN)
    typed_items, skipped_typed = items_from_rows(tok, typed_rows, MAX_LEN, HEAD_MAX_LEN)
    beam_items, skipped_beam = items_from_rows(tok, beam_rows, MAX_LEN, HEAD_MAX_LEN)
    if skipped_train or skipped_typed or skipped_beam or \
            (len(train_items), len(typed_items), len(beam_items)) != (expected_train, 1200, expected_beam_val):
        raise ValueError({"counts": [len(train_items), len(typed_items), len(beam_items)],
                          "skipped": [skipped_train, skipped_typed, skipped_beam]})
    hard = sum(max(item["target"]) >= 0.999999 for item in train_items)
    nondegenerate_soft = len(train_items) - hard
    beam_source_rows = sum(row["union_provenance"]["source"] ==
                           "api_beam_full_accepted_protected_test_split" for row in train_rows)
    typed_source_cases = sum(row["union_provenance"]["source"] ==
                             "LocalLLaMA/typed-decisions_fixed_train" for row in train_rows)
    if beam_source_rows != manifest["union"]["beam_hard_decisions"] or \
            typed_source_cases * 5 != manifest["union"]["typed_soft_decisions"]:
        raise ValueError({"beam_source_rows": beam_source_rows, "typed_source_cases": typed_source_cases})

    config = TrainConfig(
        epochs=3, micro_batch=8, grad_accum=8, encoder_lr=2.5e-5, head_lr=1e-4, min_lr=1e-6,
        weight_decay=0.01, grad_clip=1.0, loss="soft-ce", shuffle_options=(), option_layout="sequential",
        freeze_encoder=False, seed=0, max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN, amp=True,
        gradient_checkpointing=True, log_every=200)
    config.validate()
    run_config = {
        "name": "plain-modernbert-beam-typed-union-protected-test-v2-attempt2",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "initialization": {
            "checkpoint": str(BASE), "checkpoint_sha256": EXPECTED_BASE_SHA,
            "kind": "plain answerdotai/ModernBERT-large plus fresh seed-0 decision head",
            "encoder_revision": initialization["modernbert_revision"], "fresh_head_seed": initialization["seed"],
            "fresh_head_fingerprints": initialization["head_fingerprints_fp32"],
            "released_laya_weights_loaded": False,
            "beam_one_weights_loaded": False,
            "root_laya_checkpoint_sha256_for_nonidentity_reference": EXPECTED_ROOT_LAYA_SHA,
        },
        "split_manifest": str(DATA / "split_manifest.json"),
        "split_manifest_sha256": sha256_file(DATA / "split_manifest.json"),
        "training": {
            "items": len(train_items), "hard_target_items_by_shape": hard,
            "nondegenerate_soft_target_items_by_shape": nondegenerate_soft,
            "beam_hard_source_items": beam_source_rows,
            "typed_teacher_distribution_source_items": typed_source_cases * 5,
            "natural_proportion": True, "config": vars(config), "early_stopping_patience": 1,
        },
        "selection": "maximize equal-weight mean of Typed soft accuracy and Beam position-balanced accuracy; ties Typed soft, then Beam balanced, then earlier epoch",
        "beam_validation_protocol": "scenario-component-disjoint within seen task families",
        "historical_beam_final_test_evaluated": False,
        "official_typed_test_accessed": False,
        "dgx_data_accessed": False,
        "environment": {"host": platform.node(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
    }
    atomic_json(OUT / "run_config.json", run_config)
    atomic_json(OUT / "smoke_report.json", smoke(train_items))
    model.cuda()
    base_typed_records, base_typed_metrics, base_beam_records, base_beam_metrics = eval_sets(
        model, tok, typed_items, beam_items, beam_map)
    atomic_json(OUT / "base_metrics.json", {
        "typed_validation": base_typed_metrics, "beam_validation": base_beam_metrics})
    print("BASE_METRICS " + json.dumps({
        "typed_hard": base_typed_metrics["accuracy"], "typed_soft": base_typed_metrics["soft_accuracy"],
        "beam_raw": base_beam_metrics["accuracy"], "beam_balanced": base_beam_metrics["position_balanced_accuracy"],
    }, sort_keys=True), flush=True)

    history, best_epoch = train_early(
        model, tok, train_items, typed_items, beam_items, beam_map, config, base_config)
    del model
    torch.cuda.empty_cache()

    selected, selected_tok, _ = load_checkpoint(str(OUT / "selected_checkpoint"))
    selected.cuda()
    typed_records, typed_metrics, beam_records, beam_metrics = eval_sets(
        selected, selected_tok, typed_items, beam_items, beam_map)
    typed_path = OUT / "typed_validation_predictions.jsonl"
    beam_path = OUT / "beam_validation_predictions.jsonl"
    typed_path.write_text("".join(json.dumps(row) + "\n" for row in records_json(typed_records)))
    beam_path.write_text("".join(json.dumps(row) + "\n" for row in records_json(beam_records)))
    saved_typed = [json.loads(line) for line in typed_path.read_text().splitlines() if line]
    saved_beam = [json.loads(line) for line in beam_path.read_text().splitlines() if line]
    recomputed_typed = metrics(records_from_json(saved_typed))
    recomputed_beam = prediction_metrics(records_from_json(saved_beam), beam_map)
    verification = {
        "checkpoint_reload_verified": True,
        "typed_saved_prediction_rows": len(saved_typed),
        "beam_saved_prediction_rows": len(saved_beam),
        "typed_accuracy_recomputed": recomputed_typed["accuracy"],
        "typed_soft_accuracy_recomputed": recomputed_typed["soft_accuracy"],
        "beam_accuracy_recomputed": recomputed_beam["accuracy"],
        "beam_position_balanced_recomputed": recomputed_beam["position_balanced_accuracy"],
        "metrics_match": all(abs(left - right) < 1e-12 for left, right in [
            (typed_metrics["accuracy"], recomputed_typed["accuracy"]),
            (typed_metrics["soft_accuracy"], recomputed_typed["soft_accuracy"]),
            (beam_metrics["accuracy"], recomputed_beam["accuracy"]),
            (beam_metrics["position_balanced_accuracy"], recomputed_beam["position_balanced_accuracy"]),
        ]),
        "protected_final_test_evaluated": False,
        "cross_split_relationship_components": manifest["relationship_grouping"]["cross_split_components"],
    }
    atomic_json(OUT / "verification_report.json", verification)
    latency_report = latency(selected, selected_tok, typed_rows[0], typed_items[0])
    checkpoint_hash = sha256_file(OUT / "selected_checkpoint/model.safetensors")
    report = {
        "status": "completed", "best_epoch": best_epoch, "epochs_completed": len(history), "history": history,
        "base": {"typed_validation": base_typed_metrics, "beam_validation": base_beam_metrics},
        "selected": {"typed_validation": typed_metrics, "beam_validation": beam_metrics},
        "latency": latency_report,
        "selected_checkpoint": str(OUT / "selected_checkpoint"),
        "selected_checkpoint_sha256": checkpoint_hash,
        "verification": verification,
        "historical_beam_final_test_evaluated": False,
        "official_typed_test_accessed": False,
        "dgx_data_accessed": False,
    }
    atomic_json(OUT / "final_report.json", report)
    print("TRAINING_COMPLETE " + json.dumps({
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "typed_hard": typed_metrics["accuracy"], "typed_soft": typed_metrics["soft_accuracy"],
        "beam_raw": beam_metrics["accuracy"], "beam_balanced": beam_metrics["position_balanced_accuracy"],
        "checkpoint_sha256": checkpoint_hash, "metrics_match": verification["metrics_match"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
