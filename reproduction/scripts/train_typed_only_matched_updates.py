from __future__ import annotations

import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import laya.train as lt
from beam_one_common import prediction_metrics, read_jsonl as read_audit_jsonl
from laya.common import QTYPE_NAMES, collate_items
from laya.train import (
    TrainConfig,
    calibration_records,
    encode_item,
    items_from_rows,
    load_checkpoint,
    read_jsonl,
    save_checkpoint,
    soft_ce_loss,
)
from smoke_beam_one import bf16_forward
from train_laya_stage_compare import metrics, records_json


ROOT = Path(os.environ["SCOTTY_REPRO_ROOT"]).expanduser().resolve()
BASE = ROOT / "checkpoints/fresh-modernbert-45bb465"
DATA = ROOT / "analysis/typed-only-matched-updates-v1"
OUT = ROOT / "runs/typed-only-matched-updates-v1"
TYPED_TRAIN = ROOT / "analysis/beam-one-laya-stage-v1/train.jsonl"
TYPED_VAL = ROOT / "analysis/beam-one-laya-stage-v1/validation.jsonl"
BEAM_VAL = ROOT / "analysis/plain-modernbert-beam-typed-union-protected-test-v2/beam_validation_laya.jsonl"
BEAM_MAP = ROOT / "analysis/plain-modernbert-beam-typed-union-protected-test-v2/beam_validation_map.jsonl"
MAX_LEN, HEAD_MAX_LEN = 1024, 256
MILESTONES = (283, 566, 849)
SEGMENT_EXPOSURES = 18092
EXPECTED_BASE_SHA = "6f0fe404e2b83acd0fd7a949f2ceff0543cf56711d13f4d9c27f7c06d98b1002"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def eval_sets(model, tok, typed_items, beam_items, beam_map):
    device = torch.device("cuda", 0)
    typed_records = calibration_records(model, tok, typed_items, device, MAX_LEN, HEAD_MAX_LEN, parallel=False)
    beam_records = calibration_records(model, tok, beam_items, device, MAX_LEN, HEAD_MAX_LEN, parallel=False)
    return typed_records, metrics(typed_records), beam_records, prediction_metrics(beam_records, beam_map)


def records_from_json(rows):
    qtype_index = {name: index for index, name in enumerate(QTYPE_NAMES)}
    return [
        (qtype_index[row["qtype"]], row["logits"], row["target"], row["options"])
        for row in rows
    ]


class CyclingSource:
    def __init__(self, items, seed: int):
        self.items = items
        self.seed = seed
        self.cycle = 0
        self.queue = []
        self.source_passes_started = 0

    def _refill(self):
        order = list(range(len(self.items)))
        random.Random(self.seed + self.cycle).shuffle(order)
        self.cycle += 1
        self.source_passes_started += 1
        self.queue.extend(order)

    def take(self, count: int):
        selected = []
        while len(selected) < count:
            if not self.queue:
                self._refill()
            n = min(count - len(selected), len(self.queue))
            selected.extend(self.queue[:n])
            del self.queue[:n]
        return [self.items[index] for index in selected]


def smoke(train_items):
    model, tok, _ = load_checkpoint(str(BASE))
    model.cuda().train()
    batch = collate_items(
        [[encode_item(tok, item, MAX_LEN, HEAD_MAX_LEN) for item in train_items[:8]]],
        tok.pad_token_id,
    )
    logits = bf16_forward(model, batch, torch.device("cuda", 0), True, False)
    loss = soft_ce_loss(logits, batch["target"].cuda(), batch["marker_mask"].cuda())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads or any(not torch.isfinite(grad).all() for grad in grads):
        raise RuntimeError("smoke gradients invalid")
    result = {
        "loss": float(loss.detach()),
        "gradient_tensors": len(grads),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    del model
    torch.cuda.empty_cache()
    return result


def checkpoint_config(base_config, config, update, exposures):
    value = json.loads(json.dumps(base_config))
    value["max_len"] = MAX_LEN
    value["head_max_len"] = HEAD_MAX_LEN
    value.setdefault("training", {})["typed_only_matched_updates_v1"] = {
        "optimizer_update": update,
        "item_exposures": exposures,
        "config": vars(config),
    }
    return value


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "final_report.json").exists() or (OUT / "selected_checkpoint").exists():
        raise FileExistsError("typed-only matched-update artifacts already exist")
    protocol_path = DATA / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["state"] != "frozen_before_training" or protocol["training_or_scoring_started"]:
        raise ValueError("protocol is not a clean frozen protocol")
    if sha256(BASE / "model.safetensors") != EXPECTED_BASE_SHA:
        raise ValueError("fresh ModernBERT base hash mismatch")

    initialization = json.loads((BASE / "initialization_manifest.json").read_text(encoding="utf-8"))
    if initialization["modernbert_revision"] != "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13":
        raise ValueError("ModernBERT revision mismatch")
    base_config = json.loads((BASE / "rl_agent_config.json").read_text(encoding="utf-8"))
    if base_config["initialization"]["root_laya_weights_loaded"] is not False:
        raise ValueError("base unexpectedly contains released Laya weights")

    model0, tok, _ = load_checkpoint(str(BASE))
    train_rows = read_jsonl(str(TYPED_TRAIN))
    typed_val_rows = read_jsonl(str(TYPED_VAL))
    beam_val_rows = read_jsonl(str(BEAM_VAL))
    beam_map = read_audit_jsonl(BEAM_MAP)
    train_items, skipped_train = items_from_rows(tok, train_rows, MAX_LEN, HEAD_MAX_LEN)
    typed_items, skipped_typed = items_from_rows(tok, typed_val_rows, MAX_LEN, HEAD_MAX_LEN)
    beam_items, skipped_beam = items_from_rows(tok, beam_val_rows, MAX_LEN, HEAD_MAX_LEN)
    del model0
    if skipped_train or skipped_typed or skipped_beam:
        raise ValueError({"skipped": [skipped_train, skipped_typed, skipped_beam]})
    if (len(train_items), len(typed_items), len(beam_items), len(beam_map)) != (4800, 1200, 3327, 3327):
        raise ValueError("converted item counts differ from frozen protocol")

    smoke_report = smoke(train_items)
    atomic_json(OUT / "smoke_report.json", smoke_report)

    config = TrainConfig(
        epochs=3,
        micro_batch=8,
        grad_accum=8,
        encoder_lr=2.5e-5,
        head_lr=1e-4,
        min_lr=1e-6,
        weight_decay=0.01,
        grad_clip=1.0,
        loss="soft-ce",
        shuffle_options=(),
        option_layout="sequential",
        freeze_encoder=False,
        seed=0,
        max_len=MAX_LEN,
        head_max_len=HEAD_MAX_LEN,
        amp=True,
        gradient_checkpointing=True,
        log_every=50,
    )
    config.validate()

    run_config = {
        "name": "typed-only-matched-updates-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "initialization": protocol["initialization"],
        "training": protocol["training"],
        "selection": protocol["selection"],
        "training_items": 4800,
        "typed_validation_items": 1200,
        "beam_validation_items": 3327,
        "beam_training_rows": 0,
        "official_tests_accessed_during_training": False,
        "dgx_data_accessed": False,
        "smoke": smoke_report,
    }
    atomic_json(OUT / "run_config.json", run_config)

    model, tok, _ = load_checkpoint(str(BASE))
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.cuda().train()
    groups = [
        {"params": [p for name, p in model.named_parameters() if name.startswith("encoder.")], "lr": config.encoder_lr},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("encoder.")], "lr": config.head_lr},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=849, eta_min=config.min_lr)
    scaler = torch.amp.GradScaler("cuda")
    torch.manual_seed(config.seed)
    source = CyclingSource(train_items, config.seed)
    params = [p for group in groups for p in group["params"]]

    history = []
    best_key = None
    best_update = None
    update = 0
    exposures = 0
    microsteps = 0
    for segment_index, milestone in enumerate(MILESTONES, start=1):
        segment_exposures = 0
        segment_losses = []
        for local_update in range(1, 284):
            batch_size = 44 if local_update == 283 else 64
            selected = source.take(batch_size)
            microbatches = [selected[start : start + config.micro_batch] for start in range(0, batch_size, config.micro_batch)]
            optimizer.zero_grad(set_to_none=True)
            for chunk in microbatches:
                encoded = [encode_item(tok, item, MAX_LEN, HEAD_MAX_LEN, None, False) for item in chunk]
                batch = collate_items([encoded], tok.pad_token_id)
                logits = bf16_forward(model, batch, torch.device("cuda", 0), True, False)
                loss = soft_ce_loss(logits, batch["target"].cuda(), batch["marker_mask"].cuda())
                scaler.scale(loss / len(microbatches)).backward()
                segment_losses.append(float(loss.detach()))
                microsteps += 1
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, config.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update += 1
            exposures += batch_size
            segment_exposures += batch_size
            if update % 50 == 0 or update in MILESTONES:
                print(
                    "TRAIN_UPDATE "
                    + json.dumps(
                        {
                            "update": update,
                            "milestone": milestone,
                            "item_exposures": exposures,
                            "latest_microbatch_loss": segment_losses[-1],
                            "grad_norm_before_clip": grad_norm,
                            "encoder_lr": optimizer.param_groups[0]["lr"],
                            "head_lr": optimizer.param_groups[1]["lr"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if update != milestone or segment_exposures != SEGMENT_EXPOSURES:
            raise RuntimeError("update/exposure accounting mismatch")

        typed_records, typed_metrics, beam_records, beam_metrics = eval_sets(
            model, tok, typed_items, beam_items, beam_map
        )
        composite = (typed_metrics["soft_accuracy"] + beam_metrics["position_balanced_accuracy"]) / 2
        entry = {
            "segment": segment_index,
            "optimizer_update": update,
            "item_exposures": exposures,
            "equivalent_source_passes": exposures / len(train_items),
            "microsteps": microsteps,
            "train_mean_microbatch_loss_since_previous_validation": float(np.mean(segment_losses)),
            "selection_composite": composite,
            "typed_validation": typed_metrics,
            "beam_validation": beam_metrics,
        }
        history.append(entry)
        atomic_json(OUT / "milestone_metrics.json", history)
        cfg = checkpoint_config(base_config, config, update, exposures)
        checkpoint_dir = OUT / f"checkpoint_update_{update}"
        save_checkpoint(model, tok, cfg, str(checkpoint_dir))
        key = (composite, typed_metrics["soft_accuracy"], beam_metrics["position_balanced_accuracy"], -update)
        if best_key is None or key > best_key:
            best_key, best_update = key, update
            save_checkpoint(model, tok, cfg, str(OUT / "selected_checkpoint"))
            atomic_json(OUT / "best_selection.json", {"optimizer_update": update, "key": list(key), "metrics": entry})
        print(
            "MILESTONE_VALIDATION "
            + json.dumps(
                {
                    "update": update,
                    "exposures": exposures,
                    "composite": composite,
                    "typed_hard": typed_metrics["accuracy"],
                    "typed_soft": typed_metrics["soft_accuracy"],
                    "beam_raw": beam_metrics["accuracy"],
                    "beam_balanced": beam_metrics["position_balanced_accuracy"],
                    "best_update": best_update,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        model.train()

    if update != 849 or exposures != 54276:
        raise RuntimeError("final update/exposure count mismatch")
    del model
    torch.cuda.empty_cache()

    reloaded, reloaded_tok, _ = load_checkpoint(str(OUT / "selected_checkpoint"))
    reloaded.cuda().eval()
    typed_records, typed_metrics, beam_records, beam_metrics = eval_sets(
        reloaded, reloaded_tok, typed_items, beam_items, beam_map
    )
    typed_json = records_json(typed_records)
    beam_json = records_json(beam_records)
    atomic_json(OUT / "typed_validation_predictions.json", typed_json)
    atomic_json(OUT / "beam_validation_predictions.json", beam_json)
    typed_recomputed = metrics(records_from_json(json.loads((OUT / "typed_validation_predictions.json").read_text())))
    beam_recomputed = prediction_metrics(
        records_from_json(json.loads((OUT / "beam_validation_predictions.json").read_text())), beam_map
    )
    selected_sha = sha256(OUT / "selected_checkpoint/model.safetensors")
    best_milestone_sha = sha256(OUT / f"checkpoint_update_{best_update}/model.safetensors")
    report = {
        "status": "training_and_validation_completed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer_updates": update,
        "item_exposures": exposures,
        "equivalent_source_passes": exposures / len(train_items),
        "source_cycles_started": source.source_passes_started,
        "best_update": best_update,
        "selected_checkpoint": str(OUT / "selected_checkpoint"),
        "selected_checkpoint_sha256": selected_sha,
        "selected_matches_best_milestone": selected_sha == best_milestone_sha,
        "reload_verified": True,
        "typed_validation": typed_metrics,
        "beam_validation": beam_metrics,
        "independent_saved_prediction_recompute": {
            "typed_validation": typed_recomputed,
            "beam_validation": beam_recomputed,
        },
        "tests_accessed": False,
        "history": history,
    }
    atomic_json(OUT / "final_report.json", report)
    print("TRAINING_COMPLETE " + json.dumps({
        "best_update": best_update,
        "checkpoint_sha256": selected_sha,
        "typed_hard": typed_metrics["accuracy"],
        "typed_soft": typed_metrics["soft_accuracy"],
        "beam_raw": beam_metrics["accuracy"],
        "beam_balanced": beam_metrics["position_balanced_accuracy"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
