from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import laya.train as lt
from beam_one_common import prediction_metrics, read_jsonl as read_audit_jsonl, sha256_file
from laya.common import QTYPE_NAMES, collate_items, encode_text, serialize_state
from laya.train import (TrainConfig, calibration_records, encode_item, items_from_rows,
                        load_checkpoint, read_jsonl, save_checkpoint, soft_ce_loss, train_model)
from smoke_beam_one import bf16_forward


ROOT = Path(os.environ["SCOTTY_REPRO_ROOT"]).expanduser().resolve()
BASE = ROOT / "runs/beam-one-family-disjoint-v1/model/selected_checkpoint"
SPLIT = ROOT / "analysis/beam-one-laya-stage-v1"
OUT = ROOT / "runs/beam-one-laya-stage-v1"
BEAM_VAL = ROOT / "analysis/beam-one-family-selected-v1/validation_laya.jsonl"
BEAM_MAP = ROOT / "analysis/beam-one-family-selected-v1/validation_map.jsonl"
EXPECTED_BASE_SHA = "0e87a32d617ca8b2fc199c0321c62be4565782df7200e379803fd8929606e9aa"
TRAINING_CUTOFF = datetime(2026, 10, 24, 0, 4, tzinfo=timezone.utc).timestamp()
HARD_DEADLINE = datetime(2026, 10, 24, 0, 14, tzinfo=timezone.utc).timestamp()
MAX_LEN, HEAD_MAX_LEN = 1024, 256


def atomic_json(path: Path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def metrics(records):
    by = {}
    all_rows = []
    for qt, logits, target, k in records:
        z = np.asarray(logits[:k], dtype=np.float64)
        target = np.asarray(target[:k], dtype=np.float64)
        p = np.exp(z - z.max()); p /= p.sum()
        pred, gold = int(p.argmax()), int(target.argmax())
        row = {"correct": int(pred == gold), "soft_accuracy": float(target[pred]),
               "cross_entropy": float(-(target * np.log(np.clip(p, 1e-12, 1))).sum()),
               "brier": float(((p - target) ** 2).mean()), "confidence": float(p[pred]),
               "ordinal_abs_error": abs(pred - gold) if QTYPE_NAMES[int(qt)] == "score" else None}
        all_rows.append(row); by.setdefault(QTYPE_NAMES[int(qt)], []).append(row)
    def agg(rows):
        result = {"items": len(rows), "accuracy": sum(x["correct"] for x in rows)/len(rows),
                  "soft_accuracy": sum(x["soft_accuracy"] for x in rows)/len(rows),
                  "cross_entropy": sum(x["cross_entropy"] for x in rows)/len(rows),
                  "brier": sum(x["brier"] for x in rows)/len(rows),
                  "mean_confidence": sum(x["confidence"] for x in rows)/len(rows)}
        ords = [x["ordinal_abs_error"] for x in rows if x["ordinal_abs_error"] is not None]
        if ords: result["ordinal_mae"] = sum(ords)/len(ords)
        return result
    return {**agg(all_rows), "by_type": {k: agg(v) for k,v in sorted(by.items())}}


def records_json(records):
    return [{"qtype": QTYPE_NAMES[int(qt)], "logits": np.asarray(logits[:k]).tolist(),
             "target": list(target[:k]), "options": int(k)} for qt, logits, target, k in records]


@torch.no_grad()
def encode_docs(model, tok, texts, batch=24, max_length=256):
    model.encoder.eval(); out=[]
    for start in range(0, len(texts), batch):
        enc=tok(texts[start:start+batch], padding=True, truncation=True, max_length=max_length,
                return_tensors="pt")
        ids=enc["input_ids"].cuda(); mask=enc["attention_mask"].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden=model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        w=mask.unsqueeze(-1); pooled=(hidden*w).sum(1)/w.sum(1).clamp_min(1)
        out.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu())
    return torch.cat(out)


def semantic_audit(model, tok, typed_rows):
    beam_rows = read_audit_jsonl(ROOT / "analysis/beam-one-family-disjoint-v1/validation_source.jsonl") + \
                read_audit_jsonl(ROOT / "analysis/beam-one-family-disjoint-v1/test_source.jsonl")
    typed_texts=[serialize_state(r["state"]) for r in typed_rows]
    beam_texts=[r["situation"]+"\n"+r["instruction"]+"\n"+"\n".join(c["description"] for c in r["candidates"])
                for r in beam_rows]
    t=encode_docs(model,tok,typed_texts); b=encode_docs(model,tok,beam_texts)
    maxima=[]
    for start in range(0,len(t),100):
        vals,idx=(t[start:start+100].cuda() @ b.T.cuda()).max(dim=1)
        maxima.extend(zip(vals.cpu().tolist(),idx.cpu().tolist(),range(start,min(start+100,len(t)))))
    maxima.sort(reverse=True)
    report={"method":"Beam One encoder mean-pooled cosine, max_len=256",
            "threshold_counts":{str(x):sum(v>=x for v,_,_ in maxima) for x in (0.9,0.95,0.98,0.99)},
            "top20":[{"cosine":v,"typed_state_sha256":hashlib.sha256(serialize_state(typed_rows[ti]["state"]).encode()).hexdigest(),
                      "beam_id":beam_rows[bi]["id"],"beam_task_family":beam_rows[bi]["task_family"]}
                     for v,bi,ti in maxima[:20]]}
    atomic_json(OUT/"semantic_overlap_audit.json",report)
    print("SEMANTIC_AUDIT "+json.dumps(report["threshold_counts"],sort_keys=True),flush=True)


def configure(model, arm):
    for p in model.parameters(): p.requires_grad_(True)
    if arm=="head_only":
        for p in model.encoder.parameters(): p.requires_grad_(False)
    elif arm=="top2":
        for p in model.encoder.parameters(): p.requires_grad_(False)
        for p in model.encoder.layers[-2:].parameters(): p.requires_grad_(True)
        for block in model.encoder.layers[:-2]:
            for mod in block.modules():
                if isinstance(mod,torch.nn.Dropout): mod.p=0.0
    elif arm!="full": raise ValueError(arm)


def config_for(arm):
    hp={"head_only":(0.0,1e-4,True,False),"top2":(5e-6,5e-5,False,True),
        "full":(2.5e-6,2.5e-5,False,True)}[arm]
    c=TrainConfig(epochs=1,micro_batch=8,grad_accum=8,encoder_lr=hp[0],head_lr=hp[1],min_lr=1e-6,
                  weight_decay=.01,grad_clip=1.0,loss="soft-ce",shuffle_options=(),option_layout="sequential",
                  freeze_encoder=hp[2],seed=0,max_len=MAX_LEN,head_max_len=HEAD_MAX_LEN,amp=True,
                  gradient_checkpointing=hp[3],log_every=100)
    c.validate(); return c


def smoke(arm, items):
    model,tok,_=load_checkpoint(str(BASE)); configure(model,arm); model.cuda().train()
    if arm=="head_only": model.encoder.eval()
    chunk=[encode_item(tok,x,MAX_LEN,HEAD_MAX_LEN) for x in items[:8]]
    batch=collate_items([chunk],tok.pad_token_id)
    logits=bf16_forward(model,batch,torch.device("cuda",0),True,arm=="head_only")
    loss=soft_ce_loss(logits,batch["target"].cuda(),batch["marker_mask"].cuda()); loss.backward()
    train_grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    frozen_grads=[p.grad for p in model.parameters() if not p.requires_grad and p.grad is not None]
    if not train_grads or frozen_grads or any(not torch.isfinite(g).all() for g in train_grads):
        raise RuntimeError(f"gradient smoke failed {arm}")
    result={"arm":arm,"loss":float(loss),"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_parameters":sum(p.numel() for p in model.parameters() if not p.requires_grad),
            "trainable_grad_tensors":len(train_grads),"frozen_grad_tensors":len(frozen_grads)}
    del model; torch.cuda.empty_cache(); print("SMOKE "+json.dumps(result,sort_keys=True),flush=True); return result


@torch.no_grad()
def latency(model,tok,row,item):
    device=torch.device("cuda",0); encoded=encode_item(tok,item,MAX_LEN,HEAD_MAX_LEN)
    batch=collate_items([[encoded]],tok.pad_token_id)
    for _ in range(5): bf16_forward(model,batch,device,True,False)
    torch.cuda.synchronize(); values=[]
    for _ in range(20):
        t0=time.perf_counter(); bf16_forward(model,batch,device,True,False); torch.cuda.synchronize()
        values.append((time.perf_counter()-t0)*1000)
    e2e=[]
    for _ in range(10):
        t0=time.perf_counter(); its,_=items_from_rows(tok,[row],MAX_LEN,HEAD_MAX_LEN)
        one=collate_items([[encode_item(tok,its[0],MAX_LEN,HEAD_MAX_LEN)]],tok.pad_token_id)
        bf16_forward(model,one,device,True,False); torch.cuda.synchronize(); e2e.append((time.perf_counter()-t0)*1000)
    return {"model_only_batch1_ms":{"median":float(np.median(values)),"p95":float(np.percentile(values,95))},
            "tokenize_case_plus_first_decision_ms":{"median":float(np.median(e2e)),"p95":float(np.percentile(e2e,95))},
            "samples":{"model_only":20,"end_to_end":10}}


def eval_both(model,tok,typed_items,beam_items,beam_map):
    tr=calibration_records(model,tok,typed_items,torch.device("cuda",0),MAX_LEN,HEAD_MAX_LEN,parallel=False)
    br=calibration_records(model,tok,beam_items,torch.device("cuda",0),MAX_LEN,HEAD_MAX_LEN,parallel=False)
    return tr,metrics(tr),br,prediction_metrics(br,beam_map)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"comparison_report.json").exists() or any((OUT/a).exists() for a in ("head_only","top2","full")):
        raise FileExistsError("beam-one-laya-stage-v1 result artifacts already exist")
    if sha256_file(BASE/"model.safetensors")!=EXPECTED_BASE_SHA: raise ValueError("base hash mismatch")
    train_rows=read_jsonl(SPLIT/"train.jsonl"); val_rows=read_jsonl(SPLIT/"validation.jsonl")
    beam_rows=read_jsonl(BEAM_VAL); beam_map=read_audit_jsonl(BEAM_MAP)
    base,tok,cfg=load_checkpoint(str(BASE)); lt._forward=bf16_forward
    train_items,sk1=items_from_rows(tok,train_rows,MAX_LEN,HEAD_MAX_LEN)
    val_items,sk2=items_from_rows(tok,val_rows,MAX_LEN,HEAD_MAX_LEN)
    beam_items,sk3=items_from_rows(tok,beam_rows,MAX_LEN,HEAD_MAX_LEN)
    if sk1 or sk2 or sk3 or (len(train_items),len(val_items),len(beam_items))!=(4800,1200,3166):
        raise ValueError({"counts":[len(train_items),len(val_items),len(beam_items)],"skipped":[sk1,sk2,sk3]})
    soft=sum(1 for x in train_items if max(x["target"])<.999999)
    if soft<4700 or any(abs(sum(x["target"])-1)>1e-5 for x in train_items): raise ValueError("soft targets invalid")
    run_config={"name":"beam-one-laya-stage-v1","started_utc":datetime.now(timezone.utc).isoformat(),
      "base":str(BASE),"base_sha256":EXPECTED_BASE_SHA,"dataset":"LocalLLaMA/typed-decisions",
      "dataset_revision":"468b1461d59e01d6404e1cf37431da962acd43a0","split_manifest":str(SPLIT/"split_manifest.json"),
      "train_cases":960,"train_decisions":4800,"validation_cases":240,"validation_decisions":1200,
      "official_typed_test_accessed":False,"beam_final_test_accessed":False,"soft_target_items":soft,
      "shared_budget":{"epochs":1,"micro_batch":8,"grad_accum":8,"effective_batch":64,"seed":0,
                       "loss":"soft-ce","option_shuffle":False,"selection":"typed validation accuracy then soft accuracy"},
      "arms":{"head_only":{"encoder":"frozen","head_lr":1e-4},
              "top2":{"encoder":"only layers 26-27 trainable","encoder_lr":5e-6,"head_lr":5e-5},
              "full":{"encoder":"all 28 layers trainable","encoder_lr":2.5e-6,"head_lr":2.5e-5}},
      "training_cutoff_utc":"2026-10-24T00:04:00Z","hard_deadline_utc":"2026-10-24T00:14:00Z",
      "environment":{"host":platform.node(),"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0)}}
    atomic_json(OUT/"run_config.json",run_config)
    base.cuda(); semantic_audit(base,tok,train_rows+val_rows)
    btr,btm,bbr,bbm=eval_both(base,tok,val_items,beam_items,beam_map)
    atomic_json(OUT/"base_metrics.json",{"typed_validation":btm,"beam_validation":bbm})
    print("BASE_METRICS "+json.dumps({"typed_accuracy":btm["accuracy"],"typed_soft_accuracy":btm["soft_accuracy"],
          "beam_accuracy":bbm["accuracy"],"beam_position_balanced":bbm["position_balanced_accuracy"]},sort_keys=True),flush=True)
    del base; torch.cuda.empty_cache()

    smokes={a:smoke(a,train_items) for a in ("head_only","top2","full")}
    atomic_json(OUT/"smoke_report.json",smokes)
    results={}
    for arm in ("head_only","top2","full"):
        if time.time()>=TRAINING_CUTOFF:
            results[arm]={"status":"skipped_training_cutoff"}; print("ARM_SKIPPED "+arm,flush=True); continue
        arm_dir=OUT/arm; arm_dir.mkdir()
        model,atok,acfg=load_checkpoint(str(BASE)); configure(model,arm); config=config_for(arm)
        start=time.time(); print("ARM_START "+json.dumps({"arm":arm,"utc":datetime.now(timezone.utc).isoformat(),
              "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
              "config":vars(config)},sort_keys=True),flush=True)
        model.cuda()
        epoch_metric={}
        def end(epoch,loss):
            nonlocal epoch_metric
            rec=calibration_records(model,atok,val_items,torch.device("cuda",0),MAX_LEN,HEAD_MAX_LEN,parallel=False)
            epoch_metric={"epoch":epoch+1,"train_loss":loss,"typed_validation":metrics(rec)}
            atomic_json(arm_dir/"epoch_metrics.json",[epoch_metric])
            print("ARM_EPOCH "+json.dumps({"arm":arm,"loss":loss,"typed_accuracy":epoch_metric["typed_validation"]["accuracy"],
                  "typed_soft_accuracy":epoch_metric["typed_validation"]["soft_accuracy"]},sort_keys=True),flush=True)
            model.train()
        losses=train_model(model,atok,train_items,config,torch.device("cuda",0),MAX_LEN,HEAD_MAX_LEN,
                           on_epoch_end=end,parallel=False)
        final_cfg=dict(acfg,max_len=MAX_LEN,head_max_len=HEAD_MAX_LEN,fine_tuned=True)
        final_cfg.setdefault("training",{})["beam_one_laya_stage_v1"]={"arm":arm,"config":vars(config),
            "base_sha256":EXPECTED_BASE_SHA,"split_manifest":str(SPLIT/"split_manifest.json")}
        ckpt=arm_dir/"selected_checkpoint"; save_checkpoint(model,atok,final_cfg,str(ckpt))
        del model; torch.cuda.empty_cache()
        reloaded,rtok,_=load_checkpoint(str(ckpt)); reloaded.cuda()
        tr,tm,br,bm=eval_both(reloaded,rtok,val_items,beam_items,beam_map)
        lat=latency(reloaded,rtok,val_rows[0],val_items[0])
        (arm_dir/"typed_validation_predictions.jsonl").write_text("".join(json.dumps(x)+"\n" for x in records_json(tr)))
        result={"status":"completed","elapsed_seconds":time.time()-start,"train_losses":losses,
                "typed_validation":tm,"beam_validation":bm,"latency":lat,
                "checkpoint":str(ckpt),"checkpoint_sha256":sha256_file(ckpt/"model.safetensors"),
                "reload_verified":True,"official_typed_test_accessed":False,"beam_final_test_accessed":False}
        atomic_json(arm_dir/"report.json",result); results[arm]=result
        print("ARM_COMPLETE "+json.dumps({"arm":arm,"elapsed_seconds":result["elapsed_seconds"],
              "typed_accuracy":tm["accuracy"],"typed_soft_accuracy":tm["soft_accuracy"],
              "beam_accuracy":bm["accuracy"],"beam_position_balanced":bm["position_balanced_accuracy"],
              "checkpoint_sha256":result["checkpoint_sha256"]},sort_keys=True),flush=True)
        del reloaded; torch.cuda.empty_cache()
    report={"status":"completed" if all(x.get("status")=="completed" for x in results.values()) else "partial",
            "completed_utc":datetime.now(timezone.utc).isoformat(),"base":{"typed_validation":btm,"beam_validation":bbm},
            "arms":results,"hard_deadline_respected":time.time()<=HARD_DEADLINE,
            "official_typed_test_accessed":False,"beam_final_test_accessed":False}
    atomic_json(OUT/"comparison_report.json",report)
    print("COMPARISON_COMPLETE "+json.dumps({"status":report["status"],"hard_deadline_respected":report["hard_deadline_respected"],
           "arms":{a:{k:v.get(k) for k in ("status","elapsed_seconds")} for a,v in results.items()}},sort_keys=True),flush=True)


if __name__=="__main__": main()
