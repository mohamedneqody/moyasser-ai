"""
Trains the PyTorch Bidirectional GRU Sequence Slot Tagger (`ArabicSlotBiGRU`)
for Explainable Egyptian Dialect Financial Token Grounding (`models/arabic_slot_bigru.pt`).
Handles BOTH forward (`مبيعاتي 4 مليون`) AND reversed (`بعمل 4 مليون مبيعات`) Egyptian word orders,
monthly/daily temporal multipliers (`في الشهر`, `في اليوم`), and explicit negations (`معليش غرامات`).
"""

import json
import os
import random
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from moyasser.arabic_dialect_xray import SLOT_TAGS, ArabicSlotBiGRU, _norm_token

MODEL_DIR = os.path.join(ROOT_DIR, "models")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def build_tagged_egyptian_sentences(n_samples: int = 2400) -> List[List[Tuple[str, str]]]:
    rev_nouns = ["مبيعاتي", "مبيعاته", "ايراداتي", "ايرادات", "دخلي", "حجم اعمالي", "شغلي", "ببيع ب"]
    rev_post = ["مبيعات", "ايرادات", "دخل", "حجم اعمال"]
    exp_nouns = ["مصاريفي", "مصاريفه", "مصروفاتي", "تكاليفي", "ايجاري وبضاعتي", "نفقاتي"]
    exp_post = ["مصاريف", "مصروفات", "تكاليف", "ايجار وخامات"]
    pen_nouns = ["غرامات", "غرامات تاخير", "مقابل تاخير", "فوائد تاخير"]
    princ_nouns = ["اصل الضريبه", "اصل الدين", "نزاع بقيمه"]
    re_clauses = [
        ([("بعت", "RE_VAL"), ("شقه", "RE_VAL"), ("ب", "O"), ("<NUM>", "RE_VAL"), ("مليون", "RE_VAL"), ("لابني", "RE_VAL")]),
        ([("بيع", "RE_VAL"), ("عقار", "RE_VAL"), ("بقيمه", "O"), ("<NUM>", "RE_VAL"), ("مليون", "RE_VAL")]),
        ([("<NUM>", "RE_VAL"), ("مليون", "RE_VAL"), ("سعر", "O"), ("بيع", "RE_VAL"), ("فيلا", "RE_VAL")]),
    ]
    vat_clauses = [
        ([("رد", "VAT"), ("قيمه", "VAT"), ("مضافه", "VAT"), ("<NUM>", "VAT"), ("الف", "VAT")]),
        ([("<NUM>", "VAT"), ("الف", "VAT"), ("رصيد", "VAT"), ("دائن", "VAT"), ("مضافه", "VAT")]),
    ]
    neg_clauses = [
        ([("ومعليش", "NEG"), ("اي", "O"), ("غرامات", "PEN")]),
        ([("ومفيش", "NEG"), ("عليا", "O"), ("غرامات", "PEN"), ("قديمه", "O")]),
        ([("بدون", "NEG"), ("نزاع", "PRINC"), ("او", "O"), ("غرامه", "PEN")]),
    ]
    freqs = [
        [("في", "FREQ"), ("الشهر", "FREQ")],
        [("شهريا", "FREQ")],
        [("في", "FREQ"), ("اليوم", "FREQ")],
        [("في", "O"), ("السنه", "O")],
        [],
    ]
    intros = [
        [("عندي", "O"), ("مطعم", "O"), ("في", "O"), ("الجيزه", "O")],
        [("انا", "O"), ("فريلانسر", "O"), ("في", "O"), ("القاهره", "O")],
        [("شركه", "O"), ("برمجيات", "O"), ("مصريه", "O")],
        [("محل", "O"), ("ملابس", "O"), ("في", "O"), ("اسكندريه", "O")],
        [],
    ]
    nums = ["100", "200", "500", "1500000", "4", "12", "60", "180", "2500000"]
    units = ["الف", "مليون", "جنيه", ""]

    def tag_phrase(phrase: str, tag: str) -> List[Tuple[str, str]]:
        return [(w, tag) for w in phrase.split()]

    samples: List[List[Tuple[str, str]]] = []
    for _ in range(n_samples):
        sent: List[Tuple[str, str]] = list(random.choice(intros))
        mode = random.random()

        if mode < 0.72:
            # Forward OR Reversed Revenue clause!
            n_val = random.choice(nums)
            u_val = random.choice(units)
            fr = random.choice(freqs)
            if random.random() < 0.5:
                # Forward: `مبيعاتي 4 مليون في الشهر`
                sent.extend(tag_phrase(random.choice(rev_nouns), "REV"))
                sent.append((n_val, "REV"))
                if u_val:
                    sent.append((u_val, "REV"))
                sent.extend(fr)
            else:
                # Reversed Egyptian order: `بعمل 4 مليون مبيعات في الشهر`
                sent.append(("بعمل", "O"))
                sent.append((n_val, "REV"))
                if u_val:
                    sent.append((u_val, "REV"))
                sent.extend(tag_phrase(random.choice(rev_post), "REV"))
                sent.extend(fr)

            # Add Expenses clause (forward or reversed)
            if random.random() < 0.75:
                sent.append(("و", "O"))
                n2, u2 = random.choice(nums), random.choice(units)
                if random.random() < 0.55:
                    sent.extend(tag_phrase(random.choice(exp_nouns), "EXP"))
                    sent.append((n2, "EXP"))
                    if u2:
                        sent.append((u2, "EXP"))
                else:
                    sent.append(("بدفع", "O"))
                    sent.append((n2, "EXP"))
                    if u2:
                        sent.append((u2, "EXP"))
                    sent.extend(tag_phrase(random.choice(exp_post), "EXP"))

            # Add Penalty/Principal OR Negation OR VAT
            r_tail = random.random()
            if r_tail < 0.35:
                sent.append(("وعليا", "O"))
                sent.extend(tag_phrase(random.choice(pen_nouns), "PEN"))
                sent.append((random.choice(nums), "PEN"))
                sent.append(("الف", "PEN"))
                if random.random() < 0.6:
                    sent.append(("و", "O"))
                    sent.extend(tag_phrase(random.choice(princ_nouns), "PRINC"))
                    sent.append((random.choice(nums), "PRINC"))
                    sent.append(("الف", "PRINC"))
            elif r_tail < 0.65:
                sent.extend(random.choice(neg_clauses))
            elif r_tail < 0.85:
                sent.append(("وليا", "O"))
                sc = [(random.choice(nums) if w == "<NUM>" else w, t) for w, t in random.choice(vat_clauses)]
                sent.extend(sc)
        else:
            # Real estate focused or mixed
            sc = [(random.choice(nums) if w == "<NUM>" else w, t) for w, t in random.choice(re_clauses)]
            sent.extend(sc)
            if random.random() < 0.4:
                sent.extend(random.choice(neg_clauses))

        samples.append(sent)
    return samples


def main() -> None:
    print("[1/2] Generating 2,400 token-tagged Egyptian dialect sentences (forward & reversed word orders)...")
    dataset = build_tagged_egyptian_sentences(2400)
    train_sents, test_sents = train_test_split(dataset, test_size=0.20, random_state=SEED)

    # Build vocabulary strictly from training split!
    vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<NUM>": 2}
    tag2idx = {t: i for i, t in enumerate(SLOT_TAGS)}

    for sent in train_sents:
        for w, _ in sent:
            nw = _norm_token(w)
            if re.fullmatch(r"\d+(?:[.,]\d+)?", nw):
                nw = "<NUM>"
            if nw and nw not in vocab:
                vocab[nw] = len(vocab)

    max_len = 36

    def encode_batch(sents: List[List[Tuple[str, str]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X = np.zeros((len(sents), max_len), dtype=np.int64)
        Y = np.full((len(sents), max_len), -100, dtype=np.int64)
        M = np.zeros((len(sents), max_len), dtype=bool)
        for i, sent in enumerate(sents):
            for j, (w, tag) in enumerate(sent[:max_len]):
                nw = _norm_token(w)
                if re.fullmatch(r"\d+(?:[.,]\d+)?", nw):
                    nw = "<NUM>"
                X[i, j] = vocab.get(nw, 1)
                Y[i, j] = tag2idx[tag]
                M[i, j] = True
        return torch.tensor(X), torch.tensor(Y), torch.tensor(M)

    X_tr, Y_tr, _ = encode_batch(train_sents)
    X_te, Y_te, M_te = encode_batch(test_sents)

    print(f"[2/2] Training PyTorch ArabicSlotBiGRU (vocab={len(vocab)}, tags={len(SLOT_TAGS)})...")
    t0 = time.perf_counter()
    model = ArabicSlotBiGRU(vocab_size=len(vocab), emb_dim=64, hidden_dim=96, num_tags=len(SLOT_TAGS))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss(ignore_index=-100)

    bs = 64
    for epoch in range(18):
        model.train()
        perm = torch.randperm(len(X_tr))
        for start in range(0, len(X_tr), bs):
            idx = perm[start : start + bs]
            opt.zero_grad()
            logits = model(X_tr[idx])
            loss = crit(logits.view(-1, len(SLOT_TAGS)), Y_tr[idx].view(-1))
            loss.backward()
            opt.step()
    train_sec = round(time.perf_counter() - t0, 2)

    model.eval()
    with torch.no_grad():
        preds = model(X_te).argmax(dim=-1).numpy()
    y_true_flat = Y_te.numpy()[M_te.numpy()]
    y_pred_flat = preds[M_te.numpy()]
    tok_acc = round(float(accuracy_score(y_true_flat, y_pred_flat)), 4)
    tok_f1 = round(float(f1_score(y_true_flat, y_pred_flat, average="macro")), 4)

    out_pt = os.path.join(MODEL_DIR, "arabic_slot_bigru.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "vocab_size": len(vocab),
            "emb_dim": 64,
            "hidden_dim": 96,
            "tags": SLOT_TAGS,
            "torch_version": str(torch.__version__),
        },
        out_pt,
    )
    print(f"Saved -> {out_pt} ({round(os.path.getsize(out_pt)/1024, 1)} KB) | Token Acc={tok_acc} | Macro-F1={tok_f1} in {train_sec}s")

    mc_path = os.path.join(MODEL_DIR, "model_card_metrics.json")
    if os.path.exists(mc_path):
        mc = json.load(open(mc_path, "r", encoding="utf-8"))
        mc["arabic_slot_bigru_tagger"] = {
            "model_name": "PyTorch Bidirectional GRU Slot Tagger (ArabicSlotBiGRU)",
            "artifact_path": "models/arabic_slot_bigru.pt",
            "artifact_kb": round(os.path.getsize(out_pt) / 1024, 1),
            "train_seconds": train_sec,
            "selected_test_metrics": {"token_accuracy": tok_acc, "macro_f1": tok_f1},
            "capabilities": [
                "Bidirectional RTL+LTR role binding (handles both 'مبيعاتي 4 مليون' and '4 مليون مبيعات')",
                "Egyptian spoken numbers ('مليون ونص', 'نص مليون', 'ميتين ألف')",
                "Monthly (×12) & Daily (×300) auto-annualizer + Negation scope guard ('معليش غرامات')",
            ],
        }
        json.dump(mc, open(mc_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
