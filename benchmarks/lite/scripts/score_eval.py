"""Score generated evaluation sets: WER (en) / CER (ko) with Whisper large-v3-turbo,
speaker similarity (WavLM-SV cosine vs the reference clip) for cloning items.

    python score_eval.py <out>/eval/<dir> [<dir> ...]

Needs faster-whisper, librosa and torchaudio in addition to the lite runtime.
"""

import json
import os
import re
import sys
import unicodedata

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REF_WAV  # noqa: E402

_asr = None
_sv = None


def asr():
    global _asr
    if _asr is None:
        from faster_whisper import WhisperModel

        _asr = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=4)
    return _asr


def norm_en(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_KO_NUM = {"30분": "삼십분", "3층": "삼층"}


def norm_ko(s):
    s = unicodedata.normalize("NFKC", s)
    for a, b in _KO_NUM.items():  # Whisper writes spoken numerals as digits
        s = s.replace(a, b)
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", s)


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def sv_embed(path):
    global _sv
    import librosa
    import torch

    if _sv is None:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        name = "microsoft/wavlm-base-plus-sv"
        _sv = (AutoFeatureExtractor.from_pretrained(name), WavLMForXVector.from_pretrained(name).eval())
    fe, model = _sv
    wav, _ = librosa.load(path, sr=16000, mono=True)
    with torch.inference_mode():
        emb = model(**fe(wav, sampling_rate=16000, return_tensors="pt")).embeddings
    return torch.nn.functional.normalize(emb, dim=-1)[0]


_squim = None
SQUIM_REF = REF_WAV


def squim(path):
    """Non-intrusive quality estimates: (MOS, PESQ, SI-SDR)."""
    global _squim
    import librosa
    import torch

    if _squim is None:
        from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE

        ref, _ = librosa.load(SQUIM_REF, sr=16000, mono=True)
        _squim = (SQUIM_SUBJECTIVE.get_model(), SQUIM_OBJECTIVE.get_model(), torch.from_numpy(ref)[None])
    subj, obj, ref = _squim
    x, _ = librosa.load(path, sr=16000, mono=True)
    x = torch.from_numpy(x)[None]
    with torch.inference_mode():
        mos = subj(x, ref).item()
        _, pesq, sisdr = obj(x)
    return mos, pesq.item(), sisdr.item()


def score_dir(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    out_path = os.path.join(d, "scores.json")
    if os.path.exists(out_path):
        return json.load(open(out_path))
    agg = {"en": [0, 0], "ko": [0, 0]}
    sims = []
    ref_emb = {}
    rows = []
    for it in meta["items"]:
        wav = os.path.join(d, it["id"] + ".wav")
        segs, _ = asr().transcribe(wav, language=it["lang"], beam_size=5, condition_on_previous_text=False)
        hyp = " ".join(s.text for s in segs).strip()
        if it["lang"] == "en":
            r, h = norm_en(it["text"]).split(), norm_en(hyp).split()
        else:
            r, h = list(norm_ko(it["text"])), list(norm_ko(hyp))
        err = edit_distance(r, h)
        agg[it["lang"]][0] += err
        agg[it["lang"]][1] += len(r)
        row = {"id": it["id"], "err": err, "n": len(r), "hyp": hyp, "audio_s": it["audio_s"]}
        row["mos"], row["pesq"], row["sisdr"] = squim(wav)
        if it.get("ref"):
            if it["ref"] not in ref_emb:
                ref_emb[it["ref"]] = sv_embed(it["ref"])
            row["sim"] = float((sv_embed(wav) * ref_emb[it["ref"]]).sum())
            sims.append(row["sim"])
        rows.append(row)
    res = {
        "name": os.path.basename(d.rstrip("/")),
        "wer_en": round(100 * agg["en"][0] / max(agg["en"][1], 1), 2),
        "cer_ko": round(100 * agg["ko"][0] / max(agg["ko"][1], 1), 2),
        "sim": round(float(np.mean(sims)), 3) if sims else None,
        "mos": round(float(np.mean([x["mos"] for x in rows])), 3),
        "mos_min": round(float(np.min([x["mos"] for x in rows])), 3),
        "pesq": round(float(np.mean([x["pesq"] for x in rows])), 3),
        "sisdr": round(float(np.mean([x["sisdr"] for x in rows])), 2),
        "rtf": meta.get("rtf"),
        "audio_s_total": round(sum(i["audio_s"] for i in meta["items"]), 1),
        "peak_rss_mb": meta.get("mem", {}).get("peak_rss_mb"),
        "rows": rows,
    }
    json.dump(res, open(out_path, "w"), indent=1, ensure_ascii=False)
    return res


def main():
    results = [score_dir(d) for d in sys.argv[1:]]
    print(
        f"{'config':30s} {'WER-en%':>8s} {'CER-ko%':>8s} {'SIM':>6s} {'MOS':>5s} {'minMOS':>6s} {'PESQ':>5s} {'SISDR':>6s} {'RTF':>6s} {'audio_s':>8s}"
    )
    for r in results:
        sim = "-" if r["sim"] is None else f"{r['sim']:.3f}"
        print(
            f"{r['name']:30s} {r['wer_en']:8.2f} {r['cer_ko']:8.2f} {sim:>6s} {r['mos']:5.2f} {r['mos_min']:6.2f} {r['pesq']:5.2f} {r['sisdr']:6.1f} {r['rtf']:6.2f} {r['audio_s_total']:8.1f}"
        )
        bad = [x for x in r["rows"] if x["err"] > 0]
        for x in bad:
            print(f"      {x['id']}: err {x['err']}/{x['n']}  hyp: {x['hyp'][:90]}")


if __name__ == "__main__":
    main()
