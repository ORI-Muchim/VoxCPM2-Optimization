"""Generate the evaluation set for one model configuration and several sampler variants.

    python gen_eval.py <model_dir> --name int8 --loader lite --precision int8 --variants s10c2.0,s6c2.0

Writes out/eval/<name>_<variant>/{id}.wav and meta.json (timings, memory).
"""

import argparse
import json
import os
import sys
import time

import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import load_stock, mem  # noqa: E402

REF_WAV = "/home/user/openbmb/voxcpm/examples/reference_speaker.wav"

EN = [
    "The quick brown fox jumps over the lazy dog, and then it runs away into the forest.",
    "Please remember to bring your umbrella tomorrow, because the forecast says it will rain all afternoon.",
    "She opened the old wooden door slowly and found a small garden hidden behind the house.",
    "Scientists have discovered a new species of deep sea fish living near volcanic vents.",
    "Could you tell me how to get to the central train station from here?",
    "After a long day at work, he likes to cook dinner and listen to quiet music.",
    "The museum will be closed on Monday, but it opens again early on Tuesday morning.",
    "Learning a new language takes patience, practice, and a little bit of courage.",
]
KO = [
    "오늘 날씨가 정말 좋아서 공원에서 산책을 하고 싶어요.",
    "내일 오전 회의는 삼층 회의실에서 진행될 예정입니다.",
    "어제 친구와 함께 본 영화가 생각보다 훨씬 재미있었어요.",
    "이 근처에 맛있는 한식당이 있으면 추천해 주시겠어요?",
    "새로운 언어를 배우는 것은 시간과 꾸준한 노력이 필요합니다.",
    "지하철을 타고 가면 약속 장소까지 삼십 분 정도 걸립니다.",
    "주말에는 가족들과 함께 바닷가로 여행을 떠날 계획이에요.",
    "커피 한 잔과 함께 조용히 책을 읽는 시간을 가장 좋아합니다.",
]


def build_items(n_plain=8, n_clone=3):
    items = []
    for i, t in enumerate(EN[:n_plain]):
        items.append({"id": f"en{i}", "lang": "en", "text": t, "ref": None})
    for i, t in enumerate(KO[:n_plain]):
        items.append({"id": f"ko{i}", "lang": "ko", "text": t, "ref": None})
    for i, t in enumerate(EN[:n_clone]):
        items.append({"id": f"clone_en{i}", "lang": "en", "text": t, "ref": REF_WAV})
    for i, t in enumerate(KO[:n_clone]):
        items.append({"id": f"clone_ko{i}", "lang": "ko", "text": t, "ref": REF_WAV})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--loader", choices=["stock", "lite"], default="lite")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--precision", default=None)
    ap.add_argument("--two-pass", default=None)
    ap.add_argument("--solver", default=None)
    ap.add_argument("--reduce-range", type=int, default=None)
    ap.add_argument("--variants", default="s10c2.0", help="comma list of s<steps>c<cfg>")
    ap.add_argument("--n-plain", type=int, default=8)
    ap.add_argument("--n-clone", type=int, default=3)
    ap.add_argument("--out-root", default="/home/user/work/out/eval")
    ap.add_argument("--patch", default=None, help="python file exposing patch(model, args) applied after load")
    args = ap.parse_args()

    t0 = time.time()
    if args.loader == "stock":
        model = load_stock(args.model_dir, args.dtype)
    else:
        from voxcpm.lite import load_voxcpm2_lite

        model = load_voxcpm2_lite(
            args.model_dir,
            precision=args.precision,
            two_pass=args.two_pass,
            reduce_range=None if args.reduce_range is None else bool(args.reduce_range),
        )
    load_s = time.time() - t0
    if args.solver:
        model.feat_decoder.solver = args.solver
    if args.patch:
        import importlib.util

        spec = importlib.util.spec_from_file_location("patchmod", args.patch)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.patch(model, args)

    items = build_items(args.n_plain, args.n_clone)
    for variant in args.variants.split(","):
        steps = int(variant.split("c")[0][1:])
        cfg = float(variant.split("c")[1])
        out_dir = os.path.join(args.out_root, f"{args.name}_{variant}")
        os.makedirs(out_dir, exist_ok=True)
        meta = {"name": args.name, "variant": variant, "steps": steps, "cfg": cfg, "load_s": round(load_s, 1), "items": []}
        tot_gen = tot_audio = 0.0
        for i, it in enumerate(items):
            kw = dict(target_text=it["text"], inference_timesteps=steps, cfg_value=cfg, seed=1000 + i)
            if it["ref"]:
                kw["reference_wav_path"] = it["ref"]
            t = time.perf_counter()
            wav = model.generate(**kw)
            gen = time.perf_counter() - t
            dur = wav.shape[-1] / model.sample_rate
            sf.write(os.path.join(out_dir, it["id"] + ".wav"), wav.squeeze(0).numpy(), model.sample_rate)
            meta["items"].append({**it, "gen_s": round(gen, 2), "audio_s": round(dur, 2)})
            # the first clone utterance pays the one-off reference encoding; keep it out of RTF
            if not (it["ref"] and it["id"].endswith("0")):
                tot_gen += gen
                tot_audio += dur
            print(f"[{args.name}_{variant}] {it['id']}: {dur:.2f}s audio in {gen:.1f}s", flush=True)
        meta["rtf"] = round(tot_gen / tot_audio, 3)
        meta["mem"] = mem()
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)
        print(f"[{args.name}_{variant}] RTF {meta['rtf']}  mem {meta['mem']}", flush=True)


if __name__ == "__main__":
    main()
