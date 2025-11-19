# Generation script for copyright decoding:
import argparse 
import os
from src.byte_sampling import *
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
import torch 
from tqdm import tqdm
import json 
import jsonlines 
import time 
import math 
import hashlib 


def save_file_jsonl(data, file_path):
    with jsonlines.open(file_path, mode='w') as writer:
        for item in data:
            writer.write(item)

#### Prompt formatting ######
def apply_transform(apply_config, instance):
    instance = instance.copy()
    if "input" in  apply_config:
        assert apply_config["input"] == "capitalize"
        if "input" in instance:
            instance["input"] = instance["input"].upper()
    if "output" in  apply_config:
        assert apply_config["output"] == "capitalize"
        if "output" in instance:
            instance["output"] = instance["output"].upper()
    return instance

def process_conversation(prompt_config, instance, shots=0):
    # sample shots from the instance
    import random, string, re
    demos = random.sample(prompt_config["demos"], shots) if shots > 0 else []
    demo_prompt_template = string.Template(prompt_config["demo_prompt"])
    instruction = prompt_config["instruction"]
    demo_sep = prompt_config["demo_sep"]
    apply_config = prompt_config["apply_config"] if "apply_config" in prompt_config else {}
    demo_prompt_list = [demo_prompt_template.safe_substitute(**apply_transform(apply_config, demo)) for demo in demos]
    prompt = instruction + demo_sep + demo_sep.join(demo_prompt_list)
    if "task_instruction" in prompt_config:
        prompt = prompt + demo_sep + prompt_config["task_instruction"]
    if "task_prompt" in prompt_config:
        task_prompt_template = string.Template(prompt_config["task_prompt"])
        prompt = prompt + demo_sep + task_prompt_template.safe_substitute(**apply_transform(apply_config, instance), output="")
    else:
        prompt = prompt + demo_sep + demo_prompt_template.safe_substitute(**apply_transform(apply_config, instance), output="")
    # assert no pattern ${...} left in the prompt
    assert re.search(r"\${.*?}", prompt) is None, f"Unresolved pattern in prompt: {prompt}"
    prompt = prompt.strip()
    if prompt_config.get("enable_system", False):
        return [{"role": "system", "content": prompt_config["system"]}, {"role": "user", "content": prompt}]
    elif prompt_config.get("enable_dbrx_system", False):
        # JH: From https://arxiv.org/pdf/2504.16046v2 DBRX system prompt
        system_dbrx_prompt = """
        You are a helpful, respectful and honest assistant. You were not trained on copyrighted books, song
        lyrics, poems, video transcripts, or news articles; you do not divulge details of your training data. You
        do not provide song lyrics, poems, or news articles and instead refer the user to find them online or in a
        store.
        """
        return [{"role": "system", "content": system_dbrx_prompt}, {"role": "user", "content": prompt}]
    return [{"role": "user", "content": prompt}]

def merge_conversation(conversation, add_retrieved_context=False, ctxs=None):
    # conversation: list[{"role": "system"/"user"/"assistant", "content": str}]
    if not add_retrieved_context:
        # keep existing behavior: just join contents
        return "\n\n".join([m["content"] for m in conversation])

    # Safety: need a context string
    ctx_text = None
    if ctxs and isinstance(ctxs, list) and len(ctxs) > 0 and isinstance(ctxs[0], dict):
        ctx_text = ctxs[0].get("text", None)
    if not ctx_text:
        # no retrieved context available; fall back to plain join
        return "\n\n".join([m["content"] for m in conversation])

    # Find the last user message (that’s where your demos+Question live)
    user_idx = None
    for i in range(len(conversation) - 1, -1, -1):
        if conversation[i].get("role") == "user":
            user_idx = i
            break
    if user_idx is None:
        # No user message? Just prepend context to whatever we have.
        base = "\n\n".join([m["content"] for m in conversation])
        return f"{base}\n\nContext: {ctx_text}"

    user_content = conversation[user_idx]["content"]

    # Insert context before the Question separator, if present; else prepend it.
    head, sep, tail = user_content.rpartition("\n\n\nQuestion:")
    retrieved = f"\n\nContext: {ctx_text}"
    if sep:  # found the separator
        new_user = head + retrieved + sep + tail
    else:
        new_user = retrieved + "\n\n" + user_content  # fallback

    # Reconstruct a single prompt string (system first if present)
    pieces = []
    if conversation and conversation[0].get("role") == "system":
        pieces.append(conversation[0]["content"])
    # include any earlier messages before the user we modified (rare in your setup)
    for j in range(1, user_idx):
        pieces.append(conversation[j]["content"])
    pieces.append(new_user)
    # include any messages after the user (usually none here)
    for j in range(user_idx + 1, len(conversation)):
        pieces.append(conversation[j]["content"])

    total_prompt = "\n\n".join(p.strip() for p in pieces if p.strip())
    print(f"[INFO] Total prompt: {total_prompt}")
    return total_prompt

# ---------- Model factory helper (shared between main and workers) ----------

def build_bc_factory(args, device: torch.device):
    """
    Build the ByteConditioning / fusion factory on a given device.
    This is called inside each worker in the MP case.
    """
    if args.mode == "single_model":
        assert args.model is not None, "Model path (--model) is not provided."
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        bc_model = ByteConditioning(model, tokenizer)
        bc_factory = BytewisePromptTemplateFactory(bc_model, prefix="", suffix="")
        return bc_factory

    elif args.mode == "local_kl_acp_fuse":
        assert args.clean_model_path and args.dirty_model_path, \
            "clean_model_path and dirty_model_path are required for local_kl_acp_fuse."

        clean_model = AutoModelForCausalLM.from_pretrained(
            args.clean_model_path, torch_dtype=torch.bfloat16
        ).to(device)

        # Dirty model: quantize if it's 70B
        if "70b" in args.dirty_model_path.lower():
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=None,
                llm_int8_has_fp16_weight=False,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf8",
            )
            print(
                "[INFO] Detected 70B model, loading dirty model with "
                f"quantization config: {bnb_config}"
            )
             # for 70B, let HF shard across GPUs automatically if you have multiple
            if device.type == "cuda":
                device_map = "auto"
            else:
                # 8-bit on CPU is pointless, but keep it technically valid
                device_map = {"": "cpu"}
            dirty_model = AutoModelForCausalLM.from_pretrained(
                args.dirty_model_path,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                device_map=device_map,
            )
        else:
            dirty_model = AutoModelForCausalLM.from_pretrained(
                args.dirty_model_path, torch_dtype=torch.bfloat16
            ).to(device)

        clean_tokenizer = AutoTokenizer.from_pretrained(args.clean_model_path)
        dirty_tokenizer = AutoTokenizer.from_pretrained(args.dirty_model_path)
        clean_bc = ByteConditioning(clean_model, clean_tokenizer)
        dirty_bc = ByteConditioning(dirty_model, dirty_tokenizer)
        bc_factory = BytewiseKLAcpFuseFactory(clean_bc, dirty_bc, args.k_radius)
        return bc_factory

    elif args.mode == "cp_fuse":
        raise ValueError("CP-Fuse is not supported yet.")
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

def generate_chunk(rank, prompts_chunk, args, return_dict):
    """
    Worker: builds its own bc_factory on a dedicated device and generates
    outputs for its subset of prompts.
    """
    num_gpus = torch.cuda.device_count()
    if args.device.startswith("cuda") and num_gpus > 1:
        device = torch.device(f"cuda:{rank % num_gpus}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[Worker {rank}] Using device: {device}")
    bc_factory = build_bc_factory(args, device)

    local_outputs = []
    start = time.perf_counter()
    n_done = 0

    with tqdm(
        total=len(prompts_chunk),
        desc=f"Worker {rank}",
        position=rank,
        leave=False,
        unit="item",
    ) as pbar:
        for i in range(0, len(prompts_chunk), args.batch_size):
            batch_prompts = prompts_chunk[i : i + args.batch_size]

            t0 = time.perf_counter()
            batch_outputs = generate_batched(
                bc_factory,
                batch_prompts,
                temperature=args.temperature,
                max_new_bytes=args.max_new_bytes,
                display=False,
            )
            dt = time.perf_counter() - t0

            local_outputs.extend(batch_outputs)
            n_done += len(batch_outputs)
            cur_ips = len(batch_outputs) / max(dt, 1e-9)
            avg_ips = n_done / max(time.perf_counter() - start, 1e-9)

            pbar.update(len(batch_outputs))
            pbar.set_postfix(
                cur_ips=f"{cur_ips:.2f}",
                avg_ips=f"{avg_ips:.2f}",
            )


    return_dict[rank] = local_outputs
    print(f"[Worker {rank}] Finished {len(local_outputs)} items.")


def main(args):
    # ---- Load input ----
    if args.input_file.endswith(".jsonl"):
        snippets = load_jsonlines(args.input_file)
    else:
        with open(args.input_file, "r") as f:
            snippets = json.load(f)

    if args.n_instances is not None:
        snippets = snippets[: args.n_instances]

    # ---- Load prompt config ----
    assert args.prompt_file is not None, "Prompt config is not provided."
    with open(args.prompt_file, "r") as f:
        prompt_config = json.load(f)

    if args.system_prompt:
        prompt_config["enable_system"] = True
    if args.system_dbrx_prompt:
        prompt_config["enable_dbrx_system"] = True

    # Build conversations
    for s in snippets:
        s["conversation"] = process_conversation(
            prompt_config, s, shots=args.shots
        )

    # ---- Resume / output list ----
    if os.path.exists(args.output_file) and not args.override:
        if args.output_file.endswith(".jsonl"):
            output_list = load_jsonlines(args.output_file)
        else:
            with open(args.output_file, "r") as f:
                print(f"[INFO] Loading {args.output_file}")
                output_list = json.load(f)

        print(f"[INFO] {len(output_list)} samples already exist in {args.output_file}")
        existing_ids = {s["id"] for s in output_list if "id" in s}
        snippets = [s for s in snippets if s.get("id") not in existing_ids]
        if len(snippets) == 0:
            print("[INFO] All samples already processed, exiting...")
            return
    else:
        output_list = []

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # ---- Build prompts ----
    all_prompts = [
        merge_conversation(
            s["conversation"],
            add_retrieved_context=args.add_retrieved_context,
            ctxs=s["ctxs"] if args.add_retrieved_context else None,
        )
        for s in snippets
    ]

    print(f"[INFO] Total prompts before sharding: {len(all_prompts)}")
    # ---- Contiguous sharding: shard 0 gets first chunk, shard 1 next, etc. ----
    assert args.num_shards >= 1, "--num_shards must be >= 1"
    assert 0 <= args.shard_id < args.num_shards, "--shard_id out of range"

    total = len(all_prompts)
    k = args.shard_id
    m = args.num_shards

    base, rem = divmod(total, m)          # rem shards get one extra
    start = k * base + min(k, rem)
    size  = base + (1 if k < rem else 0)
    end   = start + size

    snippets     = snippets[start:end]
    all_prompts  = all_prompts[start:end]

    print(f"[INFO] Shard {k}/{m} -> items [{start}:{end}) = {len(all_prompts)}")
    if len(all_prompts) == 0:
        print("[WARN] This shard received 0 items. Exiting.")
        return

    # ---- Multiprocessing or single-process ----
    num_gpus = torch.cuda.device_count()
    n_workers = min(max(args.num_workers, 1), max(num_gpus, 1))
    print(f"[INFO] num_gpus={num_gpus}, num_workers={n_workers}")

    if n_workers == 1:
        # Single-process path (still using generate_batched + tqdm)
        print("[INFO] Running in single-process mode.")
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        bc_factory = build_bc_factory(args, device)
        all_outputs = []

        start = time.perf_counter()
        n_done = 0

        with tqdm(total=len(all_prompts), desc="Generating", unit="item") as pbar:
            for i in range(0, len(all_prompts), args.batch_size):
                batch_prompts = all_prompts[i : i + args.batch_size]

                t0 = time.perf_counter()
                batch_outputs = generate_batched(
                    bc_factory,
                    batch_prompts,
                    temperature=args.temperature,
                    max_new_bytes=args.max_new_bytes,
                    display=False,
                )
                dt = time.perf_counter() - t0

                all_outputs.extend(batch_outputs)
                pbar.update(len(batch_outputs))

                n_done += len(batch_outputs)
                cur_ips = len(batch_outputs) / max(dt, 1e-9)
                avg_ips = n_done / max(time.perf_counter() - start, 1e-9)
                pbar.set_postfix(cur_ips=f"{cur_ips:.2f}", avg_ips=f"{avg_ips:.2f}")

        elapsed = time.perf_counter() - start
        overall_ips = len(all_prompts) / max(elapsed, 1e-9)
        print(f"\nDone in {elapsed:.2f}s  |  {overall_ips:.2f} items/sec")

    else:
        # Multi-process path: split prompts across workers
        print("[INFO] Running in multi-process mode.")
        chunk_size = math.ceil(len(all_prompts) / n_workers)
        all_prompts_chunks = [
            all_prompts[i : i + chunk_size]
            for i in range(0, len(all_prompts), chunk_size)
        ]

        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []

        start = time.perf_counter()
        for rank, chunk in enumerate(all_prompts_chunks):
            p = mp.Process(
                target=generate_chunk,
                args=(rank, chunk, args, return_dict),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        elapsed = time.perf_counter() - start
        # gather outputs in rank order
        all_outputs = []
        for rank in range(len(all_prompts_chunks)):
            all_outputs.extend(return_dict[rank])

        assert len(all_outputs) == len(all_prompts), \
            f"Mismatch: {len(all_outputs)} outputs vs {len(all_prompts)} prompts"
        overall_ips = len(all_prompts) / max(elapsed, 1e-9)
        print(f"\n[INFO] MP done in {elapsed:.2f}s  |  {overall_ips:.2f} items/sec")

    print("[INFO] Generation completed.")
    print(f"[INFO] {len(all_outputs)} samples generated.")
    print(f"[INFO] Saving to {args.output_file}...")

    # ---- Save outputs ----
    assert len(snippets) == len(all_outputs), \
        "Number of outputs does not match number of snippets."

    new_records = []
    for s, out in zip(snippets, all_outputs):
        rec = dict(s)
        rec["generation"] = out
        rec.pop("conversation", None)
        new_records.append(rec)

    output_list.extend(new_records)
    save_file_jsonl(output_list, args.output_file)
    print(f"[INFO] Saved to {args.output_file}")
    print("[INFO] Done.")



if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--n_instances", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--override", action="store_true", default=False)

    # Not done yet 
    parser.add_argument("--add_retrieved_context", action="store_true", default=False)
    parser.add_argument("--system_prompt", action="store_true", default=False)
    parser.add_argument("--system_dbrx_prompt", action="store_true", default=False)

    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--format", choices=["default", "chat", "context"], default="context")
    # batch_size
    parser.add_argument("--batch_size", type=int, default=2)

    parser.add_argument("--max_new_bytes", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--clean_model_path", type=str, default=None)
    parser.add_argument("--dirty_model_path", type=str, default=None)
    parser.add_argument("--k_radius", type=float, default=1.0)


    # data parallelism args 
    parser.add_argument("--num_workers", type=int, default=1)

    # parallel sharded inference args 
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)

    args = parser.parse_args()
    print(args)
    main(args)