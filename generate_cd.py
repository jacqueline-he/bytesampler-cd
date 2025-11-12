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


def main(args):
    if args.input_file.endswith(".jsonl"):
        snippets = load_jsonlines(args.input_file)
    else:
        with open(args.input_file, "r") as f:
            snippets = json.load(f)
    if args.n_instances is not None:
        orig_len = len(snippets)
        snippets = snippets[:args.n_instances]

    if args.prompt_file is None:
        prompt_config = None
    else:
        with open(args.prompt_file, "r") as f:
            prompt_config = json.load(f)
    assert prompt_config is not None, f"Prompt config is not provided."
    prompt_tag = prompt_config.get("tag", None)

    if args.system_prompt:
        prompt_config["enable_system"] = True
    if args.system_dbrx_prompt:
        prompt_config["enable_dbrx_system"] = True

    for s in snippets:
        s["conversation"] = process_conversation(prompt_config, s, shots=args.shots)

    if os.path.exists(args.output_file) and not args.override:
        if args.output_file.endswith(".jsonl"):
            output_list = load_jsonlines(args.output_file)
        else:
            with open(args.output_file, "r") as f:
                print(f"[INFO] Loading {args.output_file}")
                output_list = json.load(f)
        print(f"[INFO] {len(output_list)} samples already exist in {args.output_file}")
        # remove snippets that are already processed
        existing_ids = [s["id"] for s in output_list]
        snippets = [s for s in snippets if s["id"] not in existing_ids]
        if len(snippets) == 0:
            print(f"[INFO] All samples already processed, exiting...")
            exit(0)
    else:
        output_list = []
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    


    ## Load byte-sampler compatible models 
    # Currently supported are:
    # 1. single-model decoding 
    # 2. local KL-ACP Fuse 
    # 3. (TBD) CP-Fuse 
    args_dict = vars(args)
    if args.mode == "single_model":
        assert args.model is not None, "Model path is not provided."
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(args.device)
        bc_model = ByteConditioning(model)
        bc_factory = BytewisePromptTemplateFactory(bc_model, prefix="", suffix="")
    elif args.mode == "local_kl_acp_fuse":
        clean_model = AutoModelForCausalLM.from_pretrained(args.clean_model_path, torch_dtype=torch.bfloat16).to(args.device)
        if "70b" in args.dirty_model_path.lower():
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,                # enable 8-bit loading
                llm_int8_threshold=6.0,           # outlier threshold (default)
                llm_int8_skip_modules=None,       # optional list of modules to skip quantization
                llm_int8_has_fp16_weight=False,   # True if model was converted from fp16 weights
                bnb_8bit_use_double_quant=True,   # second quantization layer for memory savings
                bnb_8bit_quant_type="nf8",        # quantization type: "fp8", "nf4", or "nf8" (nf8 is good default)
            )
            dirty_model = AutoModelForCausalLM.from_pretrained(args.dirty_model_path, torch_dtype=torch.bfloat16, quantization_config=bnb_config, low_cpu_mem_usage=True, device_map="cuda")
        else:
            dirty_model = AutoModelForCausalLM.from_pretrained(args.dirty_model_path, torch_dtype=torch.bfloat16).to(args.device)

        clean_tokenizer = AutoTokenizer.from_pretrained(args.clean_model_path)
        dirty_tokenizer = AutoTokenizer.from_pretrained(args.dirty_model_path)
        clean_bc=ByteConditioning(clean_model, clean_tokenizer)
        dirty_bc=ByteConditioning(dirty_model, dirty_tokenizer)
        bc_factory = BytewiseKLAcpFuseFactory(clean_bc, dirty_bc, args.k_radius)
    elif args.mode == "cp_fuse":
        raise ValueError(f"CP-Fuse is not supported yet.")
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    print(f"[INFO] Model loaded: {args.mode}")

    all_prompts = [merge_conversation(s["conversation"], add_retrieved_context=args.add_retrieved_context, ctxs=s['ctxs'] if args.add_retrieved_context else None) for s in snippets]
    all_outputs = []

    start = time.perf_counter()
    start = time.perf_counter()
    n_done = 0

    with tqdm(total=len(all_prompts), desc="Generating", unit="item") as pbar:
        for i in range(0, len(all_prompts), args.batch_size):
            batch_prompts = all_prompts[i:i+args.batch_size]

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

            # live throughput over this batch + overall average
            n_done += len(batch_outputs)
            cur_ips = len(batch_outputs) / max(dt, 1e-9)
            avg_ips = n_done / max(time.perf_counter() - start, 1e-9)
            pbar.set_postfix(cur_ips=f"{cur_ips:.2f}", avg_ips=f"{avg_ips:.2f}")

    elapsed = time.perf_counter() - start
    overall_ips = len(all_prompts) / max(elapsed, 1e-9)
    print(f"\nDone in {elapsed:.2f}s  |  {overall_ips:.2f} items/sec")

    print(f"[INFO] Generation completed.")
    print(f"[INFO] {len(all_outputs)} samples generated.")
    print(f"[INFO] Saving to {args.output_file}...")  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--n_instances", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")

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
    args = parser.parse_args()
    print(args)
    main(args)