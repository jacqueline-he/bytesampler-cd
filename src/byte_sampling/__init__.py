import uuid
from typing import Union
import pdb
import torch
import torch.nn.functional as F 

from .byte_conditioning import ByteConditioning
from .utils import sample_from_logits
from .acpfuse_utils import solve_optimization, get_fused_logp_from_weights


class EnsembleBytewiseSamplerFactory:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_bytewise_sampler(self, batch_size):
        return EnsembleBytewiseSampler(batch_size, *self.args, **self.kwargs)


class EnsembleBytewiseSampler:
    def __init__(self, batch_size, bcs: list[ByteConditioning], mode="mix", **kwargs):
        self.batch_size = batch_size
        self.bcs = bcs
        self.bss = [bc.get_bytewise_sampler(self.batch_size) for bc in bcs]
        self.mode = mode
        self.kwargs = kwargs

    def add_context(self, prompts: list[Union[str, bytes]]):
        for bs in self.bss:
            bs.add_context(prompts)

    def get_dists(self, **kwargs):
        logits = torch.stack([bs.get_dists(**kwargs) for bs in self.bss], 0).moveaxis(
            1, 0
        )
        logprobs = torch.log_softmax(logits, -1)
        if self.mode == "mix":
            return torch.log_softmax(torch.logsumexp(logprobs, 1), 1)
        elif self.mode == "product":
            power = self.kwargs.get("power", 1 / len(self.bss))
            return torch.log_softmax(logprobs.sum(1) * power, 1)


class BytewisePromptTemplateFactory:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_bytewise_sampler(self, batch_size):
        return BytewisePromptTemplate(batch_size, *self.args, **self.kwargs)

class BytewiseContrastiveDecodingFactory:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_bytewise_sampler(self, batch_size):
        return BytewiseContrastiveDecoding(batch_size, *self.args, **self.kwargs)

class BytewiseKLAcpFuseFactory:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_bytewise_sampler(self, batch_size):
        return BytewiseKLAcpFuse(batch_size, *self.args, **self.kwargs)

class BytewiseKLAcpFuse:
    def __init__(self, batch_size, tcs_clean, tcs_dirty, k_radius=1.0, **kwargs):
        self.batch_size = batch_size
        self.tcs_clean = tcs_clean
        self.tcs_dirty = tcs_dirty
        self.k_radius = k_radius
        self.kwargs = kwargs
        self.bs_clean = tcs_clean.get_bytewise_sampler(batch_size=batch_size)
        self.bs_dirty = tcs_dirty.get_bytewise_sampler(batch_size=batch_size)

        self.bss = [self.bs_clean, self.bs_dirty]
        self.kwargs = kwargs 

    def get_dists(self, **kwargs):
        logits = torch.stack([bs.get_dists(**kwargs) for bs in self.bss], 0).moveaxis(
            1, 0
        )
        clean_logits, dirty_logits = logits[:, 0, :], logits[:, 1, :] # 1, 257 (byte-level)
        # why are clean_logits, dirty_logits sometimes -inf? is this to mask out invalid continuations to the prefix? 
        bc, bd = solve_optimization(clean_logits, dirty_logits, self.k_radius)
        print(f" [DEBUG] bc: {bc}, bd: {bd}")
        fused_log_probs, _, fused_next_token_logits = get_fused_logp_from_weights(bc, bd, clean_logits, dirty_logits)
        return fused_log_probs

    def add_context(self, prompts: list[Union[str, bytes]]):
        for bs in self.bss:
            bs.add_context(prompts)



class BytewiseContrastiveDecoding:
    def __init__(
        self, batch_size, tcs_amateur, tcs_expert, alpha=1.0, beta=1.0, **kwargs
    ):
        """
        Contrastive decoding at the byte level.
        
        Args:
            batch_size: Number of sequences to process in parallel
            tcs_amateur: ByteConditioning model for amateur generation
            tcs_expert: ByteConditioning model for verification/scoring
            alpha: Weight for the amateur model logits (default: 1.0)
            beta: Weight for the contrastive term (default: 1.0)
            **kwargs: Additional arguments
        """
        self.batch_size = batch_size
        self.tcs_amateur = tcs_amateur
        self.tcs_expert = tcs_expert
        self.bs_amateur = tcs_amateur.get_bytewise_sampler(batch_size=batch_size)
        self.bs_expert = tcs_expert.get_bytewise_sampler(batch_size=batch_size)
        self.kwargs = kwargs
        self.alpha = alpha
        self.beta = beta

    def add_context(self, prompts: list[Union[str, bytes]]):
        """Add context to both amateur and expert models."""
        self.bs_amateur.add_context(prompts)
        self.bs_expert.add_context(prompts)

    def get_dists(self):
        """
        Compute contrastive decoding distributions.
        
        The contrastive decoding formula is:
        logits = α * logits_amateur + β * (logits_expert - logits_amateur)
        
        This can be rewritten as:
        logits = (α - β) * logits_amateur + β * logits_expert
        """
        logits_amateur = self.bs_amateur.get_dists()
        logits_expert = self.bs_expert.get_dists()
        
        # Convert to log probabilities
        logprobs_amateur = torch.log_softmax(logits_amateur, -1)
        logprobs_expert = torch.log_softmax(logits_expert, -1)
        
        # Apply contrastive decoding formula
        # logits = (α - β) * logits_amateur + β * logits_expert
        combined_logits = (self.alpha - self.beta) * logits_amateur + self.beta * logits_expert
        # pdb.set_trace()
        return torch.log_softmax(combined_logits, -1)


class BytewisePromptTemplate:
    def __init__(self, batch_size, bc, prefix, suffix, **kwargs):
        self.batch_size = batch_size
        self.bc = bc
        self.bs = bc.get_bytewise_sampler(batch_size)
        self.kwargs = kwargs
        self.prompt_added = False
        self.template_prefix, self.template_suffix = prefix, suffix

    def add_context(self, prompts: list[Union[str, bytes]]):
        if not self.prompt_added and self.template_prefix is not None:
            batch = [self.template_prefix] * self.batch_size
            if isinstance(self.template_prefix, (str, bytes)):
                self.bs.add_context(batch)
            else:
                self.bs.add_special_context(batch)

        self.bs.add_context(prompts)

        if not self.prompt_added and self.template_suffix is not None:
            batch = [self.template_suffix] * self.batch_size
            if isinstance(self.template_suffix, (str, bytes)):
                self.bs.add_context(batch)
            else:
                self.bs.add_special_context(batch)

            self.prompt_added = True

    def get_dists(self, **kwargs):
        return self.bs.get_dists(**kwargs)


class BytewiseInstructFactory:
    def __init__(self, bc, extra_suffix="", *args, **kwargs):
        self.bc = bc
        self.args = args
        self.kwargs = kwargs
        self.prefix, self.suffix = self.extract_chat_template(extra_suffix)

    def extract_chat_template(self, extra_suffix):
        sentinel = str(uuid.uuid4())
        template = self.bc.tokenizer.apply_chat_template(
            [{"role": "user", "content": sentinel}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix, suffix = template.split(sentinel)
        return self.bc.tokenizer.encode(
            prefix, add_special_tokens=False
        ), self.bc.tokenizer.encode(suffix + extra_suffix, add_special_tokens=False)

    def get_bytewise_sampler(self, batch_size):
        return BytewisePromptTemplate(
            batch_size, self.bc, self.prefix, self.suffix, *self.args, **self.kwargs
        )


class BytewiseQAFactory:
    """Options for QA format:
    qa: Question: {question}\nAnswer: {answer}
    qnan: Question:\n{question}\nAnswer:\n{answer}
    qna: Question:\n{question}\nAnswer: {answer}
    q: Question: {question} (if answer=None, else equivalent to qa)
    """

    def __init__(self, bc, mode="qa", *args, **kwargs):
        self.bc = bc
        self.args = args
        self.kwargs = kwargs

        if mode == "qa":
            self.prefix, self.suffix = "Question: ", "\nAnswer: "
        elif mode == "qnan":
            self.prefix, self.suffix = "Question:\n", "\nAnswer:\n"
        elif mode == "qna":
            self.prefix, self.suffix = "Question:\n", "\nAnswer: "
        else:
            raise NotImplementedError(f"Unknown mode {mode!r}")

    def get_bytewise_sampler(self, batch_size):
        return BytewisePromptTemplate(
            batch_size, self.bc, self.prefix, self.suffix, *self.args, **self.kwargs
        )


class BytewiseProxyTuningFactory:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_bytewise_sampler(self, batch_size):
        return BytewiseProxyTuning(batch_size, *self.args, **self.kwargs)


class BytewiseProxyTuning:
    def __init__(
        self, batch_size, bc_base, bc_expert, bc_antiexpert, alpha=1, **kwargs
    ):
        self.batch_size = batch_size
        self.bc_base = bc_base
        self.bc_expert = bc_expert
        self.bc_antiexpert = bc_antiexpert
        self.bs_base = bc_base.get_bytewise_sampler(batch_size=batch_size)
        self.bs_expert = bc_expert.get_bytewise_sampler(batch_size=batch_size)
        self.bs_antiexpert = bc_antiexpert.get_bytewise_sampler(batch_size=batch_size)
        self.bss = [self.bs_base, self.bs_expert, self.bs_antiexpert]
        self.kwargs = kwargs
        self.alpha = alpha

    @staticmethod
    def extract_chat_template(tokenizer):
        sentinel = str(uuid.uuid4())
        template = tokenizer.apply_chat_template(
            [{"role": "user", "content": sentinel}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix, suffix = template.split(sentinel)
        return tokenizer.encode(prefix), tokenizer.encode(suffix)

    def add_context(self, prompts: list[Union[str, bytes]]):
        for bs in self.bss:
            bs.add_context(prompts)

    def get_dists(self, **kwargs):
        logits = torch.stack([bs.get_dists(**kwargs) for bs in self.bss], 0).moveaxis(
            1, 0
        )
        logprobs = torch.log_softmax(logits, -1)
        # Do the proxy tuning!
        # print(self.alpha)
        return torch.log_softmax(
            logprobs[:, 0, :] + (logprobs[:, 1, :] - logprobs[:, 2, :]) * self.alpha, 1
        )


@torch.inference_mode()
def generate_batched(
    sampler_factory,
    prompts: list[str],
    max_new_bytes: int = 100,
    do_sample: bool = True,
    temperature: float = 1,
    top_k: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    display: bool = False,
    stop_strings: tuple[str] = (),
    include_stop_str_in_output: bool = False,
    allow_special: bool = True,
    logprob_transforms=None,
):
    assert not isinstance(
        stop_strings, str
    ), "stop_strings should be a sequence of strings"
    stop_strings = tuple(sorted(stop_strings, key=len, reverse=True))
    assert not isinstance(prompts, str)
    assert seed is None or generator is None, "can pass only one of seed/generator"

    bsize = len(prompts)
    assert not (display and bsize > 1)

    try:
        bs = sampler_factory.get_bytewise_sampler(batch_size=bsize)
    except AttributeError:
        bs = sampler_factory

    bs.add_context([prompt.encode() for prompt in prompts])

    outputs = [[] for _ in range(bsize)]
    decode_bufs = [b"" for _ in range(bsize)]
    stop_found = [False for _ in range(bsize)]

    if display:
        print(prompts[0], end="", flush=True)

    for _ in range(max_new_bytes):
        dists = bs.get_dists(logprob_transforms=logprob_transforms)
        if not allow_special:
            dists[:, 256:] = -torch.inf

        # init the generator late so we know which device to put it on
        if generator is None and seed is not None:
            generator = torch.Generator(device=dists.device).manual_seed(seed)

        new_bytes = sample_from_logits(
            dists,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        ).tolist()

        for i, new_byte in enumerate(new_bytes):
            if new_byte >= 256:
                stop_found[i] = True

        new_bytes = [
            bytes([b]) if not sf else bytes() for b, sf in zip(new_bytes, stop_found)
        ]

        bs.add_context(new_bytes)

        for i, new_byte in enumerate(new_bytes):
            if stop_found[i]:
                continue
            try:
                decode_bufs[i] += new_byte
                char = decode_bufs[i].decode()
                outputs[i].append(char)
                if display:
                    print(char, end="", flush=True)
                decode_bufs[i] = b""
            except UnicodeDecodeError:
                pass

        if stop_strings:
            for i, output in enumerate(outputs):
                if stop_found[i]:
                    continue

                suffix = "".join(output[-max(map(len, stop_strings)) :])
                if suffix.endswith(stop_strings):
                    if not include_stop_str_in_output:
                        for stop in stop_strings:
                            if suffix.endswith(stop):
                                outputs[i] = output[: -len(stop)]
                                break

                    stop_found[i] = True

        if all(stop_found):
            break

    # print(decode_bufs)
    return ["".join(output) for output in outputs]
