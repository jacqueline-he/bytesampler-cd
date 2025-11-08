from byte_sampling import *
from byte_sampling.radix_cache import RadixCacheManager
from byte_sampling.streaming_bpe import StreamingBPE
from byte_sampling.utils import scatter_logsumexp


def rolling_byte_logprobs(bc: ByteConditioning, text: bytes | str):
    if isinstance(text, str):
        text = text.encode()

    # TODO: stop rebuilding this list every time
    STOP_TOKENS = torch.tensor(
        [tid for tid, at in bc.tokenizer.added_tokens_decoder.items() if at.special],
        device=bc.device,
    )

    S = bc.get_streaming_byte_tree()
    trunk, roots, trees = [bc.bos], [], []
    pointer = {}
    full_tree = {bc.bos: pointer}

    for i, b in enumerate(text):
        tree = S.eval_tree(inclusive=True, filter_tensors=False)
        roots.append(len(trunk))
        trees.append(tree)
        StreamingBPE.tree_update(pointer, tree, copy=True)

        new_tokens = S.push(b)
        for tid in new_tokens:
            pointer = pointer.setdefault(tid, {})
        trunk.extend(new_tokens)

    rcm = RadixCacheManager(bc.model, bc.tokenizer)

    result = rcm.query([full_tree])[0]
    target_idx = 0

    def get_dists(eval_tree, lp_tree, past_bytes=0):
        byte_logprobs, stop_logprobs = [], []

        # walk the tree
        def extract_bytes(eval_tree, lp_tree, past_bytes=0):
            # print(past_bytes, eval_tree.keys(), lp_tree.keys())
            for tid, eval_subtree in eval_tree.items():
                lp_subtree = lp_tree[tid]
                if tid is None:
                    subset = eval_subtree
                    # how many bytes until the end of the prompt
                    # print(target_idx, past_bytes)
                    prompt_offset = target_idx - past_bytes

                    if prompt_offset == 0:
                        # only process special tokens at the end of the prompt
                        stop_logprobs.append(
                            torch.logsumexp(lp_subtree[STOP_TOKENS], 0)
                        )

                    selectors = bc.token_index_cache.get(prompt_offset)[subset]
                    lp_subset = lp_subtree[subset]

                    # 257th byte is for tokens with no byte representation
                    # (e.g. special ones) which are handled separately
                    byte_logprobs.append(
                        scatter_logsumexp(lp_subset, selectors, dim_size=257)
                    )

                else:
                    extract_bytes(
                        eval_subtree,
                        lp_subtree,
                        past_bytes + len(bc.vrev.get(tid, b"")),
                    )

        extract_bytes(eval_tree, lp_tree, past_bytes)
        stop_logprob = torch.logsumexp(torch.tensor(stop_logprobs, device=bc.device), 0)
        return torch.hstack(
            [
                torch.logsumexp(torch.vstack(byte_logprobs)[:, :-1], 0),
                stop_logprob,
            ]
        )

    lp_tree = result
    last_lp_root = 0
    past_bytes = 0
    dists = []
    for target_idx, (lp_root, eval_tree) in enumerate(zip(roots, trees)):
        for tid in trunk[last_lp_root:lp_root]:
            lp_tree = lp_tree[tid]
            past_bytes += len(bc.vrev.get(tid, b""))
        dists.append(get_dists(eval_tree, lp_tree, past_bytes))
        last_lp_root = lp_root

    return torch.log_softmax(torch.vstack(dists), -1)
