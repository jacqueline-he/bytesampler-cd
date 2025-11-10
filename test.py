from src.byte_sampling import *
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Measure import time ---
t0 = time.perf_counter()

clean_model = AutoModelForCausalLM.from_pretrained("common-pile/comma-v0.1-2t").to("cuda")
dirty_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B").to("cuda")

clean_tokenizer = AutoTokenizer.from_pretrained("common-pile/comma-v0.1-2t")
dirty_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

clean_bc=ByteConditioning(clean_model, clean_tokenizer)
dirty_bc=ByteConditioning(dirty_model, dirty_tokenizer)

# sample a continuation with a QA formatted prompt.
prompts = ["Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. ", "Tell me a biography about Donald Trump."]
outputs = generate_batched(
    BytewiseKLAcpFuseFactory(tcs_clean=clean_bc, tcs_dirty=dirty_bc, k_radius=0.3),
    prompts,
    temperature=0.7,
    max_new_bytes=200,
    display=False
)
print(outputs)

t1 = time.perf_counter()
print(f"Sampling took {t1 - t0:.2f} seconds")
