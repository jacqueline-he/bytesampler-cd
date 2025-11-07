from src.byte_sampling import *
import time

# --- Measure import time ---
t0 = time.perf_counter()

# load in a model
clean_bc_dir="common-pile/comma-v0.1-2t"
clean_bc = ByteConditioning(clean_bc_dir)
dirty_bc_dir = "meta-llama/Llama-3.1-8B"
dirty_bc = ByteConditioning(dirty_bc_dir)

# sample a continuation with a QA formatted prompt.
prompts = ["Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. "]
generate_batched(
    BytewiseKLAcpFuseFactory(tcs_clean=clean_bc, tcs_dirty=dirty_bc, k_radius=1.0),
    prompts,
    temperature=0.5,
    max_new_bytes=50,
    display=True
)

t1 = time.perf_counter()
print(f"Sampling took {t1 - t0:.2f} seconds")
