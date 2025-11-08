from src.byte_sampling import *
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

def log_gpu_stats(device_id, label=""):
    allocated = torch.cuda.memory_allocated(device_id)
    reserved  = torch.cuda.memory_reserved(device_id)
    max_alloc = torch.cuda.max_memory_allocated(device_id)
    logger.info(f"Device cuda:{device_id} {label} — allocated: {allocated/1024**2:.2f} MB; "
                f"reserved: {reserved/1024**2:.2f} MB; peak_alloc: {max_alloc/1024**2:.2f} MB")

# --- Measure import time ---
t0 = time.perf_counter()

#### 7b / 8b ####
# clean_model_path = "common-pile/comma-v0.1-2t"
# dirty_model_path = "meta-llama/Llama-3.1-8B"val

# clean_model = AutoModelForCausalLM.from_pretrained(clean_model_path,torch_dtype=torch.bfloat16).to("cuda:0")
# dirty_model = AutoModelForCausalLM.from_pretrained(dirty_model_path,torch_dtype=torch.bfloat16).to("cuda:0")

#### 7b / 70b ####
clean_model_path =  "jacquelinehe/comma-1.7b-v5" #"common-pile/comma-v0.1-2t"
dirty_model_path = "meta-llama/Meta-Llama-3.1-8B"


# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
# )

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,                # enable 8-bit loading
    llm_int8_threshold=6.0,           # outlier threshold (default)
    llm_int8_skip_modules=None,       # optional list of modules to skip quantization
    llm_int8_has_fp16_weight=False,   # True if model was converted from fp16 weights
    bnb_8bit_use_double_quant=True,   # second quantization layer for memory savings
    bnb_8bit_quant_type="nf8",        # quantization type: "fp8", "nf4", or "nf8" (nf8 is good default)
)

clean_model = AutoModelForCausalLM.from_pretrained(clean_model_path,torch_dtype=torch.bfloat16).to("cuda:0")
dirty_model = AutoModelForCausalLM.from_pretrained(dirty_model_path,torch_dtype=torch.bfloat16, quantization_config=bnb_config, low_cpu_mem_usage=True, device_map="cuda:0")

# clean_model = AutoModelForCausalLM.from_pretrained(clean_model_path,torch_dtype=torch.bfloat16,quantization_config=bnb_config, device_map={"":0})
# dirty_model = AutoModelForCausalLM.from_pretrained(dirty_model_path,torch_dtype=torch.bfloat16,quantization_config=bnb_config, device_map={"":1})



clean_tokenizer = AutoTokenizer.from_pretrained(clean_model_path)
dirty_tokenizer = AutoTokenizer.from_pretrained(dirty_model_path)

clean_bc=ByteConditioning(clean_model, clean_tokenizer)
dirty_bc=ByteConditioning(dirty_model, dirty_tokenizer)

log_gpu_stats(0, "after clean_model load")
log_gpu_stats(1, "after dirty_model load")


# Before generation:
torch.cuda.reset_peak_memory_stats(0)
torch.cuda.reset_peak_memory_stats(1)
log_gpu_stats(0, "before generation")
log_gpu_stats(1, "before generation")

# sample a continuation with a QA formatted prompt.
#prompts = ["You will be shown a series of passages from famous literary works. After these examples, you will receive a prefix from another passage and be asked to complete it based on the text of a famous work. Provide only the continuation for the last given prefix without any extra commentary, formatting, or additional text.\n\n\n\n\n\nComplete the prefix:\nyour belt, those are ours too. You have nothing to give us but your lives. How would you like to die, Tyrion son of Tywin?\" \"In my own bed, with a belly full of wine and a maiden's mouth around my cock, at the age of eighty,\" he replied. The huge one, Shagga, laughed first and loudest. The others seemed less amused. \"Conn, take their horses,\" Gunthor commanded. \"Kill the other and seize the halfinan. He can milk the goats and make the mothers laugh.\" Bronn sprang to his feet. \"Who dies first?\" \"No!\" Tyrion said sharply. \"Gunthor son of Gurn, hear me. My House is rich and powerful. If the Stone Crows will see us safely through these mountains, my lord father will shower you with gold.\" \"The gold of a lowland lord is as worthless as a halfman's promises,\" Gunthor said. \"Half a man I may be,\" Tyrion said"]
prompts = ["hello, how are you doing?", "goodbye, have a nice day!"]
outputs = generate_batched(
    BytewiseKLAcpFuseFactory(tcs_clean=clean_bc, tcs_dirty=dirty_bc, k_radius=0.1),
    prompts,
    temperature=0.7,
    max_new_bytes=800,
    display=False
)
print(outputs)
log_gpu_stats(0, "after generation")
log_gpu_stats(1, "after generation")

t1 = time.perf_counter()
print(f"Sampling took {t1 - t0:.2f} seconds")
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
    BytewiseKLAcpFuseFactory(tcs_clean=clean_bc, tcs_dirty=dirty_bc, k_radius=0.3),
    prompts,
    temperature=0.5,
    max_new_bytes=50,
    display=True
)

# Test with different alpha/beta combinations
print("\n" + "-"*30)
print("Copyright decoding with alpha=0.8")
print("-"*30)
generate_batched(
    BytewiseCopyrightDecodingFactory(tcs_draft=clean_bc, tcs_verify=dirty_bc, alpha=0.8),
    prompts,
    max_new_bytes=500,
    display=True
)

print("\n" + "-"*30)
print("Copyright decoding with alpha=1.2")
print("-"*30)
generate_batched(
    BytewiseCopyrightDecodingFactory(tcs_draft=clean_bc, tcs_verify=dirty_bc, alpha=1.2),
    prompts,
    max_new_bytes=500,
    display=True
)

t1 = time.perf_counter()
print(f"Sampling took {t1 - t0:.2f} seconds")