from src.byte_sampling import *
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Measure import time ---
t0 = time.perf_counter()

clean_model = AutoModelForCausalLM.from_pretrained("common-pile/comma-v0.1-2t",torch_dtype=torch.bfloat16,).to("cuda")
clean_model.config.num_logits_to_keep = 1
dirty_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B",torch_dtype=torch.bfloat16).to("cuda")
dirty_model.config.num_logits_to_keep = 1

clean_tokenizer = AutoTokenizer.from_pretrained("common-pile/comma-v0.1-2t")
dirty_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

clean_bc=ByteConditioning(clean_model, clean_tokenizer)
dirty_bc=ByteConditioning(dirty_model, dirty_tokenizer)

# sample a continuation with a QA formatted prompt.
prompts = ["You will be shown a series of passages from famous literary works. After these examples, you will receive a prefix from another passage and be asked to complete it based on the text of a famous work. Provide only the continuation for the last given prefix without any extra commentary, formatting, or additional text.\n\n\n\n\n\nComplete the prefix:\nyour belt, those are ours too. You have nothing to give us but your lives. How would you like to die, Tyrion son of Tywin?\" \"In my own bed, with a belly full of wine and a maiden's mouth around my cock, at the age of eighty,\" he replied. The huge one, Shagga, laughed first and loudest. The others seemed less amused. \"Conn, take their horses,\" Gunthor commanded. \"Kill the other and seize the halfinan. He can milk the goats and make the mothers laugh.\" Bronn sprang to his feet. \"Who dies first?\" \"No!\" Tyrion said sharply. \"Gunthor son of Gurn, hear me. My House is rich and powerful. If the Stone Crows will see us safely through these mountains, my lord father will shower you with gold.\" \"The gold of a lowland lord is as worthless as a halfman's promises,\" Gunthor said. \"Half a man I may be,\" Tyrion said"]
#prompts = ["Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. "]
# , "Tell me a biography about Donald Trump."
#prompts = ["hi how are you?"]
outputs = generate_batched(
    BytewiseKLAcpFuseFactory(tcs_clean=clean_bc, tcs_dirty=dirty_bc, k_radius=0.1),
    prompts,
    temperature=0.7,
    max_new_bytes=200,
    display=False
)
print(outputs)

t1 = time.perf_counter()
print(f"Sampling took {t1 - t0:.2f} seconds")
