from src.byte_sampling import *

# load in a model
clean_bc_dir="common-pile/comma-v0.1-2t"
clean_bc = ByteConditioning(clean_bc_dir)
dirty_bc_dir = "meta-llama/Llama-3.1-70B"
dirty_bc = ByteConditioning(dirty_bc_dir)

# sample a continuation with a QA formatted prompt.
prompts = ["Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense."]
# generate_batched(
#     BytewiseQAFactory(dirty_bc),
#     prompts,
#     max_new_bytes=500,
#     display=True
# )



# generate_batched(
#     BytewiseQAFactory(clean_bc),
#     prompts,
#     max_new_bytes=500,
#     display=True
# )

# Test Copyright decoding
print("\n" + "="*50)
print("Testing Copyright Decoding")
print("="*50)

generate_batched(
    BytewiseCopyrightDecodingFactory(tcs_draft=clean_bc, tcs_verify=dirty_bc, alpha=1.0),
    prompts,
    max_new_bytes=500,
    display=True
)

# # Test with different alpha/beta combinations
# print("\n" + "-"*30)
# print("Copyright decoding with alpha=0.8")
# print("-"*30)
# generate_batched(
#     BytewiseCopyrightDecodingFactory(tcs_draft=clean_bc, tcs_verify=dirty_bc, alpha=0.8),
#     prompts,
#     max_new_bytes=500,
#     display=True
# )

# print("\n" + "-"*30)
# print("Copyright decoding with alpha=1.2")
# print("-"*30)
# generate_batched(
#     BytewiseCopyrightDecodingFactory(tcs_draft=clean_bc, tcs_verify=dirty_bc, alpha=1.2),
#     prompts,
#     max_new_bytes=500,
#     display=True
# )
