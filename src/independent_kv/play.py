import torch
from independent_kv.modeling_llama_unsupervised_parallel_PE import LlamaIndependentKVUnsuperForCausalLM

ckpt_dir = '/data/models/Llama-2-13b-hf'

model = LlamaIndependentKVUnsuperForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16, device_map='auto')

input_ids = torch.zeros((2, 4096), dtype=torch.long)
out = model(input_ids)
