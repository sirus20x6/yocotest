import torch
from transformers import LlamaForCausalLM
from yocotest.yoco_model import YOCOWithLLaMA

def transfer_weights(llama_model, yoco_model):
    yoco_model.self_decoder.llama_decoder.load_state_dict(llama_model.model.decoder.state_dict())
    # Additional weight transfers if necessary

# Transfer weights
llama_model = LlamaForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B')
config = llama_model.config
yoco_llama_model = YOCOWithLLaMA(config)
transfer_weights(llama_model, yoco_llama_model)

# Save the converted model
yoco_llama_model.save_pretrained('path_to_save_yoco_llama')



# Fine-tune the model if necessary
