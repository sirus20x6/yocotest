import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from yocotest.cross_attention import CrossAttention

class SelfDecoder(nn.Module):
    def __init__(self, llama_decoder):
        super(SelfDecoder, self).__init__()
        self.llama_decoder = llama_decoder

    def forward(self, input_ids, attention_mask=None):
        outputs = self.llama_decoder(input_ids, attention_mask=attention_mask, use_cache=True)
        return outputs.past_key_values

class CrossDecoder(nn.Module):
    def __init__(self, args):
        super(CrossDecoder, self).__init__()
        self.cross_attention = CrossAttention(args)

    def forward(self, kv_caches, attention_mask=None, rel_pos=None):
        key, value = kv_caches
        output = self.cross_attention(key, value, rel_pos, attention_mask)
        return output
