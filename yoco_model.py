from transformers import PreTrainedModel, LlamaConfig, LlamaForCausalLM
from yocotest.yoco_decoders import SelfDecoder, CrossDecoder

class YOCOWithLLaMA(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        llama_model = LlamaForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B')
        self.self_decoder = SelfDecoder(llama_model.model.decoder)
        self.cross_decoder = CrossDecoder(config)

    def forward(self, input_ids, attention_mask=None, rel_pos=None):
        kv_caches = self.self_decoder(input_ids, attention_mask)
        output = self.cross_decoder(kv_caches, attention_mask, rel_pos)
        return output

    def save_pretrained(self, save_directory):
        super().save_pretrained(save_directory)

