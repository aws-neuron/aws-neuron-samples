import os
import torch
import torch_neuron
import numpy as np
from torch.nn import functional as F

from transformers import MarianMTModel, MarianTokenizer, MarianConfig
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.modeling_utils import PreTrainedModel

def infer(model, tokenizer, text, num_beams=4, max_encoder_length=32, max_decoder_length=32):

    # Truncate and pad the max length to ensure that the token size is compatible with fixed-sized encoder (Not necessary for pure CPU execution)
    batch = tokenizer(text, max_length=max_decoder_length, truncation=True, padding='max_length', return_tensors="pt")
    output = model.generate(**batch, max_length=max_decoder_length, num_beams=num_beams, num_return_sequences=num_beams)
    results = [tokenizer.decode(t, skip_special_tokens=True) for t in output]

    print('Texts:')
    for i, summary in enumerate(results):
        print(i + 1, summary)


def reduce(hidden, index):
    _, n_length, _ = hidden.shape

    # Create selection mask
    mask = torch.arange(n_length, dtype=torch.float32) == index
    mask = mask.view(1, -1, 1)

    # Broadcast mask
    masked = torch.multiply(hidden, mask)

    # Reduce along 1st dimension
    summed = torch.sum(masked, 1)
    return torch.unsqueeze(summed, 1)


class NeuronEncoder(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids, attention_mask=attention_mask, return_dict=False)


class NeuronDecoder(torch.nn.Module):

    def __init__(self, model, max_length):
        super().__init__()
        self.weight = model.model.shared.weight.clone().detach()
        self.bias = model.final_logits_bias.clone().detach()
        self.decoder = model.model.decoder
        self.max_length = max_length

    def forward(self, input_ids, attention_mask, encoder_outputs, index):

        # Build a fixed sized causal mask for the padded decoder input ids
        mask = np.triu(np.ones((self.max_length, self.max_length)), 1)
        mask[mask == 1] = -np.inf
        causal_mask = torch.tensor(mask, dtype=torch.float)

        # Invoke the decoder
        hidden, = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_outputs,
            encoder_padding_mask=attention_mask,
            decoder_padding_mask=None,
            decoder_causal_mask=causal_mask,
            return_dict=False,
            use_cache=False,
        )

        # Reduce decoder outputs to the specified index (current iteration)
        hidden = reduce(hidden, index)

        # Compute final linear layer for token probabilities
        logits = F.linear(
            hidden,
            self.weight,
            bias=self.bias
        )
        return logits
    
class NeuronGeneration(PreTrainedModel, GenerationMixin):

    def trace(self, model, num_texts, num_beams, max_encoder_length, max_decoder_length):
        """
        Traces the encoder and decoder modules for use on Neuron.
        
        This function fixes the network to the given sizes. Once the model has been
        compiled to a given size, the inputs to these networks must always be of
        fixed size.
        
        Args:
            model (GenerationMixin): The transformer-type generator model to trace
            num_texts (int): The number of input texts to translate at once
            num_beams (int): The number of beams to computer per text
            max_encoder_length (int): The maximum number of encoder tokens
            max_encoder_length (int): The maximum number of decoder tokens
        """
        self.config.max_decoder_length = max_decoder_length

        # Trace the encoder
        inputs = (
            torch.ones((num_texts, max_encoder_length), dtype=torch.long),
            torch.ones((num_texts, max_encoder_length), dtype=torch.long),
        )
        encoder = NeuronEncoder(model)
        self.encoder = torch_neuron.trace(encoder, inputs)

        # Trace the decoder (with expanded inputs)
        batch_size = num_texts * num_beams
        inputs = (
            torch.ones((batch_size, max_decoder_length), dtype=torch.long),
            torch.ones((batch_size, max_encoder_length), dtype=torch.long),
            torch.ones((batch_size, max_encoder_length, model.config.d_model), dtype=torch.float),
            torch.tensor(0),
        )
        decoder = NeuronDecoder(model, max_decoder_length)
        self.decoder = torch_neuron.trace(decoder, inputs)

    # ------------------------------------------------------------------------
    # Beam Search Methods (Copied directly from transformers)
    # ------------------------------------------------------------------------

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            self._force_token_id_to_be_generated(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")
    
    # ------------------------------------------------------------------------
    # Encoder/Decoder Invocation 
    # ------------------------------------------------------------------------

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        encoder_outputs=None,
        attention_mask=None,
        **model_kwargs
    ):
        # Pad the inputs for Neuron
        current_length = decoder_input_ids.shape[1]
        pad_size = self.config.max_decoder_length - current_length
        return dict(
            input_ids=F.pad(decoder_input_ids, (0, pad_size)),
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs.last_hidden_state,
            current_length=torch.tensor(current_length - 1),
        )

    def get_encoder(self):
        """Helper to invoke the encoder and wrap the results in the expected structure"""
        def encode(input_ids, attention_mask, **kwargs):
            output, = self.encoder(input_ids, attention_mask)
            return BaseModelOutput(
                last_hidden_state=output,
            )
        return encode
        
    def __call__(self, input_ids, attention_mask, encoder_outputs, current_length, **kwargs):
        """Helper to invoke the decoder and wrap the results in the expected structure"""
        logits = self.decoder(input_ids, attention_mask, encoder_outputs, current_length)
        return Seq2SeqLMOutput(logits=logits)

    # ------------------------------------------------------------------------
    # Serialization 
    # ------------------------------------------------------------------------
        
    def save_pretrained(self, directory):
        if os.path.isfile(directory):
            print(f"Provided path ({directory}) should be a directory, not a file")
            return
        os.makedirs(directory, exist_ok=True)
        torch.jit.save(self.encoder, os.path.join(directory, 'encoder.pt'))
        torch.jit.save(self.decoder, os.path.join(directory, 'decoder.pt'))
        self.config.save_pretrained(directory)

    @classmethod
    def from_pretrained(cls, directory):
        config = MarianConfig.from_pretrained(directory)
        obj = cls(config)
        obj.encoder = torch.jit.load(os.path.join(directory, 'encoder.pt'))
        obj.decoder = torch.jit.load(os.path.join(directory, 'decoder.pt'))
        return obj
    
    @property
    def device(self):
        return torch.device('cpu')
    