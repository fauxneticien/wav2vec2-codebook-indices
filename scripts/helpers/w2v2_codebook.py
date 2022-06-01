import torch
import torchaudio
import warnings

from dataclasses import dataclass
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Config,
    Wav2Vec2ForPreTrainingOutput,
    Wav2Vec2GumbelVectorQuantizer,
    Wav2Vec2ForPreTraining
)
from typing import Optional, Tuple, Union

# Extend class, modify forward() to return 'codebook_pairs'
class Wav2Vec2GumbelVectorQuantizerWithCodebookIndices(Wav2Vec2GumbelVectorQuantizer):

    def forward(self, hidden_states, mask_time_indices=None):
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # project to codevector dim
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        # take argmax in non-differentiable way
        # comptute hard codevector distribution (one hot)
        codevector_idx = hidden_states.argmax(dim=-1)
        codevector_probs = hidden_states.new_zeros(*hidden_states.shape).scatter_(
            -1, codevector_idx.view(-1, 1), 1.0
        )
        codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)
        codebook_pairs = torch.argmax(codevector_probs, dim=-1)

        perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # use probs to retrieve codevectors
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        return codevectors, perplexity, codebook_pairs

# Extend output class to have place for 'codebook_pairs'
@dataclass
class Wav2Vec2ForPreTrainingOutputWithCodebookIndices(Wav2Vec2ForPreTrainingOutput):
    codebook_pairs: Optional[Tuple[torch.FloatTensor]] = None

# Extend class, modify forward() to return 'codebook_pairs'
class Wav2Vec2ForPreTrainingWithCodebookIndices(Wav2Vec2ForPreTraining):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)

        self.quantizer = Wav2Vec2GumbelVectorQuantizerWithCodebookIndices(config)

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2ForPreTrainingOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.project_hid(outputs[0])

        # 2. quantize all (unmasked) extracted features and project to final vq dim
        extract_features = self.dropout_features(outputs[1])

        # Modified quantizer provides 'codebook_pairs' as third item in returned tuple
        quantized_features, codevector_perplexity, codebook_pairs = self.quantizer(
            extract_features, mask_time_indices=mask_time_indices
        )
        quantized_features = self.project_q(quantized_features)

        loss = contrastive_loss = diversity_loss = None

        if not return_dict:
            if loss is not None:
                return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

        return Wav2Vec2ForPreTrainingOutputWithCodebookIndices(
            loss=loss,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
            codebook_pairs=codebook_pairs
        )

def get_codebook_indices(wav_file):
    # Catch HF gradient_checkpointing deprecation warning
    warnings.filterwarnings(action='ignore', category=UserWarning, module=r'.*configuration_utils')

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForPreTrainingWithCodebookIndices.from_pretrained("facebook/wav2vec2-base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    wav, _ = torchaudio.load(wav_file)
    wav = wav.squeeze()

    input_values = feature_extractor(wav, sampling_rate=16_000, return_tensors="pt").input_values  # Batch size 1

    with torch.no_grad():
        outputs = model(input_values.to(device), output_hidden_states=True, return_dict=True)

    return outputs.codebook_pairs.cpu().numpy()
