from typing import Optional, Iterable, List
import torch
from transformers import Pipeline, PreTrainedModel


class BatchTextGenerationPipeline(Pipeline):
    __doc__ = PreTrainedModel.generate.__doc__
    def __call__(self, prompt: str, generate_length: Optional[int] = None, *args, **kwargs) -> List[str]:

        if not prompt:
            input_ids = None
            input_length = 0
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            input_length = input_ids.shape[1]
            input_ids.to(self.device)

        if not generate_length:
            outputs = self.model.generate(input_ids, pad_token_id=self.model.config.eos_token_id, *args, **kwargs)
        else:
            expected_length = input_length + generate_length
            outputs = self.model.generate(input_ids, *args, min_length=expected_length, max_length=expected_length, pad_token_id=self.model.config.eos_token_id, **kwargs)

        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    def generate(
        self,
        prompt: str,
        generate_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_specific_kwargs
    ) -> List[str]:
        return self.__call__(
            prompt = prompt,
            generate_length = generate_length,
            do_sample = do_sample,
            early_stopping = early_stopping,
            num_beams = num_beams,
            temperature = temperature,
            top_k = top_k,
            top_p = top_p,
            repetition_penalty = repetition_penalty,
            bad_words_ids = bad_words_ids,
            bos_token_id = bos_token_id,
            eos_token_id = eos_token_id,
            length_penalty = length_penalty,
            no_repeat_ngram_size = no_repeat_ngram_size,
            num_return_sequences = num_return_sequences,
            attention_mask = attention_mask,
            decoder_start_token_id = decoder_start_token_id,
            use_cache = use_cache,
            **model_specific_kwargs
        )
