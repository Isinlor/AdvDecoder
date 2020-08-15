from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, RobertaForSequenceClassification

from AdvDecoder import decode, batch_decode
from BatchTextGenerationPipeline import BatchTextGenerationPipeline
from IsFakePipeline import IsFakePipeline

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

detector_tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
detector_model = RobertaForSequenceClassification.from_pretrained("roberta-base-openai-detector")

model.to(0)
detector_model.to(0)

classifier = IsFakePipeline(model=detector_model, tokenizer=detector_tokenizer, device=0)
generator = BatchTextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)

print("\n\n ============== Top-K Decoder\n\n")
print("\n--------------------\n".join(generator.generate("", top_k=150, generate_length=100, num_return_sequences=10, do_sample=True, no_repeat_ngram_size=3)))

print("\n\n ============== Adversarial Decoder\n\n")
print("\n--------------------\n".join(batch_decode(10, prompt="", generate_length=100, step=10, sequences=10, generator=generator, classifier=classifier)))