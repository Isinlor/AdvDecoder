from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, RobertaForSequenceClassification

from AdvDecoder import decode
from BatchTextGenerationPipeline import BatchTextGenerationPipeline
from IsFakePipeline import IsFakePipeline

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

detector_tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
detector_model = RobertaForSequenceClassification.from_pretrained("roberta-base-openai-detector")

classifier = IsFakePipeline(model=detector_model, tokenizer=detector_tokenizer)
generator = BatchTextGenerationPipeline(model=model, tokenizer=tokenizer)

print(decode(prompt="", generate_length=50, step=10, sequences=3, generator=generator, classifier=classifier))