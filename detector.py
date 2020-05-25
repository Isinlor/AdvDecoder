from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, RobertaForSequenceClassification

from BatchTextGenerationPipeline import BatchTextGenerationPipeline
from IsFakePipeline import IsFakePipeline

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

detector_tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
detector_model = RobertaForSequenceClassification.from_pretrained("roberta-base-openai-detector")

classifier = IsFakePipeline(model=detector_model, tokenizer=detector_tokenizer)
generator = BatchTextGenerationPipeline(model=model, tokenizer=tokenizer)

prompt = ""
for i in range(10):
    texts = generator.generate(prompt=prompt, generate_length=10, num_return_sequences=50, do_sample=True, top_p=0.95, no_repeat_ngram_size=3)
    prompt, classification = classifier.getLeastFake(texts)
    print("\n\n", classification, prompt)
