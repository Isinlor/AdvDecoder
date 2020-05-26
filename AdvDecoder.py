def decode(prompt, generate_length, step, sequences, generator, classifier):
    for i in range(0, generate_length, step):
        texts = generator.generate(prompt=prompt, generate_length=10, num_return_sequences=sequences, do_sample=True, top_p=0.99, no_repeat_ngram_size=3)
        prompt, _ = classifier.getLeastFake(texts)
    return prompt

def batch_decode(batch, prompt, generate_length, step, sequences, generator, classifier):
    return [decode(prompt, generate_length, step, sequences, generator, classifier) for _ in range(batch)]