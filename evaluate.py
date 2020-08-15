from time import sleep

from AdvTranslator import AdvTranslator

source = open("./data/eng-fin/test.src")
target = open("./data/eng-fin/test.trg")
output = open("./outputs/adv-en-fi-10.txt", mode="a")

src = 'en'  # source language
trg = 'fi'  # target language
sampling = 10

translator = AdvTranslator(src, trg)

for source_sample, target_sample in zip(source, target):
    translation = translator.translate(source_sample, sampling)
    print(translation, end="\n", file=output, flush=True)
    print(translation)