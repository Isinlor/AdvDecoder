from AdvTranslator import AdvTranslator

src = 'en'  # source language
trg = 'fi'  # target language
sampling = 10

translator = AdvTranslator(src, trg)

source_texts = [
    "A security breach in the administration system of Twitter results in many prominent accounts promoting a bitcoin scam.",
    "At least sixteen people are killed in border clashes between Armenian and Azerbaijani armed forces.",
    "A Boyar Wedding Feast is an oil-on-canvas painting created by Russian artist Konstantin Makovsky in 1883.",
    "She was the winner of the Miss World 2000 pageant, and is one of India's highest-paid and most popular entertainers.",
    "Coronavirus disease 2019 (COVID-19) is an infectious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2).",
    "The George Floyd protests are an ongoing series of protests and civil unrest which began in Minneapolis in the United States on May 26, 2020."
]

translations = [translator.translate(text, sampling) for text in source_texts]

for source, translation in zip(source_texts, translations):
    print(
        "\nSource:\t\t\t", source,
        "\nTranslation:\t", translator.forward_translate(source, 1)[0],
        "\nAdvTranslation:\t", translation
    )
