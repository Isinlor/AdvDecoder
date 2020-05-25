from typing import List, Tuple

import numpy
from transformers import Pipeline

class IsFakePipeline(Pipeline):

    def __call__(self, *args, **kwargs):
        outputs = super().__call__(*args, **kwargs)
        scores = numpy.exp(outputs) / numpy.exp(outputs).sum(-1, keepdims=True)
        return [item[0] for item in scores]
    
    def getLeastFake(self, texts: List[str]) -> Tuple[List[str], float]:
        classifications = numpy.array(self.predict(texts))
        return texts[classifications.argmin()], classifications.min(initial=1)