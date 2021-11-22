import pandas as pd
import asyncio

def load_electra_model():
    electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
    electra_model = ElectraModel.from_pretrained('google/electra-small-generator')

    return electra_tokenizer, electra_model


async def prediction_by_electra(graph, nodes, no_of_predictions=5, epsilon=10 ** -5):


    return

