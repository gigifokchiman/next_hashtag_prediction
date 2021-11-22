# %%
import torch
import string
import igraph
from GraphTheory import *
from LDAPrediction import *
from Top2VecHashtagPrediction import *
from awd_lstm import prediction_by_awd_lstm, load_awd_lstm_model
from electra import prediction_by_electra, load_electra_model
import asyncio


# load models here
awd_lstm_tokenizer, awd_lstm_model = load_awd_lstm_model()
electra_tokenizer, electra_model = load_electra_model()



# async def say_after(delay, what):
#     await asyncio.sleep(delay)
#     print(what)
#
# async def main():
#     print(f"started at {time.strftime('%X')}")
#
#     await say_after(1, 'hello')
#     await say_after(2, 'world')
#
#     print(f"finished at {time.strftime('%X')}")
#
# asyncio.run(main())
async def test():
    return True

async def get_all_predictions(text_sentence, number_of_predictions=5):
    """ redirect to the function to the right place """

    # can use async here to call functions from different files
    # and then return the results

    electra = await prediction_by_electra(text_sentence, electra_tokenizer, electra_model, number_of_predictions)
    awd_lstm = await prediction_by_awd_lstm(text_sentence, awd_lstm_tokenizer, awd_lstm_model, number_of_predictions)
    top2vec = await prediction_by_top2vec(top2vec_model, input_hashtags, number_of_predictions, number_of_predictions)
    lda = await prediction_by_LDA(lda_model, input_hashtags, number_of_predictions)
    graph_theory = await prediction_by_graph_theory(graph_theory_model, input_hashtags, number_of_predictions)


    return {'graph_theory': graph_theory,
            'lda': lda,
            'top2vec': top2vec,
            'awd_lstm': awd_lstm,
            'electra': electra}


# ========================= GRAPH THEORY =================================
G = igraph.Graph.Read_GML('hashtag_with_community.gml')
Top2Vec_model = Top2Vec.load('Top2Vec_model')

