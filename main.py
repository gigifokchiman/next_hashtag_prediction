# %%
import torch
import string
import igraph
from GraphTheory import prediction_by_graph_theory
from LDAPrediction import *
from top2vec import Top2Vec

from GraphTheory import prediction_by_graph_theory
from Top2VecHashtagPrediction import prediction_by_top2vec
from awd_lstm import prediction_by_awd_lstm, load_awd_lstm_model
from electra import prediction_by_electra, load_electra_model
import re

# load models here
# awd_lstm_tokenizer, awd_lstm_model = load_awd_lstm_model()
electra_tokenizer, electra_model, for_prediction = load_electra_model()
top2vec_model = Top2Vec.load('Top2Vec_model')

graph_theory_model = igraph.Graph.Read_GML('hashtag_with_community.gml')


def get_all_predictions(text_sentence, number_of_predictions=5):
    """ redirect to the function to the right place """

    # can use async here to call functions from different files
    # and then return the results

    # cannot line break here
    input_hashtags = re.findall(r"#(?![0-9]+)([a-zA-Z0-9_]+)(\b)", "I #yes try #1go #ohhh")
    input_hashtags = [x[0].lower() for x in input_hashtags]


    electra = prediction_by_electra(text_sentence, electra_tokenizer, electra_model, for_prediction,
                                          number_of_predictions)
    awd_lstm = ""
    # awd_lstm = prediction_by_awd_lstm(text_sentence, awd_lstm_tokenizer, awd_lstm_model, number_of_predictions)
    top2vec = prediction_by_top2vec(top2vec_model, input_hashtags, number_of_predictions)

    if len(input_hashtags) > 2:
        lda = ""
        # lda = prediction_by_LDA(lda_model, input_hashtags, number_of_predictions)
        graph_theory = prediction_by_graph_theory(graph_theory_model, input_hashtags, number_of_predictions)
    else:
        lda = ""
        graph_theory = ""

    print(graph_theory)
    print(lda)
    print(top2vec)
    print(awd_lstm)
    print(electra)

    return {'graph_theory': graph_theory,
            'lda': lda,
            'top2vec': top2vec,
            'awd_lstm': awd_lstm,
            'electra': electra}


# ========================= GRAPH THEORY =================================
G = igraph.Graph.Read_GML('hashtag_with_community.gml')
Top2Vec_model = Top2Vec.load('Top2Vec_model')

