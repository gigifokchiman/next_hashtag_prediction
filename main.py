# %%
import torch
import string
import igraph
from GraphTheory import *
from LDAPrediction import *
from top2vec import *
from awd_lstm import prediction_by_awd_lstm, load_awd_lstm_model
from electra import prediction_by_electra, load_electra_model


# load models here
xxx = load_awd_lstm_model()
xxx = load_electra_model()


electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')

def get_all_predictions(text_sentence, top_clean=5):
    """ redirect to the function to the right place """

    # can use async here to call functions from different files
    # and then return the results


    return {'graph_theory': graph_theory,
            'lda': lda,
            'top2vec': top2vec,
            'awd_lstm': awd_lstm,
            'electra': electra}


# ========================= GRAPH THEORY =================================
G = igraph.Graph.Read_GML('hashtag_with_community.gml')
Top2Vec_model = Top2Vec.load('Top2Vec_model')

