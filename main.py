# %%
import torch
import string
import igraph
from GraphTheory import prediction_by_graph_theory
from LDAPrediction import *
from top2vec import Top2Vec
from Top2VecHashtagPrediction import prediction_by_top2vec
from awd_lstm import prediction_by_awd_lstm, load_awd_lstm_model
from electra import prediction_by_electra, load_electra_model
import re

# load models here
awd_lstm_tokenizer, awd_lstm_model = load_awd_lstm_model()
electra_tokenizer, electra_model = load_electra_model()
top2vec_model = Top2Vec.load('Top2Vec_model')
lda_model = gensim.models.ldamodel.LdaModel.load(datapath("LDA_Model_for_next_hashtag_prediction"))
graph_theory_model = igraph.Graph.Read_GML('hashtag_with_community.gml')


async def get_all_predictions(text_sentence, number_of_predictions=5):
    """ redirect to the function to the right place """

    # can use async here to call functions from different files
    # and then return the results

    # cannot line break here
    input_hashtags = re.findall(r"#(?![0-9]+)([a-zA-Z0-9_]+)(\b)", "I #yes try #1go #ohhh")
    input_hashtags = [x[0].lower() for x in input_hashtags]

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

