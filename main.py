# %%
import torch
import string
import igraph
from GraphTheory import prediction_by_graph_theory
from LDAPrediction import *
import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

from Top2VecHashtagPrediction import *
from awd_lstm import prediction_by_awd_lstm, load_awd_lstm_model
from electra import prediction_by_electra, load_electra_model
import re
import pandas as pd

# load models here
# Graph Theory
graph_theory_model = igraph.Graph.Read_GML('hashtag_with_community.gml')
# LDA
list_of_hashtags = pd.read_csv('train_list_of_hashtag.csv')['hashtag_editted'].tolist()
list_of_hashtags = [x.replace(' ', '') for x in list_of_hashtags]
list_of_hashtags = [x.split(',') for x in list_of_hashtags]
hashtags_set = list(set([h for sublist in list_of_hashtags for h in sublist]))
popularity_df = pd.read_csv('popularity_df.csv')[['hashtag', 'count']]
popular_hashtags = popularity_df[popularity_df['count'] >= 300]['hashtag'].tolist()
common_dictionary = Dictionary(list_of_hashtags)
common_corpus = [common_dictionary.doc2bow(text) for text in list_of_hashtags]
lda_model = gensim.models.ldamodel.LdaModel.load("LDA_Model_for_next_hashtag_prediction")
# Top2Vec
top2vec_model = Top2Vec.load('Top2Vec_model')
# AWD-LSTM
# awd_lstm_tokenizer, awd_lstm_model = load_awd_lstm_model()
# Electra
electra_tokenizer, electra_model, for_prediction = load_electra_model()


def get_all_predictions(text_sentence, number_of_predictions=5):
    """ redirect to the function to the right place """

    # can use async here to call functions from different files
    # and then return the results

    # cannot line break here
    # input_hashtags = re.findall(r"#(?![0-9]+)([a-zA-Z0-9_]+)(\b)", "I #yes try #1go #ohhh")
    input_hashtags = re.findall(r"#([a-zA-Z0-9_]+)(\b)", text_sentence)

    input_hashtags = [x[0].lower() for x in input_hashtags]

    print(input_hashtags)

    text_sentence += " #"
    text_sentence = text_sentence.replace("#", "# ")
    text_sentence = text_sentence.replace("  ", " ")

    print(text_sentence)

    electra = prediction_by_electra(text_sentence, electra_tokenizer, electra_model, for_prediction,
                                    number_of_predictions)
    # awd_lstm = ""
    # awd_lstm = prediction_by_electra(text_sentence, electra_tokenizer, electra_model, for_prediction,
    #                                  number_of_predictions * 2)
    # awd_lstm = '\n'.join(awd_lstm.split('\n')[::2])
    awd_lstm = "only available on GPU"
    # awd_lstm = prediction_by_awd_lstm(text_sentence, awd_lstm_tokenizer, awd_lstm_model, number_of_predictions)
    top2vec = prediction_by_top2vec_tweet(top2vec_model, text_sentence, number_of_predictions)
    if len(input_hashtags) >= 1:
        lda = prediction_by_LDA(lda_model, input_hashtags,
                                common_dictionary, popular_hashtags,
                                hashtags_set, number_of_predictions)

        graph_theory = prediction_by_graph_theory(graph_theory_model, input_hashtags, number_of_predictions)
    else:
        lda = ""
        graph_theory = ""

    return {'graph_theory': graph_theory,
            'lda': lda,
            'top2vec': top2vec,
            'awd_lstm': awd_lstm,
            'electra': electra}


