# %%
import torch
import string
import igraph
from GraphTheory import *
from LDAPrediction import *
from top2vec import *
from awd_lstm import *
from electra import *

# load models here


def get_all_predictions(text_sentence, top_clean=5):
    """ redirect to the function to the right place """

    # can use async here

    # # ========================= BERT =================================
    # print(text_sentence)
    # input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    # with torch.no_grad():
    #     predict = bert_model(input_ids)[0]
    # bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    #
    # # ========================= XLNET LARGE =================================
    # input_ids, mask_idx = encode(xlnet_tokenizer, text_sentence, False)
    # perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
    # perm_mask[:, :, mask_idx] = 1.0  # Previous tokens don't see last token
    # target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
    # target_mapping[0, 0, mask_idx] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)
    #
    # with torch.no_grad():
    #     predict = xlnet_model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)[0]
    # xlnet = decode(xlnet_tokenizer, predict[0, 0, :].topk(top_k).indices.tolist(), top_clean)
    #
    # # ========================= XLM ROBERTA BASE =================================
    # input_ids, mask_idx = encode(xlmroberta_tokenizer, text_sentence, add_special_tokens=True)
    # with torch.no_grad():
    #     predict = xlmroberta_model(input_ids)[0]
    # xlm = decode(xlmroberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    #
    # # ========================= BART =================================
    # input_ids, mask_idx = encode(bart_tokenizer, text_sentence, add_special_tokens=True)
    # with torch.no_grad():
    #     predict = bart_model(input_ids)[0]
    # bart = decode(bart_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    #
    # # ========================= ELECTRA =================================
    # input_ids, mask_idx = encode(electra_tokenizer, text_sentence, add_special_tokens=True)
    # with torch.no_grad():
    #     predict = electra_model(input_ids)[0]
    # electra = decode(electra_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    #
    # # ========================= ROBERTA =================================
    # input_ids, mask_idx = encode(roberta_tokenizer, text_sentence, add_special_tokens=True)
    # with torch.no_grad():
    #     predict = roberta_model(input_ids)[0]
    # roberta = decode(roberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    return {'graph_theory': graph_theory,
            'lda': lda,
            'top2vec': top2vec,
            'awd_lstm': awd_lstm,
            'electra': electra}


# ========================= GRAPH THEORY =================================
G = igraph.Graph.Read_GML('hashtag_with_community.gml')
Top2Vec_model = Top2Vec.load('Top2Vec_model')

