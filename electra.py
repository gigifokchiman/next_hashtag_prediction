import pickle

import numpy as np
import torch
import torch.nn as nn
from transformers import ElectraTokenizer, ElectraForMaskedLM


def load_electra_model():
    electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
    electra_model = ElectraForMaskedLM.from_pretrained('res/checkpoint-1-epoch-1-g').eval()

    # with open('filename.pickle', 'wb') as handle:
    #     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('res/hashtag_tokenised_1000.pkl', 'rb') as handle:
        hashtag_tokens = pickle.load(handle)

    return electra_tokenizer, electra_model, hashtag_tokens

def prediction_by_electra(text_sentence,
                          electra_tokenizer,
                          electra_model,
                          for_prediction,
                          number_of_predictions=5,
                          use_cuda=False):

    text_sentence = text_sentence.lower()
    text_sentence_split = text_sentence.split()
    text_sentence = ' '.join(text_sentence_split)
    input_ids_raw = electra_tokenizer.encode(text_sentence, add_special_tokens=True)[:-1]
    mask_token_id = electra_tokenizer.encode(electra_tokenizer.mask_token, add_special_tokens=False)[0]

    sm = nn.Softmax(dim=1)

    prediction_list = list()

    text_sentence_split_set = set(text_sentence_split)
    for_prediction = [i for i in for_prediction if i[0].lower() not in text_sentence_split_set]

    # ==================================================================
    # Pad to same length
    max_length = max([f[3] for f in for_prediction])

    for m in range(max_length):
        # print(m)
        filtered_meta = [(f[0], f[1]) for f in for_prediction if f[3] == m + 1]
        filtered = [f[2] for f in for_prediction if f[3] == m + 1]
        filtered_arr = np.array(filtered)

        if not filtered:
            continue

        prob = np.ones(shape=(len(filtered),))

        for n in range(m+1):
            # print(n)
            filtered_concat = np.array([
                np.array([*input_ids_raw, *current_id[0:n]] + [mask_token_id])
                for current_id in filtered
            ])
            input_ids = torch.tensor(filtered_concat)

            # print(input_ids.shape)
            with torch.no_grad():
                predict = electra_model(input_ids)[0]
            res_raw = predict[:, -1, :]
            res = sm(res_raw).detach().cpu().numpy()

            # filtered_arr[:, n]
            #
            # take = [
            #   res[idx, filtered_arr]
            #   for idx in range(res.shape[0])
            # ]
            # print(res.shape)
            # print(filtered_arr[:, n].shape)
            take = np.take_along_axis(res, np.expand_dims(filtered_arr[:, n], 1), 1)
            # print(take.shape)
            new_prob = np.squeeze(take, 1)
            # print(f"m: {m}, n:{n}, mean: {new_prob.mean()}; std{new_prob.std()}")
            # print(new_prob.std())
            prob *= new_prob/new_prob.mean()

        # result = list(tuple(zip(filtered, prob)))
        result = [(i[0], i[1], filtered[k], prob[k]) for k, i in enumerate(filtered_meta)]
        prediction_list = [*prediction_list, *result]

    # prediction_list.sort(key=lambda x: x[3], reverse=True)

    # final_result = [i[0] for i in prediction_list[:number_of_predictions]]
    #
    # return prediction_list

    prediction_list.sort(key=lambda x: x[3], reverse=True)

    final_result = ["#" + i[0] for i in prediction_list[:number_of_predictions]]

    return "\n".join(final_result).upper()

