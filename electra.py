import torch
from transformers import ElectraTokenizer, ElectraForMaskedLM
import numpy as np
import torch.nn as nn
import pickle

def load_electra_model():
    electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
    electra_model = ElectraForMaskedLM.from_pretrained('res/checkpoint-1-epoch-1-g').eval()

    # with open('filename.pickle', 'wb') as handle:
    #     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('res/hashtag_tokenised_1000.pkl', 'rb') as handle:
        hashtag_tokens = pickle.load(handle)

    return electra_tokenizer, electra_model, hashtag_tokens


async def prediction_by_electra(text_sentence,
                                electra_tokenizer,
                                electra_model,
                                for_prediction,
                                number_of_predictions=5,
                                use_cuda=False):
    text_sentence = ' '.join(text_sentence.split())

    input_ids_raw = list(torch.tensor([electra_tokenizer.encode(text_sentence,
                                                                add_special_tokens=True)]).numpy()[0][:-1])

    mask_token_id = electra_tokenizer.encode(electra_tokenizer.mask_token)[1]

    # mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]

    prediction_list = list()

    for i in for_prediction:

        # text_sentence += ' <mask>'
        # input_ids, mask_idx = encode(electra_tokenizer, text_sentence, add_special_tokens=True)
        current_id = i[2]

        # input_ids = np.concatenate((np.array(input_ids_raw)[0, :-1]))

        prob = 1

        for j, k in enumerate(current_id):
            # input_ids = torch.tensor([np.concatenate(([input_ids_raw[0], current_id[0:j]]))])

            input_ids = torch.tensor([np.array([*input_ids_raw, *current_id[0:j]] + [mask_token_id])])

            with torch.no_grad():
                if use_cuda:
                    predict = electra_model(input_ids.cuda())[0]
                else:
                    predict = electra_model(input_ids)[0]

                res_raw = predict[0, len(input_ids_raw) + j, :]
                m = nn.Softmax(dim=0)
                res = m(res_raw)
                prob *= res.cpu().numpy()[current_id[j]]

        # adj = len(current_id) -1
        # if adj > 0:
        #   prob = np.power(prob, (1/adj))
        # prob

        prediction_list.append((i[0], i[1], i[2], prob))

    prediction_list.sort(key=lambda x: x[3], reverse=True)

    return prediction_list[:number_of_predictions]



