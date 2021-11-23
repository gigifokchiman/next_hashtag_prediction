import pandas as pd
import fastai
from fastai.text import *
from fastai.metrics import *
from fastai.text.all import *


def load_awd_lstm_model():
    dls_lm = torch.load("res/dls_lm.pkl")

    learn = language_model_learner(
        dls_lm, AWD_LSTM, drop_mult=0.3,
        metrics=[accuracy, Perplexity()]).to_fp16()

    learn.load_encoder('res/finetuned_full')

    with open("res/hashtag_location_awd_lstm.pkl", "rb") as f:
        hashtag_location = pickle.load(f)

    return learn, hashtag_location


def prediction_by_awd_lstm(text, learn, hashtag_location, number_of_predictions=5):

    no_unk=True

    learn.model.reset()
    # idxs = idxs_all = learn.dls.test_dl([text]).items[0].to(learn.dls.device)

    idxs = idxs_all = Numericalize(learn.dls.vocab)

    learn.model.reset()

    num = Numericalize(learn.dls.vocab)
    idxs = idxs_all = num(text);

    if no_unk: unk_idx = learn.dls.vocab.index(UNK)

    preds, _ = learn.get_preds(dl=[(idxs[None],)])

    res = preds[0][-1].numpy()

    prediction = list(zip(learn.dls.vocab, res))

    prediction_hashtag = [j for i, j in enumerate(prediction) if i in hashtag_location]

    prediction_hashtag.sort(key=lambda x: x[1], reverse=True)
    final_result = prediction_hashtag[0:number_of_predictions]

    return "\n".join(final_result)



