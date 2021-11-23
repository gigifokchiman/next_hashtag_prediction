from top2vec import Top2Vec


def prediction_by_top2vec(top2vec_model, input_hashtags, number_of_predictions=5):
    valid_hashtags = [hashtag for hashtag in input_hashtags if hashtag in top2vec_model.model.wv.vocab]
    if len(valid_hashtags) > 0:
        word, word_scores = top2vec_model.similar_words(keywords=valid_hashtags, keywords_neg=[],
                                                        num_words=number_of_predictions)
        return "\n".join([w.upper() for w in word])
    else:
        return 'nil'


if __name__ == '__main__':
    model = Top2Vec.load('Top2Vec_model')
    print(prediction_by_top2vec(model, ['yes', 'biden']))
