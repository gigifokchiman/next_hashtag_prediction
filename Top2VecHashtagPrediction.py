from top2vec import Top2Vec
import re


def prediction_by_top2vec(top2vec_model, input_hashtags, number_of_predictions=5):
    valid_hashtags = [hashtag for hashtag in input_hashtags if hashtag in top2vec_model.model.wv.vocab]
    if len(valid_hashtags) > 0:
        word, word_scores = top2vec_model.similar_words(keywords=valid_hashtags,
                                                        keywords_neg=[], num_words=number_of_predictions)
        return word
    else:
        return 'nil'


def prediction_by_top2vec_tweet(top2vec_model, input_tweet, number_of_predictions=5):
    similar_tweets, doc_scores, doc_ids = top2vec_model.query_documents(query=input_tweet,
                                                                        num_docs=20)
    output_hashtags = [re.findall(r"#([a-zA-Z0-9_]+)(\b)", tweet) for tweet in similar_tweets]
    output_hashtags = [y[0].upper() for x in output_hashtags for y in x]

    input_hashtags = re.findall(r"#([a-zA-Z0-9_]+)(\b)", input_tweet)
    input_hashtags = list(set([x[0].upper() for x in input_hashtags]))

    output = []
    for hashtag in output_hashtags:
        if hashtag not in output and hashtag not in input_hashtags:
            output.append(hashtag)
            if len(output) == number_of_predictions:
                break

    return "\n".join(["#" + hashtag for hashtag in output])


if __name__ == '__main__':
    model = Top2Vec.load('Top2Vec_model')
    print(prediction_by_top2vec_tweet(model, "Proof positive that #Biden is, in fact, senile"))
