import numpy as np


def prediction_by_LDA(lda_model, hashtags, dictionary, popular_hashtags, hashtags_set, k=5):
    hashtags = [hashtag.upper() for hashtag in hashtags]
    if len(set(hashtags).intersection(set(hashtags_set))) == 0:
        return 'nil'
    else:
        hashtags = list(set(hashtags).intersection(set(hashtags_set)))
    test_corpus = dictionary.doc2bow(hashtags)
    topic_distribution = lda_model[test_corpus]
    temp_list = list(map(list, zip(*topic_distribution)))
    prob_distribution = temp_list[1]
    mean_prob = np.mean(prob_distribution)
    sd_prob = np.std(prob_distribution)
    updated_topic_distribution = [lda_model[dictionary.doc2bow(hashtags + h.split())]
                                  for h in popular_hashtags]

    se_list = [0] * len(popular_hashtags)
    for i in range(len(popular_hashtags)):
        hashtag_topic = temp_list[0]
        updated_hashtag_topic = list(map(list, zip(*updated_topic_distribution[i])))[0]
        for t in range(30):
            if t in hashtag_topic:
                index = hashtag_topic.index(t)
                if t in updated_hashtag_topic:
                    if prob_distribution[index] > mean_prob:
                        se_list[i] += (topic_distribution[index][1]-updated_topic_distribution[i][updated_hashtag_topic.index(t)][1])\
                                      *(prob_distribution[index]-mean_prob)/sd_prob
                else:
                    if prob_distribution[index] > mean_prob:
                        se_list[i] += (topic_distribution[index][1]-0)\
                                      *(prob_distribution[index]-mean_prob)/sd_prob

    k_smallest_index = np.argsort(se_list)
    return "\n".join([popular_hashtags[i] for i in k_smallest_index if popular_hashtags[i] not in hashtags][:k])


if __name__ == '__main__':
    import gensim
    from gensim.corpora import Dictionary
    from gensim.models.ldamodel import LdaModel
    import pandas as pd

    list_of_hashtags = pd.read_csv('train_list_of_hashtag.csv')['hashtag_editted'].tolist()
    list_of_hashtags = [x.replace(' ', '') for x in list_of_hashtags]
    list_of_hashtags = [x.split(',') for x in list_of_hashtags]
    hashtags_set = list(set([h for sublist in list_of_hashtags for h in sublist]))
    popularity_df = pd.read_csv('popularity_df.csv')[['hashtag', 'count']]
    popular_hashtags = popularity_df[popularity_df['count'] >= 300]['hashtag'].tolist()
    common_dictionary = Dictionary(list_of_hashtags)
    common_corpus = [common_dictionary.doc2bow(text) for text in list_of_hashtags]

    model = gensim.models.ldamodel.LdaModel.load("LDA_Model_for_next_hashtag_prediction")
    print(prediction_by_LDA(model, ['1234556', 'MAC12HINELEARNING'],
                            common_dictionary, popular_hashtags,
                            hashtags_set))