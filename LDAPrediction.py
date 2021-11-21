import pandas as pd
import numpy as np
import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import datapath

list_of_hashtags = pd.read_csv('train_list_of_hashtag.csv')['hashtag_editted'].tolist()
list_of_hashtags = [x.replace(' ','') for x in list_of_hashtags]
list_of_hashtags = [x.split(',') for x in list_of_hashtags]
hashtags_set = list(set([h for sublist in list_of_hashtags for h in sublist]))
popularity_df = pd.read_csv('popularity_df.csv')[['hashtag','count']]
popular_hashtags = popularity_df[popularity_df['count']>=300]['hashtag'].tolist()
common_dictionary = Dictionary(list_of_hashtags)
common_corpus = [common_dictionary.doc2bow(text) for text in list_of_hashtags]



def prediction_by_LDA(lda_model, test_tweet, k=5):
    test_corpus = common_dictionary.doc2bow(test_tweet)
    topic_distribution = lda_model[test_corpus]
    prob_distribution = list(map(list, zip(*topic_distribution)))[1]
    mean_prob = sum(prob_distribution)/len(prob_distribution)
    sd_prob = sum([(prob_distribution[i]-mean_prob)**2/len(prob_distribution)
                   for i in range(len(prob_distribution))])
    updated_topic_distribution = [lda_model[common_dictionary.doc2bow(test_tweet + h.split())]
                                  for h in popular_hashtags]

    se_list = []
    for i in range(len(popular_hashtags)):
        se = 0
        hashtag_topic = list(map(list, zip(*topic_distribution)))[0]
        updated_hashtag_topic = list(map(list, zip(*updated_topic_distribution[i])))[0]
        for t in range(30):
            if t in hashtag_topic:
                if t in updated_hashtag_topic:
                    if prob_distribution[hashtag_topic.index(t)]>mean_prob:
                        se += (topic_distribution[hashtag_topic.index(t)][1]-updated_topic_distribution[i][updated_hashtag_topic.index(t)][1])\
                              *(prob_distribution[hashtag_topic.index(t)]-mean_prob)/sd_prob
                else:
                    if prob_distribution[hashtag_topic.index(t)]>mean_prob:
                        se += (topic_distribution[hashtag_topic.index(t)][1]-0)\
                              *(prob_distribution[hashtag_topic.index(t)]-mean_prob)/sd_prob
        se_list.append(se)

    k_smallest_index = np.argsort(se_list)
    return [popular_hashtags[i] for i in k_smallest_index if popular_hashtags[i] not in test_tweet][:k]

if __name__ == '__main__':
    model = gensim.models.ldamodel.LdaModel.load(datapath("LDA_Model_for_next_hashtag_prediction"))
    print(prediction_by_LDA(model, ['TRUMP', 'BIDEN']))