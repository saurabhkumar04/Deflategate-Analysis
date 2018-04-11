#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:     Social Media Mining Project
#
# Author:      Saurabh kumar
# Format:      PEP 8
# Created:     12/09/2016
# Copyright:   (c) Saurabh kumar 2016
# Licence:     <your licence>
# Description:  This code performs Sentiment Analysis on #Deflategate twitter data.
#-------------------------------------------------------------------------------


#Package Imports
import nltk
import re
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
from random import shuffle
from wordcloud import WordCloud


#Initializing for short usage
stopset = list(set(stopwords.words('english')))
ps = PorterStemmer()
lemm = WordNetLemmatizer()


#Function for basic preprocessing of the tweets
def preprocess(y):

    #The following code removes links, special characters etc.
    y = re.sub(r'http\S+', ' ', y)
    y = re.sub(r'[^\x00-\x7f]', r' ', y)
    y = re.sub(r'[^a-zA-Z]+', r' ', y)
    y = re.sub(r'\b\w{1,2}\b', r' ', y)

    return (y)


#Function for advanced preprocessing of the tweets; Removing Stopwords, tokenizing and stemming
def advprocess(y):

    #tokenizing using word_tokenize function from nltk package
    y = nltk.word_tokenize(y)

    #The following code make use of list comprehension for compact view
    y = [word for word in y if word not in stopset]
    y = [lemm.lemmatize(word) for word in y]
    #y = [ps.stem(word) for word in y]

    #Returning a dictionary as it is a suitable datatype for Naive Bayes Classifier
    return dict([(word, True) for word in y])


#Function to be used as a substitute for above advprocess()
#Returns a dictionary to be used for Naive Bayes Classifier
def word_feats(words):

    return [word for word in words.split() if word not in stopset]


#Function to classify sentiment for tweets in the test_set.
#Uses Naive Bayes Classifier
def classify_naivebayes(train_set, test_set):

    #Training Naive Bayes
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    #Classifying the test set
    output = [classifier.classify(text) for text in test_set]

    return output


#Function to find the main context of data
def find_contexts(test_data):

    #Categorizing contexts in four groups based on specific conditions
    if "organization" in test_data.lower() or "management" in test_data.lower():
        return ("Management")

    elif "tom" in test_data.lower() or "brady" in test_data.lower():
        return ("Tom Brady")

    elif "patriots" in test_data.lower():
        return ("Patriots")

    else:
        return ("Other")


#Function for finding insights in the tweets and comparison statistics
def analysis (contexts, sentiments):

    #initializing different type of sentiments - They store the value of number of tweets belonging to a specific category
    irrelevant_sentiments = 0
    positive_sentiments = 0
    negative_sentiments = 0
    neutral_sentiments = 0

    #initializing intermediate values
    mgt_positive = 0
    mgt_negative = 0
    tom_positive = 0
    tom_negative = 0
    pat_positive = 0
    pat_negative = 0

    #For calculating different values
    for i in range(len(contexts)):

        #Calculating the number of tweets in each sentiment category
        if sentiments[i] == 'positive':
            positive_sentiments+=1

            if contexts[i] == 'Management':
                mgt_positive += 1

            elif contexts[i] == 'Tom Brady':
                tom_positive += 1

            elif contexts[i] == 'Patriots':
                pat_positive += 1


        elif sentiments[i] == 'negative':
            negative_sentiments+=1

            if contexts[i] == 'Management':
                mgt_negative+=1

            elif contexts[i] == 'Tom Brady':
                tom_negative+=1

            elif contexts[i] == 'Patriots':
                pat_negative+=1

        elif sentiments[i] == 'neutral':
            neutral_sentiments+=1

        elif sentiments[i] == 'irrelevant':
            irrelevant_sentiments+=1

        else:
            print ("Discrepancy")

    #Creating a dictionary for different sentiment statistics
    sentiment_stat = {'irrelevant': irrelevant_sentiments, 'neutral': neutral_sentiments, 'positive': positive_sentiments, 'negative': negative_sentiments}

    #Creating a dictionary for different blame statistics
    blame_stat = {'Management': mgt_negative,'Tom Brady': tom_negative,'Patriots':pat_negative}

    #Creating a dictionary for different support statistics
    support_stat = {'Management': mgt_positive,'Tom Brady': tom_positive,'Patriots': pat_positive}

    #Stats to compare blame and support for Tom Brady
    tom_stat = {'Blame':tom_negative,'Support':tom_positive}

    # Stats to compare blame and support for Patriots
    pat_stat = {'Blame':pat_negative,'Support':pat_positive}

    #Calling graph function to plot histograms for various statistics
    graphs(sentiment_stat, blame_stat, support_stat, tom_stat, pat_stat)


#Function to create graphs for the different findings
def graphs(sentiment_stat, blame_stat, support_stat, tom_stat, pat_stat):

    #Create Sentiment Statistics Histogram
    jet = plt.get_cmap('jet')
    plt.bar(range(len(sentiment_stat)), sentiment_stat.values(), align='center', color=jet(np.linspace(0, 1.0, len(sentiment_stat))))
    plt.xticks(range(len(sentiment_stat)), list(sentiment_stat.keys()))
    plt.xlabel('Sentiments')
    plt.ylabel('Number of Tweets')
    plt.title('Sentiment Statistics')
    plt.show()

    #Comparing blame and support statistics for Tom Brady
    plt.bar(range(len(tom_stat)), tom_stat.values(), align='center',color=jet(np.linspace(0, 1.0, len(tom_stat))))
    plt.xticks(range(len(tom_stat)), list(tom_stat.keys()))
    plt.xlabel('View')
    plt.ylabel('Number of Tweets')
    plt.title('Tom Brady Statistics')
    plt.show()

    #Comparing blame and support statistics for Patriots
    plt.bar(range(len(pat_stat)), pat_stat.values(), align='center', color=jet(np.linspace(0, 1.0, len(pat_stat))))
    plt.xticks(range(len(pat_stat)), list(pat_stat.keys()))
    plt.xlabel('View')
    plt.ylabel('Number of Tweets')
    plt.title('Patriots Statistics')
    plt.show()

    #Create Blame Statistics Histogram
    plt.bar(range(len(blame_stat)), blame_stat.values(), align='center', color=jet(np.linspace(0, 1.0, len(blame_stat))))
    plt.xticks(range(len(blame_stat)), list(blame_stat.keys()))
    plt.xlabel('Blamed')
    plt.ylabel('Number of Tweets')
    plt.title('Blame Statistics')
    plt.show()

    #Create Supportt Statistics Histogram
    plt.bar(range(len(support_stat)), support_stat.values(), align='center', color=jet(np.linspace(0, 1.0, len(support_stat))))
    plt.xticks(range(len(support_stat)), list(support_stat.keys()))
    plt.xlabel('Support')
    plt.ylabel('Number of Tweets')
    plt.title('Support Statistics')
    plt.show()


def word_cloud(text):

    #Removing stop words
    text = [word_feats(words) for words in text]

    #Generate a word cloud image
    wordcloud = WordCloud().generate(str(text))

    #Plot the wordcloud
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


#Primary function thats holds the logic for entire analysis
def context_sentiment_analysis():

    #Fetching the test data and train data
    data = pd.read_csv("input.csv", encoding = 'latin1')
    train_data = pd.read_csv("corpus.csv", encoding = 'latin1')


    #Preprocessing Test and Train data
    data.text = [preprocess(text) for text in data.text]
    train_data.TweetText = [preprocess(text) for text in train_data.TweetText]
    #data.text = [advprocess(text) for text in data.text]
    #train_data.TweetText = [advprocess(text) for text in train_data.TweetText]

    #Splitting Training data into Irrelevant, neutral, positive and negative tweets.
    #This helps in better processing and training Naive Bayes Classifier
    irrids = list(train_data.TweetText[:1689])
    negids = list(train_data.TweetText[1690:2261])
    netids = list(train_data.TweetText[2262:4595])
    posids = list(train_data.TweetText[4596:])

    #Applying advanced processing on the training tweets
    pos_feats = [(advprocess(f), 'positive') for f in posids]
    neg_feats = [(advprocess(f), 'negative') for f in negids]
    irr_feats = [(advprocess(f), 'irrelevant') for f in irrids]
    net_feats = [(advprocess(f), 'neutral') for f in netids]

    #Getting the complete list of tweets for training
    trainfeats = irr_feats + pos_feats + neg_feats + net_feats

    #Shuffling the different category of tweets to mix them up
    shuffle(trainfeats)

    #Preparing Test set
    test_set = list(advprocess(words) for words in data.text)

    #Finding Sentiments for the Test set
    output_sentiments = classify_naivebayes(trainfeats, test_set)

    #Finding Contexts for the Test set
    output_contexts = [find_contexts(data) for data in data.text]

    #Analysing the results of sentiment and context classification
    analysis(output_contexts, output_sentiments)

    #Getting the wordcloud out of the tweets
    word_cloud(data.text)

    #Writing Sentiments, contexts and tweets to the output file
    output = pd.DataFrame({"Sentiments": output_sentiments, "Contexts": output_contexts, "Tweets": data.text})
    output.to_csv("output_sentiments.csv", index=False)

    #Alternate method of writing to the output file
    #with open('output_sentiments.csv', 'w') as myfile:
     #   wr = csv.writer(myfile)
      #  wr.writerow(list(output_sentiments))


#Executes when the code is used as a script
if __name__ == '__main__':

    #Calling the primary function for context and sentiment analysis
    context_sentiment_analysis()


