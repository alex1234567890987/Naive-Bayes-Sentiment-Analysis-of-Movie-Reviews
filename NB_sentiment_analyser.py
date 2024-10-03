# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
import pandas
import string
import numpy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import chi2_contingency



"""
IMPORTANT, modify this part with your details
"""
USER_ID = "acc21ag" #your unique student ID, i.e. the IDs starting with "acp", "mm" etc that you use to login into MUSE 

# gets a list of spotwords from nltk package
STOP_WORDS = set(stopwords.words("english"))

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "adjs", "verbs", "adverbs", "superlatives", "all_pos", "polarity", "most_significant", "all", "features"])  # add in more where it says features, i.e adj, verb etc
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args

# func calcuates prior probabilites for each class
def calculate_prior_probabilties(data, number_classes):
    prior_probabilties = {}
    
    for sentiment in range(number_classes):
        prior_probabilties[sentiment] = calculate_total_phrase_for_sentiment_x(data, sentiment) / data.shape[0]   # count of phrases of sen x / count of all phrases
        
    return prior_probabilties

# func reduces sentiment classes from 5 to 3
def reduce_sentiment_classes(sentiment):
    if sentiment > 2:
        new_sentiment = 2
    elif sentiment < 2:
        new_sentiment = 0
    elif sentiment == 2:
        new_sentiment = 1
        
    return new_sentiment
    
# func counts phrase belonging to sentiment x - used on training data
def calculate_total_phrase_for_sentiment_x(data, sentiment):
    return data[data['Sentiment'] == sentiment].shape[0]

# func filters the data by a sentiment - used on training data
def filter_by_sentiment(data, sentiment):
    return data[data['Sentiment'] == sentiment]

# func preprocesses phrase
def preprocess_phrase(text):
    words = word_tokenize(text)
    
    processed_words = [] 
    
    # stop list 
    for word in words:
        if word.lower() not in STOP_WORDS:
            processed_words.append(word.lower())
            
    processed_words_2 = []
    
    # remove punctuation
    punctuation = string.punctuation.replace('-', '')
    
    for word in processed_words:
        if word not in string.punctuation and not any(set(word) & set(punctuation)):
            processed_words_2.append(word)
        if '!' in word:
            processed_words_2.append(word)
            
    processed_words_3 = processed_words_2.copy()
    
    for index, word in enumerate(processed_words_2):
        if (word == "not" or word == "never") and index != len(processed_words_2)-1:
            new_word = processed_words_2[index] + "_" + processed_words_2[index+1]
            processed_words_3.pop(index)
            processed_words.pop(index)
            processed_words.append(new_word)
            
    # add more steps here
    
    processed_text = ' '.join(processed_words_3)
    
    return processed_text

# func filters out non features from phrase
def apply_features_to_phrase(phrase, feature_list):
    words = word_tokenize(phrase)
    
    feature_words = []
    
    for word in words:
        if word in feature_list:
            feature_words.append(word)
            
    processed_text = ' '.join(feature_words)
    
    return processed_text

# func processes all phrases for features
def process_phrases_for_feature(data, feature_list):
    data['Preprocessed_Phrase'] = data['Preprocessed_Phrase'].apply(apply_features_to_phrase, feature_list=feature_list)

# func gets all words in a list from a list of phrases 
def get_all_words_from_phrases(phrase_list):
    all_words = []
    
    for phrase in phrase_list:
        words = phrase.split()
        for word in words:
            all_words.append(word)
            
    return(all_words)

# func returns the count of words for each sentiment
def create_word_for_sentiment_len_dict(data, number_classes, all_words):
    word_for_sentiment_len_dict = {}
    
    for c in range(number_classes): 
        data_for_sentiment = filter_by_sentiment(data, c)
        phrase_list = data_for_sentiment['Preprocessed_Phrase'].tolist()
        all_words_for_sentiment = get_all_words_from_phrases(phrase_list)
        word_for_sentiment_len_dict[str(c)] = len(all_words_for_sentiment) 
        
    return word_for_sentiment_len_dict

# func creates a likelihood dict for each word by sentiment of the training data
def calculate_likelihood_for_sentiment(data, sentiment, all_words):
    likelihood = {}
    
    data_for_sentiment = filter_by_sentiment(data, sentiment)
    phrase_list = data_for_sentiment['Preprocessed_Phrase'].tolist()
    all_words_for_sentiment = get_all_words_from_phrases(phrase_list)
    
    for word in all_words:
        if word in all_words_for_sentiment:
            likelihood[word] = (all_words_for_sentiment.count(word) + 1) / (len(all_words_for_sentiment) + len(all_words))
        else:
            likelihood[word] = 1 / (len(all_words_for_sentiment) + len(all_words))
        
    return likelihood

# func gets the likilihood product of a phrase
def get_likelihood_product_for_phrase(phrase, likelihoods, all_words, sentiment, training_data, word_for_sentiment_len_dict):
    likelihood_product = 1
    
    phrase_words = []
    
    words = phrase.split()
    for word in words:
        phrase_words.append(word)
        
    for word in phrase_words:
        if likelihoods.get(word) is None:
            likelihood_product = likelihood_product * (1 / (word_for_sentiment_len_dict[str(sentiment)] + len(all_words)))
        else:
            likelihood_product = likelihood_product * likelihoods.get(word) 
        
    return likelihood_product

# func calculates the posterior probabilty for a sentiment 
def calculate_post_probabilty(phrase, prior_probabilty, likelihoods, all_words, sentiment, training_data, word_for_sentiment_len_dict):
    likelihood = get_likelihood_product_for_phrase(phrase, likelihoods, all_words, sentiment, training_data, word_for_sentiment_len_dict)
        
    return likelihood * prior_probabilty
    
# func preprocces all phrases in dataset
def preprocess_phrases_for_data(data):
    data['Preprocessed_Phrase'] = data['Phrase'].apply(preprocess_phrase)
    
# func calcuates the estimate sentiment of all phrases
def calculate_estimate_sentiment(training_data, data, all_words, prior_probabilties, number_classes, word_for_sentiment_len_dict):
    for c in range(number_classes):
        likelihoods = calculate_likelihood_for_sentiment(training_data, c, all_words)
        
        prior_probabilty = prior_probabilties[c]
        
        data[str(c)] = data['Preprocessed_Phrase'].apply(calculate_post_probabilty, prior_probabilty=prior_probabilty, likelihoods=likelihoods, all_words=all_words, sentiment=c, training_data=training_data, word_for_sentiment_len_dict=word_for_sentiment_len_dict)
        
    if number_classes == 3:
        data['Estimated Sentiment'] = data[["0", "1", "2"]].idxmax(axis=1)
    else:
        data['Estimated Sentiment'] = data[["0", "1", "2", "3", "4"]].idxmax(axis=1)
        
    data['Estimated Sentiment'] = data['Estimated Sentiment'].astype(int)

# func calculates f1 score and confusion matrix
def compare_estimate_sentiment_to_real(data, number_classes, use_con_matrix):
    f1_scores = {}    
    f1_macro_scores = {}
    
    for c in range(number_classes):
        f1_scores[str(c)] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    
    for c in range(number_classes):        
        
        f1_scores[str(c)]["TP"] = len(data[(data['Sentiment'] == c) & (data['Sentiment'] == data['Estimated Sentiment'])])
        f1_scores[str(c)]["TN"] = len(data[(data['Sentiment'] != c) & (data['Sentiment'] == data['Estimated Sentiment'])])
        f1_scores[str(c)]["FP"] = len(data[(data['Estimated Sentiment'] == c) & (data['Sentiment'] != data['Estimated Sentiment'])])
        f1_scores[str(c)]["FN"] = len(data[(data['Estimated Sentiment'] != c) & (data['Sentiment'] != data['Estimated Sentiment'])])
    
    for c in range(number_classes):
        f1_macro_scores[str(c)] = (2 * int(f1_scores[str(c)]["TP"])) / ((2 * int(f1_scores[str(c)]["TP"])) + int(f1_scores[str(c)]["FP"]) + int(f1_scores[str(c)]["FN"]))
        
        
    f1_macro_score = numpy.mean(list(f1_macro_scores.values()))
    
    if use_con_matrix == True:
        y_true = [data['Sentiment'].astype(int).tolist()]
        y_pred = [data['Estimated Sentiment'].astype(int).tolist()]
        
        # calculate confusion matrix
        con_matrix = pandas.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=False)
        
        return(f1_macro_score, f1_scores, con_matrix)

    
    return (f1_macro_score, f1_scores)

        
# gets the feature lists according to chosen feature
def get_features(features, all_words, pos_tags, training_data):
    if features == "adjs":
        pos_features = []
        for word, pos in pos_tags:
            if pos.startswith('JJ') and word in all_words:
                pos_features.append(word)
                
        all_words = pos_features
        
    if features == "verbs":
        pos_features = []
        for word, pos in pos_tags:
            if pos.startswith('VB')  and word in all_words:
                pos_features.append(word)
                
        all_words = pos_features
        
    if features == "adverbs":  
        pos_features = []
        for word, pos in pos_tags:
            if pos.startswith('RB') and word in all_words:
                pos_features.append(word)
                
        all_words = pos_features
        
    if features == "superlatives":
        pos_features = []
        for word, pos in pos_tags:
            if pos.startswith('JJS') and word in all_words:
                pos_features.append(word)
                
        all_words = pos_features
        
    if features == "all_pos":
        pos_features = []
        for word, pos in pos_tags:
            if (pos.startswith('JJ') or pos.startswith('RB') or pos.startswith('VB')) and word in all_words:
                pos_features.append(word)
                
        all_words = pos_features
        
    if features == "polarity":
        pol_features = []
        
        sid = SentimentIntensityAnalyzer()
        
        for word in all_words:
            if (sid.polarity_scores(word)['compound'] < -0.5) or (sid.polarity_scores(word)['compound'] > 0.5):
                pol_features.append(word)
        
        all_words = pol_features
    
    if features == "most_significant":
        contingency_tables = {}
        
        for word in all_words:
            contingency_table = pandas.crosstab(training_data[training_data['Preprocessed_Phrase'].str.contains(word)]['Sentiment'], training_data[training_data['Preprocessed_Phrase'].str.contains(word)]['Sentiment'])
            contingency_tables[word] = contingency_table

        # apply test on each word, append if significant
        significant_words = []
        for word, table in contingency_tables.items():
            chi2, p, _, _ = chi2_contingency(table)
            if p < 0.05:
                significant_words.append(word)
                
        all_words = significant_words
        
    if features == "all":
        contingency_tables = {}
        
        for word in all_words:
            contingency_table = pandas.crosstab(training_data[training_data['Preprocessed_Phrase'].str.contains(word)]['Sentiment'], training_data[training_data['Preprocessed_Phrase'].str.contains(word)]['Sentiment'])
            contingency_tables[word] = contingency_table

        # apply test on each word, append if significant
        significant_words = []
        for word, table in contingency_tables.items():
            chi2, p, _, _ = chi2_contingency(table)
            if p < 0.05:
                significant_words.append(word)
                
        pol_features = []
        
        sid = SentimentIntensityAnalyzer()
        
        for word in all_words:
            if (sid.polarity_scores(word)['compound'] > 0.5) or (sid.polarity_scores(word)['compound'] < -0.5):
                pol_features.append(word)

        pos_features = []
        for word, pos in pos_tags:
            if (pos.startswith('JJ') or pos.startswith('RB') or pos.startswith('VB')) and word in all_words:
                pos_features.append(word)
                
        all_words = list(set(significant_words + pol_features + pos_features))
    
    
def main():
    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix
    
    """
    ADD YOUR CODE HERE
    Create functions and classes, using the best practices of Software Engineering
    """
    # create a training data panda
    training_data = pandas.read_csv("moviereviews/"+training,sep='\t')
    
    # create a dev data panda
    dev_data = pandas.read_csv("moviereviews/"+dev,sep='\t')
    
    if number_classes == 3:
        training_data['Sentiment'] = training_data['Sentiment'].apply(reduce_sentiment_classes)
        dev_data['Sentiment'] = dev_data['Sentiment'].apply(reduce_sentiment_classes)

    # create a test data panda
    test_data = pandas.read_csv("moviereviews/"+test,sep='\t')
    
    # preprocess all phrases
    preprocess_phrases_for_data(training_data)
    preprocess_phrases_for_data(dev_data)
    preprocess_phrases_for_data(test_data)
    
    # get a list of the phrases
    training_data_phrase_list = training_data['Preprocessed_Phrase'].tolist()
    
    # get all_words from data set
    all_words = list(set(get_all_words_from_phrases(training_data_phrase_list)))
    
    pos_tags = pos_tag(all_words)
    
    get_features(features, all_words, pos_tags, training_data)
    
    if features != "all_words":
        process_phrases_for_feature(training_data, all_words)
        process_phrases_for_feature(dev_data, all_words)
        process_phrases_for_feature(test_data, all_words)
        
    # creates a dict of prior probabilties for each class
    prior_probabilties = calculate_prior_probabilties(training_data, number_classes)
        
    word_for_sentiment_len_dict = create_word_for_sentiment_len_dict(training_data, number_classes, all_words)
    
    # creates a new column for estimated sentiment in the given data set 
    # comment out for which one you want to run (must run both to output files, do not run confusion matrix for test data)
    calculate_estimate_sentiment(training_data, dev_data, all_words, prior_probabilties, number_classes, word_for_sentiment_len_dict)  # run on development data 
    #calculate_estimate_sentiment(training_data, test_data, all_words, prior_probabilties, number_classes, word_for_sentiment_len_dict)  # run on test data

    f1_macro_score = 0
    
    if confusion_matrix == True:
        f1_macro_score, f1_scores, con_matrix = compare_estimate_sentiment_to_real(dev_data, number_classes, confusion_matrix)
        print("Confusion Matrix:")
        print(con_matrix)
    else:
        f1_macro_score, f1_scores = compare_estimate_sentiment_to_real(dev_data, number_classes, confusion_matrix)
    
    f1_score = f1_macro_score
    
    if output_files == True:
        dev_data_output_format = dev_data[['SentenceId', 'Estimated Sentiment']]
        dev_data_output_format = dev_data_output_format.rename(columns={'Estimated Sentiment': 'Sentiment'})

        test_data_output_format = test_data[['SentenceId', 'Estimated Sentiment']]
        test_data_output_format = test_data_output_format.rename(columns={'Estimated Sentiment': 'Sentiment'})

        if number_classes == 3:
            dev_data_output_format.to_csv('dev_predictions_3class_acc21ag.tsv', sep='\t', index=False)
            
            test_data_output_format.to_csv('test_predictions_3class_acc21ag.tsv', sep='\t', index=False)
        
        if number_classes == 5:
            dev_data_output_format.to_csv('dev_predictions_5class_acc21ag.tsv', sep='\t', index=False)
            
            test_data_output_format.to_csv('test_predictions_5class_acc21ag.tsv', sep='\t', index=False)
    
    """
    IMPORTANT: your code should return the lines below. 
    However, make sure you are also implementing a function to save the class predictions on dev and test sets as specified in the assignment handout
    """
    print("Student\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()
