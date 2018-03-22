# ****************************************
# Quora Question Pair solution 
# *************************************** #

import os

from Utility import Utility
#Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import xgboost as xgb
from sklearn import metrics

from xgboost.sklearn import XGBClassifier

if __name__ == '__main__':

    #Reading test and train data.
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'train.csv'))    
    
    # print simple stats
    print("number of rows (question pairs): %i"%(train.shape[0]))
    print(train['is_duplicate'].value_counts())
    
    # find unique question ids
    unique_qids = set(list(train['qid2'].unique()) + list(train['qid1'].unique()))
    print("number of unique questions: %i" % (len(unique_qids)))
    
    '''
   # encode questions to unicode
    train['question1'] = train['question1'].apply(lambda x: unicode(str(x),"utf-8"))
    train['question2'] = train['question2'].apply(lambda x: unicode(str(x),"utf-8"))

    # Initialize an empty list to hold the clean the question pairs
    clean_train_questions = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list
    
    list_question1 = list(train['question1'])
    list_question2 = list(train['question2'])

    #all_questions = pd.concat(list_question1, list_question2, axis = 0)
    train['combined'] = train['question1'] + train['question2']
    
    print "Cleaning and parsing the training set questions...\n"
    
    for i in xrange( 0, len(train['combined'])):
        clean_train_questions.append(" ".join(Utility.review_to_wordlist(train['combined'][i], True)))
    '''   

    #Combine all questions into corpus for analysis similar to Term-frequency in TFIDF
    train_questions = pd.Series(train['question1'].tolist() + train['question2'].tolist()).astype(str)
    
    eps = 5000 
    words = (" ".join(train_questions)).lower().split()
    counts = Counter(words)
    weights = {word: Utility.get_weight(count) for word, count in counts.items()}
    print "Most common words: ", (sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])

        
    def word_share_norm(x):
        w1 = set(map(lambda word: word.lower().strip(), str(x['question1']).split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), str(x['question2']).split(" ")))    
        return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
    
    #Calculate TFIDF word match share as our new feature
    def tfidf_word_share_norm(x):
        w1 = set(map(lambda word: word.lower().strip(), str(x['question1']).split(" ")))
        w2 = set(map(lambda word: word.lower().strip(), str(x['question2']).split(" "))) 
        if len(w1)==0 or len(w2)==0:
            return 0
        common = w1 & w2
        share_weight = [weights.get(word, 0) for word in common]
        total_weight = [weights.get(word, 0) for word in w1]+[weights.get(word, 0) for word in w2]
        return np.sum(share_weight)/np.sum(total_weight)

    
    
    train_data = pd.DataFrame(dtype='float64')
    train_data['q1chrlen'] = train['question1'].str.len()
    train_data['q2chrlen'] = train['question2'].str.len()
    train_data['q1_nword'] = train['question1'].apply(lambda x: len(str(x).split(" ")))
    train_data['q2_nword'] = train['question2'].apply(lambda y: len(str(y).split(" ")))
    train_data['word_share'] = train.apply(word_share_norm, axis=1)
    train_data['TFIDF_share'] = train.apply(tfidf_word_share_norm, axis=1, raw=True)
    print train_data.head()

    #Check if there's NaN values in the data. If yes, replace them.
    np.sum(np.isnan(train_data))
    train_data.loc[:,['q1chrlen','q2chrlen']] = np.nan_to_num(train_data.loc[:,['q1chrlen','q2chrlen']])
    print np.sum(np.isnan(train_data))
    
    #Normalized feature values
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    
    y = train['is_duplicate']
    X_train, X_valid, y_train, y_valid = train_test_split(train_data, y, test_size=0.2, random_state=10)
    X_train_scaled, X_valid_scaled, y_train, y_valid = train_test_split(train_data_scaled, y, test_size=0.2, random_state=10)

    
    #define the XGBClassifier for predict the loss and accuracy.
    alg = XGBClassifier(learning_rate =0.01, n_estimators=5000, max_depth=4, min_child_weight=6,
                                     gamma=0, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005, objective= 'binary:logistic',
                                     nthread=4, scale_pos_weight=1, seed=27)
    xgb_param = alg.get_xgb_params()
    
    d_train = xgb.DMatrix(X_train_scaled, label=y_train)
    d_valid = xgb.DMatrix(X_valid_scaled, label=y_valid)
        
    cvresult = xgb.cv(xgb_param, d_train, num_boost_round=alg.get_params()['n_estimators'], nfold=5,
            metrics='auc', early_stopping_rounds=200)
    
    alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(X_train_scaled, y_train,eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_valid_scaled)
    dtrain_predprob = alg.predict_proba(X_valid_scaled)[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(y_valid, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(y_valid, dtrain_predprob)
    print "LogLoss (Train): %f" % metrics.log_loss(y_valid, dtrain_predprob)
   
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    Utility.calulateRFClassifier(X_train_scaled, X_valid_scaled, y_train, y_valid)
    Utility.calulateNB(X_train, X_valid, y_train, y_valid)
    Utility.calulateLR(X_train_scaled, X_valid_scaled, y_train, y_valid)
    Utility.calulateSGD(X_train_scaled, X_valid_scaled, y_train, y_valid)
    Utility.calculateKN(X_train_scaled, X_valid_scaled, y_train, y_valid)
     
'''
    y = train['is_duplicate']
    
    X_train, X_test, y_train, y_test = train_test_split(clean_train_questions, y, test_size=0.2, random_state=10)
       
        
    # ****** Create a TFIDF-Vectorizer from the training set
    #
    tfv = TfidfVectorizer(min_df = 3,  max_features = 3000, 
            strip_accents = 'unicode', analyzer = 'word', token_pattern = r'\w{1,}',
            ngram_range = (1, 2), use_idf = 1, smooth_idf = 1, sublinear_tf = 1,
            stop_words = 'english', norm = 'l1', lowercase = True)
        
    # ****** building the pipline for SGD model.
    #
    pipline_SGD = Utility.get_SGD_model(tfv)
    
    # ****** Calculating log loss for SGD model
    #
    Utility.calculate_log_loss("SGDClassifier", pipline_SGD, X_train, X_test, y_train, y_test)
    
    # ****** building the pipline for RandomForestClassifier model.
    #
    pipline_rfc = Utility.get_RFClassifier_model(tfv)
    
    # ****** Calculating log loss for RandomForestClassifier model
    #
    Utility.calculate_log_loss("RandomForestClassifier", pipline_rfc, X_train, X_test, y_train, y_test)
    
    
    # ****** building the pipline for LogisticRegression model.
    #
    pipline_lr = Utility.get_LR_model(tfv)
    
    # ****** Calculating log loss for LogisticRegression model
    #
    Utility.calculate_log_loss("LogisticRegression", pipline_lr, X_train, X_test, y_train, y_test)
    
    
    Utility.calulateNB(X_train, X_test, y_train, y_test)
    
    Utility.calulateXgb(X_train, X_test, y_train, y_test)
    
    XGBpipe  = Utility.xgb_model(tfv)
    
    Utility.calculate_log_loss("XGBoost", XGBpipe, X_train, X_test, y_train, y_test)
    
    Utility.calulateDesisionTree(X_train, X_test, y_train, y_test)
    
'''

 
    
