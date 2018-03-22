#!/usr/bin/env python

import re

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from sklearn import metrics


class Utility(object):
    """utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def review_to_wordlist( review, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        # 1. Remove HTML
        review_text = BeautifulSoup(review).get_text()
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case and split them
        words = review_text.lower().split()
        #
        # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # 5. Return a list of words
        return(words)
        
    #Implement TFIDF function
    @staticmethod
    def get_weight(count, eps=5000, min_count=2):
        if count < min_count:
            return 0 #remove words only appearing once 
        else:
            R = 1.0 / (count + eps)
            return R

    # Define a function to split a review into parsed sentences
    @staticmethod
    def review_to_sentences( review, tokenizer, remove_stopwords=False ):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append(Utility.review_to_wordlist( raw_sentence, \
                  remove_stopwords ))
        #
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences
    
    
    # Define a function to get SGD model to calculate log_loss
    @staticmethod
    def get_SGD_model(tfidfVec):
        
        #define the SGD classifier for predict the loss and accuracy.
        classifier = SGDClassifier(loss='log', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
        
        #
        # Return the pipeline to transforms with a final estimator 
        # so this returns pipeline
        return Pipeline([('tfidfvec', tfidfVec), ('tfidfTras', TfidfTransformer()), ('clf-svm', classifier)])
    
    
    # Define a function to get RandomForestClassifier model to calculate log_loss
    @staticmethod
    def get_RFClassifier_model(tfidfVec):
        
        #define the SGD classifier for predict the loss and accuracy.
        classifier = RandomForestClassifier(n_estimators = 200, min_samples_leaf = 10, n_jobs = -1)
            
        #
        # Return the pipeline to transforms with a final estimator 
        # so this returns pipeline
        return Pipeline([('tfidfvec', tfidfVec), ('tfidfTras', TfidfTransformer()), ('rfc', classifier)])
    
    # Define a function to get LogisticRegression model to calculate log_loss
    @staticmethod
    def get_LR_model(tfidfVec):
        
        #define the LogisticRegression for predict the loss and accuracy.
        regression = LogisticRegression()
            
        #
        # Return the pipeline to transforms with a final estimator 
        # so this returns pipeline
        return Pipeline([('tfidfvec', tfidfVec), ('tfidfTras', TfidfTransformer()), ('lr', regression)])
    
    
    
    # Define a function to calculate log loss and accuracy score
    @staticmethod
    def calculate_log_loss(model_name, pipeline, X_train, X_test, y_train, y_test):

        # fit and transforms our training data
        # so model can be used to predict the test data.
        
        # Fit the pipeline to the training set, using TF-IDF as
        # features and the is_duplicate labels as the response variable
        #
        # This may take a few minutes to run
        model = pipeline.fit(X_train, y_train)
        
        print "Calculating log loss and accuracy (this may take a while)..."
    
        # predict test data using model
        predict_model = model.predict_proba(X_test)[:,1]
                
        # calculate log loss using predicted value and y_test 
        logloss_rf = log_loss(y_test, predict_model)
        print "Calculated log-loss using ", model_name, " is :", logloss_rf, "\n"
        
        accuracy = accuracy_score(y_test, predict_model.round())
        print "Calculated accuracy-score using ", model_name, " is :", accuracy, "\n\n"
        
        
    @staticmethod
    def calulateXgb(X_train_scaled, X_valid_scaled, y_train, y_valid):
        param_test1 = {
         'max_depth':range(3,10,2),
         'min_child_weight':range(1,6,2)
        }
        gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
                                 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
                                 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
        gsearch1.fit(X_train_scaled, y_train)
        
        # predict test data using model
        predict_model = gsearch1.predict_proba(X_valid_scaled)
        
        
        print ("XGB logloss: %0.3f " % Utility.multiclass_logloss(y_valid, predict_model))
        
        
        
    @staticmethod
    def calulateNB(X_train_scaled, X_valid_scaled, y_train, y_valid):
        
        clf = MultinomialNB()
        clf.fit(X_train_scaled, y_train)
        predictions = clf.predict_proba(X_valid_scaled)
        
        print ("Naive Bayes on TFIDF logloss: %0.3f " % Utility.multiclass_logloss(y_valid, predictions))
        
        
    @staticmethod
    def calulateSGD(X_train_scaled, X_valid_scaled, y_train, y_valid):
        
        #define the SGD classifier for predict the loss and accuracy.
        classifier = SGDClassifier(loss='log', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
        classifier.fit(X_train_scaled, y_train)
        predictions = classifier.predict_proba(X_valid_scaled)
        
        print ("SGD classifier logloss: %0.3f " % Utility.multiclass_logloss(y_valid, predictions))
        
    @staticmethod
    def calulateLR(X_train_scaled, X_valid_scaled, y_train, y_valid):
        
        #define the LogisticRegression for predict the loss and accuracy.
        regression = LogisticRegression(random_state=10)
        grid = {
            'C': [1e-6, 1e-3, 1e0, 1e3, 1e6],
            'penalty': ['l1', 'l2']
        }
        cv = GridSearchCV(regression, grid, scoring='neg_log_loss', verbose = True)
        cv.fit(X_train_scaled, y_train)
        predictions = cv.predict_proba(X_valid_scaled)
        print ("LogisticRegression logloss: %0.3f " % Utility.multiclass_logloss(y_valid, predictions))
        
    @staticmethod
    def calulateRFClassifier(X_train_scaled, X_valid_scaled, y_train, y_valid):
        
        #define the RandomForestClassifier for predict the loss and accuracy.
        rfclassifier = RandomForestClassifier(max_depth=5, random_state=0)
        rfclassifier.fit(X_train_scaled, y_train)
        predictions = rfclassifier.predict_proba(X_valid_scaled)
        
        print ("RandomForestClassifier logloss: %0.3f " % Utility.multiclass_logloss(y_valid, predictions))
    
    @staticmethod
    def calculateKN(X_train_scaled, X_valid_scaled, y_train, y_valid):
        #Model 4: K Nearest Neighbors
        #Use grid search for best parameter
        knn = KNeighborsClassifier()
        grid = {
            'n_neighbors': list(range(2, 10, 2)),
            'weights': ['uniform', 'distance']
        }
        cv = GridSearchCV(knn, grid, scoring='neg_log_loss', verbose = True)
        cv.fit(X_train_scaled, y_train)
        predictions = cv.predict_proba(X_valid_scaled)
        
        print ("KNeighborsClassifier logloss: %0.3f " % metrics.log_loss(y_valid, predictions))
        
        
    @staticmethod   
    def multiclass_logloss(actual, predicted, eps=1e-15):
        # Convert 'actual' to a binary array if it's not already:
        if len(actual.shape) == 1:
            actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
            for i, val in enumerate(actual):
                actual2[i, val] = 1
            actual = actual2
    
        clip = np.clip(predicted, eps, 1 - eps)
        rows = actual.shape[0]
        vsota = np.sum(actual * np.log(clip))
        return -1.0 / rows * vsota
                        
       
   
            
