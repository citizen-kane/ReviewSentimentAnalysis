
# coding: utf-8

# In[119]:

import numpy
import sys
import nltk
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import scipy
import pandas as pd
from string import digits
import string
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


# In[87]:

#READING ALL THE THREE FILES
file_imdb=open("imdb_labelled.txt","r")
imdb=[]
for line in file_imdb:
    line = line.split('\n')
    line = line[0].split('\t')
    line[1] = int(line[1])
    imdb.append(line)
file_imdb.close()
imdb_df = pd.DataFrame(imdb, columns=["Summary", "Score"])
file_amazon=open("amazon_cells_labelled.txt","r")
amazon=[]
for line in file_amazon:
    line = line.split('\n')
    line = line[0].split('\t')
    line[1] = int(line[1])
    amazon.append(line)
file_amazon.close()
amazon_df = pd.DataFrame(amazon, columns=["Summary", "Score"])
file_yelp=open("yelp_labelled.txt","r")
yelp=[]
for line in file_yelp:
    line = line.split('\n')
    line = line[0].split('\t')
    line[1] = int(line[1])
    yelp.append(line)
file_yelp.close()
yelp_df = pd.DataFrame(yelp, columns=["Summary", "Score"])


# In[88]:

#MERGING ALL FILES INTO ONE DATAFRAME
final_data_df = pd.concat([imdb_df,amazon_df,yelp_df], ignore_index= True)
print ("Count of final table: ", final_data_df["Summary"].count())


# In[115]:

#DATA CLEANSING APPROACHES
data=[]
ps = PorterStemmer()
lmtzr = WordNetLemmatizer()
#data_df['Summary']=data_df['Summary'].str.lower()
for index in range(len(final_data_df['Summary'])):
    #NORMAL
    #data.append([final_data_df['Summary'][index],final_data_df['Score'][index]])
    #LOWERCASE
    #data.append([final_data_df['Summary'][index].lower(),final_data_df['Score'][index]])
    #No Numbers
    #data.append([final_data_df['Summary'][index].translate(None, digits),final_data_df['Score'][index]])
    #No Punctuation
    #data.append([final_data_df['Summary'][index].translate(None, string.punctuation),final_data_df['Score'][index]])
    #No Numbers AND Punctuation
    #data.append([final_data_df['Summary'][index].translate(None, string.punctuation).translate(None, digits),final_data_df['Score'][index]])
    #Word Tokenize
    #words = word_tokenize(final_data_df['Summary'][index])
    #data.append([' '.join(word for word in words),final_data_df['Score'][index]])
    #StopWords
    #words = [word for word in final_data_df['Summary'][index].split() if word not in stopwords.words('english')]
    #data.append([' '.join(word for word in words),final_data_df['Score'][index]])
    #Stemming
    #words = word_tokenize(final_data_df['Summary'][index].translate(None, string.punctuation))
    #words = [ps.stem(word.decode("utf8")) for word in words]
    #data.append([' '.join(word for word in words),final_data_df['Score'][index]])
    #Lemmatizer

data_df = pd.DataFrame(data, columns=["Summary", "Score"])

print (data_df[:10])


# In[116]:

#SPLITTING FILES RANDOMLY - 70% TRAINING & 30% TESTING
[Data_train,Data_test,Train_labels,Test_labels] = train_test_split(data_df['Summary'],
                                                                   data_df['Score'] , 
                                                                   test_size=0.3, 
                                                                   random_state=42)
print ("Data train count = ", Data_train.count())
print ("Data test count = ", Data_test.count())

print ("\nTraining Data :")
print (Data_train[:5])

print ("\nTesting Data :")
print (Data_test[:5])


# In[117]:

classifier = DecisionTreeClassifier(random_state=20160121, criterion='entropy')


# In[118]:

#Bigram Model
bigram_vec = TfidfVectorizer(ngram_range=(1, 2),                     
                             strip_accents='unicode',
                             min_df=2,
                             norm='l2')

bigram_model = bigram_vec.fit(data_df['Summary'])
bigram_train = bigram_model.transform(Data_train)
bigram_test  = bigram_model.transform(Data_test)

bigram_clf = classifier.fit(bigram_train, Train_labels)
bigram_prediction = bigram_clf.predict(bigram_test)

print (metrics.classification_report(Test_labels.values, bigram_prediction))

bi_conf_mat = metrics.confusion_matrix(Test_labels.values, bigram_prediction)

print ("\nConfusion Matrix for BIGRAM:")
print ('\t\tPrediction')
print ("\t\tNEG\tPOS")

print "Actual","\tNEG\t", bi_conf_mat[0][0], "\t", bi_conf_mat[0][1]

print "\tPOS\t", bi_conf_mat[1][0], "\t", bi_conf_mat[1][1]

bi_accuracy = (float(bi_conf_mat[0][0]+bi_conf_mat[1][1])*100/900)
print ("\nAccuracy for BIGRAM")
print (bi_accuracy)


# In[60]:

#Unigram Model
unigram_vec = TfidfVectorizer(ngram_range=(1, 1),                     
                             strip_accents='unicode',
                             min_df=2,
                             norm='l2')

unigram_model = unigram_vec.fit(data_df['Summary'])
unigram_train = unigram_model.transform(Data_train)
unigram_test  = unigram_model.transform(Data_test)

unigram_clf = classifier.fit(unigram_train, Train_labels)
unigram_prediction = unigram_clf.predict(unigram_test)
print (metrics.classification_report(Test_labels.values, unigram_prediction))

uni_conf_mat = metrics.confusion_matrix(Test_labels.values, unigram_prediction)
print ("\nConfusion Matrix for UNIGRAM:")
print ('\t\tPrediction')
print ("\t\tNEG\tPOS")
print ("Actual\tNEG\t", uni_conf_mat[0][0], "\t", uni_conf_mat[0][1])
print ("\tPOS\t", uni_conf_mat[1][0], "\t", uni_conf_mat[1][1])

uni_accuracy = (float(uni_conf_mat[0][0]+uni_conf_mat[1][1])*100/900)
print ("\nAccuracy for UNIGRAM")
print (uni_accuracy)


# In[61]:

#Trinigram Model
trigram_vec = TfidfVectorizer(ngram_range=(1, 3),
                             strip_accents='unicode',
                             min_df=2,
                             norm='l2')

trigram_model = trigram_vec.fit(data_df['Summary'])
trigram_train = trigram_model.transform(Data_train)
trigram_test  = trigram_model.transform(Data_test)

trigram_clf = classifier.fit(trigram_train, Train_labels)
trigram_prediction = trigram_clf.predict(trigram_test)
print (metrics.classification_report(Test_labels.values, trigram_prediction))

tri_conf_mat = metrics.confusion_matrix(Test_labels.values, trigram_prediction)
print ("\nConfusion Matrix for TRIGRAM:")
print ('\t\tPrediction')
print ("\t\tNEG\tPOS")
print ("Actual\tNEG\t", tri_conf_mat[0][0], "\t", tri_conf_mat[0][1])
print ("\tPOS\t", tri_conf_mat[1][0], "\t", tri_conf_mat[1][1])

accuracy = (float(tri_conf_mat[0][0]+tri_conf_mat[1][1])*100/900)
print ("\nAccuracy for TRIGRAM")
print (accuracy)


# In[62]:

i=0
fp_count=0
fn_count=0
false_pos=[]
false_neg=[]
for idx, value in Test_labels.iteritems():
    if(value==0 and bigram_prediction[i]==1):
        false_pos.append("False Positive: "+Data_test[idx])
        fp_count+=1
    if(value==1 and bigram_prediction[i]==0):
        false_neg.append("False Negative: "+Data_test[idx])
        fn_count+=1
    i+=1
#print (fp_count)
#print (fn_count)
print (false_pos[1])
print (false_neg[1])


# In[ ]:



