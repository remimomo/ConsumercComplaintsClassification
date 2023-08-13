#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('pip', 'install matplotlib')
get_ipython().run_line_magic('pip', 'install pandas')
get_ipython().run_line_magic('pip', 'install numpy')
get_ipython().system('pip install xgboost')
get_ipython().system('pip install lightgbm')
get_ipython().system('pip install shap')
get_ipython().system('pip install shapash')
get_ipython().system('pip install lime')
get_ipython().system('pip install Pillow==9.0.0')
get_ipython().system('pip install nltk')
get_ipython().system('pip install pytest')
get_ipython().system('pip install gensim')
get_ipython().system('pip install bs4')
get_ipython().system('pip install lightgbm')
get_ipython().system('pip install scipy')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install pickle')
#Installing packages at notebook level
#dbutils.library.installPyPI("nltk")
#dbutils.library.installPyPI("pytest")
#dbutils.library.installPyPI("gensim")
#dbutils.library.installPyPI("bs4")
#dbutils.library.installPyPI("scipy", "1.2.1")
#dbutils.library.installPyPI("scikit-learn", "0.23.2")


# In[ ]:


import numpy as np
import pandas as pd
seed = 69 # set random seed for whole document

# Graph plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Displaying dataframes
from IPython.display import display

# Natural Language Processing Thingamajibs
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.models import Word2Vec, word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import gensim

# Classifiers
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC

# Metrics to score classifiers
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve, log_loss

# Data splitting, CV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

# Lifesaver
import pickle


# ##Web Scrapping

# In[ ]:


get_ipython().run_line_magic('sh', '')
pip install --upgrade pip


# In[ ]:


get_ipython().run_line_magic('sh', '')
pip install beautifulsoup4


# In[ ]:


get_ipython().run_line_magic('sh', '')
pip install datefinder


# In[ ]:


import requests
page = requests.get("https://www.complaintsboard.com/imperial-tobacco-australia-b128435")
page.status_code


# In[ ]:


import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import csv
complaint_collect=[]
brand_collect=[]
date_collect=[]
URL ='https://www.complaintsboard.com/imperial-tobacco-australia-b128435/page/'
for page in range(1,35):
    req = requests.get(URL + str(page) + '/')
    soup = bs(req.content, 'html.parser')
    main_content = soup.find(class_="bn-complaints")
    main_items = main_content.find_all(itemprop_="reviewBody")
    date = [sd.get_text() for sd in main_content.select(".complaint-main .author-header__date")]
    complaints = [sd.get_text() for sd in main_content.select(".complaint-main .complaint-main__text")]
    brand=[sd.get_text() for sd in main_content.select(".complaint-main .complaint-main__header-name")]
    complaint_collect.append(complaints)
    brand_collect.append(brand)
    date_collect.append(date)
tuple1= list(zip(brand_collect))
tuple2=list(zip(complaint_collect))
tuple3=list(zip(date_collect))
df=pd.DataFrame(tuple1,columns=['brand'])
df2=pd.DataFrame(tuple2,columns=['complaints'])
df3=pd.DataFrame(tuple3,columns=['Date'])

#frames=[df,df2]
exp1=df.explode('brand')
exp2=df2.explode('complaints')
exp3=df3.explode('Date')
exp1=exp1.reset_index()
exp1=exp1.drop(['index'], axis=1)
exp2=exp2.reset_index()
exp2=exp2.drop(['index'], axis=1)
exp3=exp3.reset_index()
exp3=exp3.drop(['index'], axis=1)
exp4=pd.concat([exp3,exp1, exp2], axis=1)
exp4


# In[ ]:


exp4['Date'].isna().sum()


# In[ ]:


exp4['Date'].count()


# In[ ]:


df_final = exp4[exp4['Date'].notna()]


# In[ ]:


df = spark.createDataFrame(df_final)
display(df)


# In[ ]:


df.select('brand').distinct().show()


# In[ ]:


df.select('brand').distinct().count()


# In[ ]:


df.distinct().count()


# In[ ]:


len(df.columns)


# In[ ]:


#convert to pandas df
df1 = df.toPandas()


# In[ ]:


#Dropping all rows that do not have Customer Complaint entries in them
df1.dropna(axis=0, subset=['complaints'], 
          inplace=True)


# In[ ]:


# Subsetting dataframe into columns useful for our text multi-classification problem
df_pr_com = df1[['brand', 'complaints']]


# In[ ]:


# # Pickling our subsetted dataframe
with open('df_product_and_complaint.pickle', 'wb') as to_write:
    pickle.dump(df_pr_com, to_write)


# In[ ]:


# Loading our pickled subsetted dataframe
with open('df_product_and_complaint.pickle', 'rb') as to_read:
    df_pr_com = pickle.load(to_read)


# In[ ]:


# Checking our dataframe
df_pr_com.info()

# Great! We have no null values in each column


# In[ ]:


# Now let's take a look at some observations of what these rows look like
df_pr_com.head(5)


# In[ ]:


# Exploring the number of mutli-class categories we have
print('--------------')
print('Categories in brand column:')
print('--------------\n')
print(df_pr_com['brand'].unique(), '\n')
print('--------------')
print('# of unique categories: ', df_pr_com['brand'].nunique())
print('--------------')


# In[ ]:


fig, ax = plt.subplots(figsize=(10,8))

ax = sns.countplot(y='brand', 
                   data=df_pr_com, 
                   order=df_pr_com['brand'].value_counts().iloc[:60].index)

ax.set_title('NUMBER OF COMPLAINTS IN EACH CATEGORY',size=15)

# Setting labels
# Dealing with y-labels
ax.set_ylabel('COMPLAINT', rotation=0, size=14, labelpad=10)
              
# Dealing with x-labels
ax.set_xlabel('NUMBER OF COMPLAINTS', size=14)

sns.despine()
            
plt.savefig('freq_of_uncombined_class.png', transparency=True)


# In[ ]:


print('Number of rows in Dataframe: ', len(df_pr_com))


# In[ ]:


# Pre-drop category value_counts
df_pr_com.brand.value_counts()


# In[ ]:


#Replace whole string if it contains substring & make it case insensitive
df_pr_com.loc[df_pr_com['brand'].str.contains('WhiteOx|Ox|White Ox|White|Ox|25 gram white ox', case=False), 'brand'] = 'WhiteOx'
df_pr_com.loc[df_pr_com['brand'].str.contains('White ox 50g and 25g tobacco', case=False), 'brand'] = 'WhiteOx'
df_pr_com.loc[df_pr_com['brand'].str.contains('rivers tone|Riverstone|rivers', case=False), 'brand'] = 'Riverstone'
df_pr_com.loc[df_pr_com['brand'].str.contains('25g riverstone pouch of tobacco', case=False), 'brand'] = 'Riverstone'
df_pr_com.loc[df_pr_com['brand'].str.contains('river|Rivestone rum blend 25g tabacco', case=False), 'brand'] = 'Riverstone'
df_pr_com.loc[df_pr_com['brand'].str.contains('Riverstone', case=False), 'brand'] = 'Riverstone'
df_pr_com.loc[df_pr_com['brand'].str.contains('parker and simpson|parker|simpson', case=False), 'brand'] = 'Parker&Simpson'
df_pr_com.loc[df_pr_com['brand'].str.contains('Parker & Simpson 25g Pouch TOBACCO BLUE', case=False), 'brand'] = 'Parker&Simpson'
df_pr_com.loc[df_pr_com['brand'].str.contains('champion|legendary ruby|legendary|Champon ruby', case=False), 'brand'] = 'ChampionRuby'
df_pr_com.loc[df_pr_com['brand'].str.contains('Champion ruby|Champion legendary ruby', case=False), 'brand'] = 'ChampionRuby'
df_pr_com.loc[df_pr_com['brand'].str.contains('Tally-ho Papers|Tally ho|Tally-ho|papers|ho', case=False), 'brand'] = 'TallyHo'
df_pr_com.loc[df_pr_com['brand'].str.contains('JPS endless blue|JPS Superking|JPS abundant|JPS blue', case=False), 'brand'] = 'JPS'
df_pr_com.loc[df_pr_com['brand'].str.contains('JPS super king silver|JPS gold|Jps gold 25g pouch', case=False), 'brand'] = 'JPS'
df_pr_com.loc[df_pr_com['brand'].str.contains('JPS+crush ball blue|Jsp abundant gold 25 gram|JPS+ crushball blue', case=False), 'brand'] = 'JPS'
df_pr_com.loc[df_pr_com['brand'].str.contains('JPS 93mm Long blue 20’s|Carton jps red 40’s|JPS 25G RED', case=False), 'brand'] = 'JPS'
df_pr_com.loc[df_pr_com['brand'].str.contains('JPS+Crush ball Gold|jps eternal red 25g|JPS 25g eternal red', case=False), 'brand'] = 'JPS'
df_pr_com.loc[df_pr_com['brand'].str.contains('Blue balls|JPS Abundant gold tobacco 25gm.|JPS RYO pouch', case=False), 'brand'] = 'JPS'
df_pr_com.loc[df_pr_com['brand'].str.contains('jps tobacco|Jps crushball cigarettes|JPS RED, 23pk|Jps 25 gram', case=False), 'brand'] = 'JPS'
df_pr_com.loc[df_pr_com['brand'].str.contains('Large black balls in packets|Jps 25 gram pouch|Jps crushball +', case=False), 'brand'] = 'JPS'
df_pr_com.loc[df_pr_com['brand'].str.contains('Tobacco Pouch Plastic material|Drum 25g pouch|JPS red ryo', case=False), 'brand'] = 'JPS'
df_pr_com.loc[df_pr_com['brand'].str.contains('Jps silver + firm touch filter|JPS packet of 40s|Jps eternal red', case=False), 'brand'] = 'JPS'
df_pr_com.loc[df_pr_com['brand'].str.contains('JPS John Player Special rolling tobacco|JPS+Crush ball Gold', case=False), 'brand'] = 'JPS'
df_pr_com.loc[df_pr_com['brand'].str.contains('JPS 93mm Long- Blue (1H13AQB 1624)|Jps red 25g pouch', case=False), 'brand'] = 'JPS'
df_pr_com.loc[df_pr_com['brand'].str.contains('JPS 30s blue Firm touch filter|jps red + firm filter', case=False), 'brand'] = 'JPS'
df_pr_com.loc[df_pr_com['brand'].str.contains("Jps red 30's cigarette pack contaminated|Jps firm touch filter", case=False), 'brand'] = 'JPS'
df_pr_com.loc[df_pr_com['brand'].str.contains("JPS Red 30's packet.|JPS+ Crushball Red 20pack Cigarettes", case=False), 'brand'] = 'JPS'
df_pr_com.loc[df_pr_com['brand'].str.contains('Jps silver + firm touch filter|Super king 👑black', case=False), 'brand'] = 'JPS'
df_pr_com.loc[df_pr_com['brand'].str.contains("25g pouch jps red|jps eternal reds roll your own", case=False), 'brand'] = 'JPS'
df_pr_com.loc[df_pr_com['brand'].str.contains("jps select hand stripped blend|jps crushball blue 25's", case=False), 'brand'] = 'JPS'
df_pr_com['brand'] = df_pr_com['brand'].replace('25 gram pouch tobacco','JPS') 
df_pr_com['brand'] = df_pr_com['brand'].replace('jps absolute gold 25g pouch tobacco','JPS') 
df_pr_com['brand'] = df_pr_com['brand'].replace('Jps 93 mm long blue packet','JPS') 
df_pr_com['brand'] = df_pr_com['brand'].replace('jps red + firm filter','JPS') 
df_pr_com['brand'] = df_pr_com['brand'].replace('JPS+Crush ball Gold','JPS') 
df_pr_com['brand'] = df_pr_com['brand'].replace('JPS+ Crushball Red 20pack Cigarettes','JPS')
df_pr_com['brand'] = df_pr_com['brand'].replace('Jps silver + firm touch filter','JPS')
df_pr_com['brand'] = df_pr_com['brand'].replace('JPS 93mm Long- Blue (1H13AQB 1624)','JPS')
df_pr_com['brand'] = df_pr_com['brand'].replace('jps red + firm filter','JPS')
df_pr_com['brand'] = df_pr_com['brand'].replace('JPS (external red) roll your own tobacco','JPS')
df_pr_com['brand'] = df_pr_com['brand'].replace('JPS+ crushball blue','JPS')
df_pr_com.loc[df_pr_com['brand'].str.contains('L&B original silver|Lambert & butler|Lambert', case=False), 'brand']='L&BorPeterJacksonBlue'
df_pr_com.loc[df_pr_com['brand'].str.contains('Horizon blue tobacco|Horizon|Horizon blue', case=False), 'brand'] = 'HorizonBlue'
df_pr_com.loc[df_pr_com['brand'].str.contains('50g pouch of horizon blue', case=False), 'brand'] = 'HorizonBlue'
df_pr_com.loc[df_pr_com['brand'].str.contains('Peter Jackson blue|Peter Jackson', case=False), 'brand'] = 'L&BorPeterJacksonBlue'
df_pr_com.loc[df_pr_com['brand'].str.contains('Peter|Jackson|Jackson Blue', case=False), 'brand'] = 'L&BorPeterJacksonBlue'


# In[ ]:


#Rename all other brand titles as others
df_pr_com.brand[(df_pr_com.brand !='JPS')&(df_pr_com.brand !='L&BorPeterJacksonBlue')
                &(df_pr_com.brand !='WhiteOx')&(df_pr_com.brand !='HorizonBlue')&(df_pr_com.brand !='ChampionRuby')
                &(df_pr_com.brand !='TallyHo')&(df_pr_com.brand !='Riverstone')&(df_pr_com.brand !='Parker&Simpson')] = "Others"


# In[ ]:


df1=df_pr_com


# In[ ]:


df1.brand.value_counts()


# In[ ]:


#a = df1['brand'].unique()
#print(sorted(a))
mylist =  list(df1.brand.unique())
mylist


# In[ ]:


print('Number of rows in Dataframe: ', len(df1))


# In[ ]:


#We can see from the graph that there lies a major class imbalance 
#We have to deal with this when we are going to use our model for prediction
#Now we are doing multilabel prediction. 
fig, ax = plt.subplots(figsize=(10,8))

ax = sns.countplot(y='brand', 
                   data=df1, 
                   order=df1['brand'].value_counts().iloc[:60].index)

ax.set_title('NUMBER OF COMPLAINTS IN EACH BRAND',size=15)

# Setting labels
# Dealing with y-labels
ax.set_ylabel('COMPLAINT', rotation=0, size=14, labelpad=10)
              
# Dealing with x-labels
ax.set_xlabel('NUMBER OF COMPLAINTS', size=14)

sns.despine()
            
plt.savefig('freq_of_uncombined_class.png', transparency=True)


# In[ ]:


# Applying encoding to the PRODUCT column
df_pr_com['brand_ID'] = df_pr_com['brand'].factorize()[0] 

#.factorize[0] arranges the index of each encoded number accordingly to the 
# index of your categorical variables in the PRODUCT column


# Creates a dataframe of the PRODUCT to their respective PRODUCT_ID
category_id_df = df_pr_com[['brand', 'brand_ID']].drop_duplicates()


# Dictionaries for future use. Creating our cheatsheets for what each encoded label represents.
category_to_id = dict(category_id_df.values) # Creates a PRODUCT: PRODUCT_ID key-value pair
id_to_category = dict(category_id_df[['brand_ID', 'brand']].values)  # Creates a PRODUCT_ID: PRODUCT key-value pair

# New dataframe
df_pr_com.head(10) 


# In[ ]:


# We still get the same length as per the original df.
len(df_pr_com)


# In[ ]:


#Now that we have encoded our columns, time to move on to the next step -- cleaning the text data
#Save df here so we don't run into memory issues later and we can start from a new starting point further down the notebook

# # Pickling reduced dataframe
with open('df_pr_com.pickle', 'wb') as to_write:
     pickle.dump(df_pr_com, to_write)


# In[ ]:


# Loading Pickled DataFrame
with open('df_pr_com.pickle', 'rb') as to_read:
    df_pr_com = pickle.load(to_read)


# In[ ]:


# Reviewing our Loaded Dataframe
print(df_pr_com.info())
print('--------------------------------------------------------------------------------------')
print(df_pr_com.head().to_string())


# In[ ]:


# Looking at a sample text
sample_complaint = list(df_pr_com.complaints[:5])[4]

# Converting to a list for TfidfVectorizer to use
list_sample_complaint = []
list_sample_complaint.append(sample_complaint)
list_sample_complaint


# In[ ]:


# Observing what words are extracted from a TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf3 = TfidfVectorizer(stop_words='english')
check3 = tf_idf3.fit_transform(list_sample_complaint)

print(tf_idf3.get_feature_names())


# In[ ]:


########Train/StratifiedCV/Test split
#prepare the model for text pre-processing using Tfidfvecotirzer with the stop_words and such. As we have tested above, it also ignores #punctuation so cleans the text really well for the purposes of doing text classification


# In[ ]:


# Split the data into X and y data sets
X, y = df_pr_com.complaints, df_pr_com.brand_ID
print('X shape:', X.shape, 'y shape:', y.shape)


# In[ ]:


# Split the data into X and y data sets
X, y = df_pr_com.complaints, df_pr_com.brand_ID
print('X shape:', X.shape, 'y shape:', y.shape)

# For text classification, ALWAYS split data first before vectorizing for best practice
# This is to avoid having features (words) from the test data already being inside the training data
from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, 
                                                            test_size=0.2,   # 80% train/cv, 20% test
                                                            stratify=y,
                                                            random_state=seed)
print('X_train', X_train_val.shape)
print('y_train', y_train_val.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)


# In[ ]:


# Performing Text Pre-Processing

# Import tfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Text Preprocessing
# The text needs to be transformed to vectors so as the algorithms will be able make predictions. 
# In this case it will be used the Term Frequency – Inverse Document Frequency (TFIDF) weight 
# to evaluate how important A WORD is to A DOCUMENT in a COLLECTION OF DOCUMENTS.

# tfidf1 = 1-gram only. 
tfidf1 = TfidfVectorizer(sublinear_tf=True, # set to true to scale the term frequency in logarithmic scale.
                        min_df=5,
                        stop_words='english')

X_train_val_tfidf1 = tfidf1.fit_transform(X_train_val).toarray()
X_test_tfidf1 = tfidf1.transform(X_test)

# tfidf2 = unigram and bigram
tfidf2 = TfidfVectorizer(sublinear_tf=True, # set to true to scale the term frequency in logarithmic scale.
                        min_df=5, 
                        ngram_range=(1,2), # we consider unigrams and bigrams
                        stop_words='english')
X_train_val_tfidf2 = tfidf2.fit_transform(X_train_val).toarray()
X_test_tfidf2 = tfidf2.transform(X_test)


# # StratifiedKFold -> Split 5
# ## We now want to do stratified kfold to preserve the proportion of the category imbalances 
# # (number is split evenly from all the classes)

# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


# In[ ]:


print('1-gram number of (rows, features):', X_train_val_tfidf1.shape)


# In[ ]:


def metric_cv_stratified(model, X_train_val, y_train_val, n_splits, name):
    """
    Accepts a Model Object, converted X_train_val and y_train_val, n_splits, name
    and returns a dataframe with various cross-validated metric scores 
    over a stratified n_splits kfold for a multi-class classifier.
    """
    # Start timer
    import timeit
    start = timeit.default_timer()
    
    ### Computations below
    
    # StratifiedKFold
    ## We now want to do stratified kfold to preserve the proportion of the category imbalances 
    # (number is split evenly from all the classes)
    from sklearn.model_selection import StratifiedKFold  # incase user forgest to import
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Initializing Metrics
    accuracy = 0.0
    micro_f1 = 0.0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
#     roc_auc = 0.0    Not considering this score in this case

    # Storing metrics
    from sklearn.model_selection import cross_val_score  # incase user forgets to import
    accuracy = np.mean(cross_val_score(model, X_train_val, y_train_val, cv=kf, scoring='accuracy'))
#     micro_f1 = np.mean(cross_val_score(model, X_train_val, y_train_val, cv=kf, scoring='f1_micro'))
    macro_precision = np.mean(cross_val_score(model, X_train_val, y_train_val, cv=kf, scoring='precision_macro'))
    macro_recall = np.mean(cross_val_score(model, X_train_val, y_train_val, cv=kf, scoring='recall_macro'))
    macro_f1 = np.mean(cross_val_score(model, X_train_val, y_train_val, cv=kf, scoring='f1_macro'))
    weighted_precision = np.mean(cross_val_score(model, X_train_val, y_train_val, cv=kf, scoring='precision_weighted'))
    weighted_recall = np.mean(cross_val_score(model, X_train_val, y_train_val, cv=kf, scoring='recall_weighted'))
    weighted_f1 = np.mean(cross_val_score(model, X_train_val, y_train_val, cv=kf, scoring='f1_weighted'))
    
    # Stop timer
    stop = timeit.default_timer()
    elapsed_time = stop - start
    
    return pd.DataFrame({'Model'    : [name],
                         'Accuracy' : [accuracy],
#                          'Micro F1' : [micro_f1],
                         'Macro Precision': [macro_precision],
                         'Macro Recall'   : [macro_recall],
                         'Macro F1score'  : [macro_f1],
                         'Weighted Precision': [weighted_precision],
                         'Weighted Recall'   : [weighted_recall],
                         'Weighted F1'  : [weighted_f1],
                         'Time taken': [elapsed_time]  # timetaken: to be used for comparison later
                        })


# In[ ]:


# ## Data Science Story:
# # Testing on MultinomialNB first

# # Initialize Model Object
mnb = MultinomialNB()

results_cv_stratified_1gram = metric_cv_stratified(mnb, X_train_val_tfidf1, y_train_val, 5, 'MultinomialNB1')
results_cv_stratified_2gram = metric_cv_stratified(mnb, X_train_val_tfidf2, y_train_val, 5, 'MultinomialNB2')


# In[ ]:


results_cv_stratified_1gram
results_cv_stratified_2gram


# In[ ]:


## Testing on all Models using 1-gram 

# Initialize Model Object
gnb = GaussianNB()
mnb = MultinomialNB()
logit = LogisticRegression(random_state=seed)
randomforest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
linearsvc = LinearSVC()

## We do NOT want these two. They take FOREVER to train AND predict
knn = KNeighborsClassifier()  
decisiontree = DecisionTreeClassifier(random_state=seed)

# to concat all models
results_cv_straitified_1gram = pd.concat([metric_cv_stratified(mnb, X_train_val_tfidf1, y_train_val, 5, 'MultinomialNB1'),
                                           metric_cv_stratified(gnb, X_train_val_tfidf1, y_train_val, 5, 'GaussianNB1'),
                                           metric_cv_stratified(logit, X_train_val_tfidf1, y_train_val, 5, 'LogisticRegression1'),
                                           metric_cv_stratified(randomforest, X_train_val_tfidf1, y_train_val, 5, 'RandomForest1'),
                                           metric_cv_stratified(linearsvc, X_train_val_tfidf1, y_train_val, 5, 'LinearSVC1')
                                          ], axis=0).reset_index()


# In[ ]:


results_cv_straitified_1gram


# In[ ]:


# # Saving our results to avoid retraining thing
with open('results_cv_straitified_1gram_df.pickle', 'wb') as to_write:
        pickle.dump(results_cv_straitified_1gram, to_write)


# In[ ]:


# Loading our pickled results
with open('results_cv_straitified_1gram_df.pickle', 'rb') as to_read:
    results_cv_straitified_1gram = pickle.load(to_read)


# In[ ]:


#### Keep running into memory issues with 2-gram. Therefore, will not test it anymore
## Testing on all Models using 2-gram 

# # Initialize Model Object
gnb = GaussianNB()
mnb = MultinomialNB()
logit = LogisticRegression(random_state=seed)
knn = KNeighborsClassifier()
decisiontree = DecisionTreeClassifier(random_state=seed)
randomforest = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
linearsvc = LinearSVC()

# # to concat all models
results_cv_straitified_2gram = pd.concat([metric_cv_stratified(mnb, X_train_val_tfidf2, y_train_val, 5, 'MultinomialNB2'),
                                            metric_cv_stratified(gnb, X_train_val_tfidf2, y_train_val, 5, 'GaussianNB2'),
                                            metric_cv_stratified(logit, X_train_val_tfidf2, y_train_val, 5, 'LogisticRegression2'),
                                            metric_cv_stratified(randomforest, X_train_val_tfidf2, y_train_val, 5, 'RandomForest2'),
                                            metric_cv_stratified(linearsvc, X_train_val_tfidf2, y_train_val, 5, 'LinearSVC2')
                                           ], axis=0).reset_index()


# In[ ]:


results_cv_straitified_2gram


# In[ ]:


import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


def get_average_word2vec(complaints_lst, model, num_features=300):
    """
    Function to average the vectors in a list.
    Say a list contains 'flower' and 'leaf'. Then this function gives - model[flower] + model[leaf]/2
    - index2words gets the list of words in the model.
    - Gets the list of words that are contained in index2words (vectorized_lst) and 
      the number of those words (nwords).
    - Gets the average using these two and numpy.
    """
    #complaint_feature_vecs = np.zeros((len(complaints_lst),num_features), dtype="float32") #?used?
    index2word_set = set(model.wv.index2word)
    vectorized_lst = []
    vectorized_lst = [model[word] if word in index2word_set else np.zeros(num_features) for word in                       complaints_lst]    
    nwords = len(vectorized_lst)
    summed = np.sum(vectorized_lst, axis=0)
    averaged_vector = np.divide(summed, nwords)
    return averaged_vector


# In[ ]:


from gensim.test.utils import common_texts
from gensim.models import Word2Vec

model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")


# In[ ]:


def complaint_to_wordlist(review, remove_stopwords=False):
    """
    Convert a complaint to a list of words. Removal of stop words is optional.
    """
    # remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
    
    # convert to lower case and split at whitespace
    words = review_text.lower().split()
    
    # remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return words    # list of tokenized and cleaned words


# In[ ]:


# num_features refer to the dimensionality of the model you are using
# model refers to the trained word2vec/glove model
# words refer to the words in a single document/entry

def make_feature_vec(words, model, num_features):
    """
    Average the word vectors for a set of words
    """
    feature_vec = np.zeros((num_features,),  # creates a zero matrix of (num_features, )
                           dtype="float32")  # pre-initialize (for speed)
    
    # Initialize a counter for the number of words in a complaint
    nwords = 0.
    index2word_set = set(model.index2word)  # words known to the model

    
    # Loop over each word in the comment and, if it is in the model's vocabulary, add its feature vector to the total
    for word in words:   # for each word in the list of words
        if word in index2word_set:   # if each word is found in the words known to the model
            nwords = nwords + 1.     # add 1 to nwords
            feature_vec = np.add(feature_vec, model[word])   
    
    # Divide by the number of words to get the average 
    if nwords > 0:
        feature_vec = np.divide(feature_vec, nwords)
    
    return feature_vec


# In[ ]:


# complaints refers to the whole corpus you intend to put in. 
# Therefore you need to append all these info from your df into a list first

def get_avg_feature_vecs(complaints, model, num_features):
    """
    Calculate average feature vectors for ALL complaints
    """
    # Initialize a counter for indexing 
    counter = 0
    
    # pre-initialize (for speed)
    complaint_feature_vecs = np.zeros((len(complaints),num_features), dtype='float32')  
    
    for complaint in complaints: # each complaint is made up of tokenized/cleaned/stopwords removed words
        complaint_feature_vecs[counter] = make_feature_vec(complaint, model, num_features)
        counter = counter + 1
    return complaint_feature_vecs


# In[ ]:


results_compiled.columns


# In[ ]:


results_compiled = pd.concat([results_cv_straitified_1gram,
                           results_cv_straitified_2gram]).reset_index().drop(['level_0','index'],axis=1)


# In[ ]:


with open('results_compiled.pickle', 'wb') as to_write:
    pickle.dump(results_compiled, to_write)

results_compiled


# In[ ]:


# Sorting results to see which one gives us the best results
results_compiled_sorted_by_accuracy = results_compiled.sort_values(by='Accuracy', ascending=False)

with open('results_compiled_sorted_by_accuracy.pickle', 'wb') as to_write:
    pickle.dump(results_compiled_sorted_by_accuracy, to_write)

results_compiled_sorted_by_accuracy


# In[ ]:


# Retrieving the Model that provides the highest Accuracy
results_highest_accuracy = results_compiled[results_compiled.Accuracy == results_compiled.Accuracy.max()]

with open('results_highest_accuracy.pickle', 'wb') as to_write:
    pickle.dump(results_highest_accuracy, to_write)

results_highest_accuracy


# In[ ]:


# We see from the above that LogisiticRegression and LinearSVC on 1-gram gives pretty good results.
# But LogReg is the best. So let's go with that!

# Now let us do the final test!!
# Pretty decent accuracy, actually.
# I want to try 2-gram but my computer just simply won't be able to handle it


# In[ ]:


######Splitting to 80% Train and 20% unseen data
# Split the data into X and y data sets
X, y = df1.complaints, df1.brand
print('X shape:', X.shape, 'y shape:', y.shape)

# For text classification, ALWAYS split data first before vectorizing.
# This is because you don't want to cheat by having features (words) from the test data already being inside your train data
from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, 
                                                            test_size=0.2,   # 80% train/cv, 20% test
                                                            stratify=y,
                                                            random_state=seed)
print('X_train', X_train_val.shape)
print('y_train', y_train_val.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)


# In[ ]:


# tfidf1 = 1-gram only. 
tfidf1 = TfidfVectorizer(sublinear_tf=True, # set to true to scale the term frequency in logarithmic scale.
                        min_df=5,
                        stop_words='english')
X_train_val_tfidf1 = tfidf1.fit_transform(X_train_val).toarray()
X_test_tfidf1 = tfidf1.transform(X_test)


# In[ ]:


with open('fitted_tfidf_to_use.pickle', 'wb') as to_write:
    pickle.dump(tfidf1, to_write)


# In[ ]:


with open('fitted_tfidf_to_use.pickle', 'rb') as to_read:
    logit_finalized = pickle.load(to_read)


# In[ ]:


# Initializing our chosen logreg model
logit = LogisticRegression()

# Fitting our model
logit_finalized = logit.fit(X_train_val_tfidf1, y_train_val)

# Obtaining prediction
y_pred = logit_finalized.predict(X_test_tfidf1)


# In[ ]:


# Pickle trained Model for use in Flask App
with open('logit_finalized.pickle', 'wb') as to_write:
    pickle.dump(logit_finalized, to_write)


# In[ ]:


# Now we have a pickled trained model, we can use this for our flask app!
with open('logit_finalized.pickle', 'rb') as to_read:
    logit_finalized = pickle.load(to_read)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,8))

ax = sns.countplot(y='brand', 
                   data=df1, 
                   order=df1['brand'].value_counts().iloc[:60].index)

ax.set_title('NUMBER OF COMPLAINTS FOR EACH BRAND CATEGORY',size=15)

# Setting labels
# Dealing with y-labels
ax.set_ylabel('BRAND', rotation=0, labelpad=14, size=10)
              
# Dealing with x-labels
ax.set_xlabel('NUMBER OF COMPLAINTS', size=14)

sns.despine()

plt.savefig('freq_of_removed_classes_and_reduced_observations.png')
#lt.show();


# In[ ]:


cnt_pro = df1['brand'].value_counts()

plt.figure(figsize=(12,4))
sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('brand', fontsize=12)
plt.xticks(rotation=90)
plt.show();


# In[ ]:


# Create a function to calculate the error metrics, since we'll be doing this several times
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

def conf_matrix(actual, predicted):
    
    # Creates a dataframe of the brand to their respective brand_ID
    category_id_df = df1[['brand', 'brand_ID']]

    # Dictionaries for future use. Creating our cheatsheets for what each encoded label represents.
    category_to_id = dict(category_id_df.values) # Creates a brand: brand_ID key-value pair
    id_to_category = dict(category_id_df[['brand_ID', 'brand']].values)  # Creates a brand_ID: brand key-value pair

    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8,8))
    g= sns.heatmap(conf_mat, 
                   annot=True, annot_kws={"size":10},
                   cmap=plt.cm.Reds, square=True,
                   fmt='d',
                   xticklabels=category_id_df.brand.values, 
                   yticklabels=category_id_df.brand.values)
    
#     # Changing the size of the xticks and ytick labels
#     ax.set_yticklabels(g.get_yticklabels(), rotation=90, size=10);
#     ax.set_xticklabels(g.get_xticklabels(), size=10);
    
    # Changing axis orientation & setting titles
    ax.set_xlabel('Prediction', size=14)
    ax.set_ylabel('Actual', rotation=0, labelpad=40,size=14)

#    plt.title("CONFUSION MATRIX - {}\n".format(name), size=16);
    
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)


# In[ ]:


with open('results_highest_accuracy.pickle', 'rb') as to_read:
    results_highest_accuracy = pickle.load(to_read)

results_highest_accuracy

# Score model
print("---------------------------------------------------------")
print("LogisticRegression (1-gram) 80% TRAIN/20% TEST SCORES:")
print("---------------------------------------------------------")
print('\n')
print('Train/Cross-Validation Test Accuracy Score (also micro F1) for LogisticRegression: {:.4f}'.format(results_highest_accuracy.Accuracy.max()))
print('Final Test Accuracy Score (also micro F1) for LogisticRegression: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('\n')
print('Macro Precision Score for LogisticRegression: {:.4f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Macro Recall Score for LogisticRegression: {:.4f}'.format(recall_score(y_test, y_pred, average='macro')))
print('Macro F1 score = {:.4f}'.format(f1_score(y_test, y_pred, average='macro')))
print('\n')
print('Micro Precision Score for LogisticRegression: {:.4f}'.format(precision_score(y_test, y_pred, average='micro')))
print('Micro Recall Score for LogisticRegression: {:.4f}'.format(recall_score(y_test, y_pred, average='micro')))
print('Micro F1 score = {:.4f}'.format(f1_score(y_test, y_pred, average='micro')))
print('\n')
print('Weighted Precision Score for LogisticRegression: {:.4f}'.format(precision_score(y_test, y_pred, average='weighted')))
print('Weighted Recall Score for LogisticRegression: {:.4f}'.format(recall_score(y_test, y_pred, average='weighted')))
print('Weighted F1 score = {:.4f}'.format(f1_score(y_test, y_pred, average='weighted')))
print('\n')
print('Classification report for LogisticRegression (1-gram):\n {}'.format(classification_report(y_test, 
                                                                                             y_pred,
                                                                                             target_names=df1.brand.unique())))
print('Confusion Matrix for LogisticRegression (1-gram):\n'.format(conf_matrix(y_test, y_pred)))


# In[ ]:


conf_matrix(y_test, y_pred)
plt.savefig('confusion_matrix', transparent=True)
plt.show();


# In[ ]:


# These are the ONLY packages we need to use from this point forward!
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# In[ ]:


# Loading our saved models
# Consider training on the whole data instead now?
with open('fitted_tfidf_to_use.pickle', 'rb') as to_read:  # loading the fitted tfidf with our 80% trained data
    fitted_tfidf_to_use = pickle.load(to_read)
    
with open('logit_finalized.pickle', 'rb') as to_read:  # loading our model
    logit_finalized = pickle.load(to_read)


# In[ ]:


# Using our models

complaint = """
Have been a white ox smoker 4 over 20 years no"""

# After fitting the tfidf vectorizor, then you can do transforms!
new_vectorized_complaint = fitted_tfidf_to_use.transform([complaint])

# Fitting vectorized complaint into model
y_customized_prediction = logit_finalized.predict(new_vectorized_complaint)
y_customized_prediction[0]


# In[ ]:


# Using our models

complaint_2 = """
Bought a pouch of 25gm tabacco from tobacconis
"""

# After fitting the tfidf vectorizor, then you can do transforms!
new_vectorized_complaint_2 = fitted_tfidf_to_use.transform([complaint_2])

# Fitting vectorized complaint into model
y_customized_prediction = logit_finalized.predict(new_vectorized_complaint_2)
y_customized_prediction


# In[ ]:


# Using our models

complaint_3 = """
Bought a pkt of Jps Red 30s, code at btm 71672"""

# After fitting the tfidf vectorizor, then you can do transforms!
new_vectorized_complaint_3 = fitted_tfidf_to_use.transform([complaint_3])

# Fitting vectorized complaint into model
y_customized_prediction = logit_finalized.predict(new_vectorized_complaint_3)
y_customized_prediction


# In[ ]:


# Using our models

complaint_4 = """
jps blue crushball 25pk	After smoking jps blue crushball smokos for 4"""

# After fitting the tfidf vectorizor, then you can do transforms!
new_vectorized_complaint_4 = fitted_tfidf_to_use.transform([complaint_4])

# Fitting vectorized complaint into model
y_customized_prediction = logit_finalized.predict(new_vectorized_complaint_4)
y_customized_prediction


# In[ ]:


# tfidf2 = unigram and bigram
tfidf2 = TfidfVectorizer(sublinear_tf=True, # set to true to scale the term frequency in logarithmic scale.
                        min_df=5, 
                        ngram_range=(1,2), # we consider unigrams and bigrams
                        stop_words='english')

# We transform each complaint into a vector
features = tfidf2.fit_transform(df_pr_com.complaints).toarray()
# Labelling our data
labels = df_pr_com.brand_ID

print("Each of the %d complaints is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features.shape))


# In[ ]:


category_to_id


# In[ ]:


# Finding the three most correlated terms with each of the product categories
N = 3
for Brand, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf2.get_feature_names())[indices]
    
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("\n==> %s:" %(Brand))
    print("  * Most Correlated Unigrams are: %s" %(', '.join(unigrams[-N:])))
    print("  * Most Correlated Bigrams are: %s" %(', '.join(bigrams[-N:])))


# In[ ]:





# In[ ]:





# In[ ]:




