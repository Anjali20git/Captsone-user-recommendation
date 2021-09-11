# import libraties
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


lr_pickle_model = joblib.load('./models/lr_model.pkl') 
user_final_rating = pd.read_pickle('./models/final_user_rating.pkl')
products_details = pd.read_pickle('./models/final_data_df.pkl')
#products_details = pd.read_csv('sample30.csv')
tfidf_vector = TfidfVectorizer(stop_words="english")

def RecommendProducts(user_input):
# This is the function where we use the recommendation system with the sentiment analysis model and get the top 5 products for recommendation 
    pro_df = user_final_rating.loc[user_input]
    pro_df = pro_df.reset_index()
    pro_df = pro_df.sort_values(by=user_input,ascending=False)[0:20]
    

    pro_reviews=dict()
    pro_review_per=dict()
    pro_df = pro_df.reset_index(drop = True)
    
    for pro_name in pro_df['index']:
        
        review_test = products_details[products_details['name']==pro_name].reviews_text
        pro_reviews[pro_name] = lr_pickle_model.predict(review_test)
        (unique, counts) = np.unique(pro_reviews[pro_name], return_counts=True)
        
        frequencies = np.asarray((unique, counts)).T
        
        if len(frequencies)==2: ### get the counts for positive and negative reviews
            pos_rev=frequencies[1][1]
            neg_rev=frequencies[0][1]
        else:
            if frequencies[0][0]=='Positive':
                pos_rev=frequencies[0][1]
                neg_rev=0
            else:
                pos_rev=0
                neg_rev=frequencies[0][1]
        total= pos_rev+neg_rev
      
        pro_review_per[pro_name] = pos_rev/total 
        

    pro_review_per_sorted=sorted(pro_review_per.items(), key=lambda x: x[1], reverse=True)
    #pro_review_per_sorted=list(zip(*pro_review_per_sorted))
   
    output=[t[0] for t in pro_review_per_sorted][:5]
    output=pd.DataFrame(output) #sort and get the 5 products
    
    return output
