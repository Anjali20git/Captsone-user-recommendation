from flask import Flask, jsonify,  request, render_template
#import sklearn.external.joblib as extjoblib
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
import xgboost as xgb
from xgboost import XGBClassifier
 
#print('before')
app = Flask(__name__)
#model_load = joblib.load("./models/xgclf.pkl")
model_load = xgb.Booster({'nthread': 4})  # init model
model_load.load_model('./models/xgclf.model')  # load data
user_final_rating1 = pd.read_csv('user_final_rating.csv')
user_final_rating = pd.read_csv('user_final_rating.csv',index_col=0)
products_details = pd.read_csv('sample30.csv')
tfidf_vector = TfidfVectorizer(stop_words="english")
#print('after')
#print(user_final_rating.head())


@app.route('/')
def home():
    return render_template('index.html',Recommended_Products='Recommended Products')


@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        user_input = [x for x in request.form.values()]
        user_input = user_input[0]
        #print('userinput',user_input)
        pro_df = user_final_rating.loc[user_input]
        #print('p1')
        #print(pro_df)
        #print(pro_df.index)
        pro_df = pro_df.reset_index()
        #print('p2')
        #print("prodf",pro_df)
        #print(pro_df.index)
        pro_df = pro_df.sort_values(by=user_input,ascending=False)[0:20]
        #print('after sort', pro_df)

        pro_reviews=dict()
        pro_review_per=dict()
        #pro_df = pro_df.reset_index(drop = True)
        #print(pro_df['index'].head())
        for pro_name in pro_df['index']:
            #print('pro_name', pro_name)
            review_test = products_details[products_details['name']==pro_name].reviews_text
            tfidf_vector.fit(review_test.values.tolist())

            x_test_new = tfidf_vector.transform(review_test.values.tolist())
            #print(type(x_test_new))
            x_test_new=xgb.DMatrix(x_test_new)
            pro_reviews[pro_name] = model_load.predict(x_test_new)
            (unique, counts) = np.unique(pro_reviews[pro_name], return_counts=True)
            #print((unique, counts))
            frequencies = np.asarray((unique, counts)).T
            #print(frequencies)
            if len(frequencies)==2:
                pos_rev=frequencies[1][1]
                neg_rev=frequencies[0][1]
            else:
                if frequencies[0][0]=='Positive':
                    pos_rev=frequencies[0][1]
                    neg_rev=0
                else:
                    pos_rev=0
                    neg_rev=frequencies[0][1]
            
            
            #print(pos_rev, neg_rev)
            total= pos_rev+neg_rev
            pro_review_per[pro_name] = pos_rev/total  
        pro_review_per_sorted=sorted(pro_review_per.items(), key=lambda x: x[1], reverse=True)
        pro_review_per_sorted=list(zip(*pro_review_per_sorted))
        output=list(pro_review_per_sorted[0])[:5]

       # output = model_load.predict(final_features).tolist()
        return render_template('index.html', Recommended_Products='Recommended Products {}'.format(output))
    else :
        return render_template('index.html',Recommended_Products='Recommended Products')

@app.route("/predict_api", methods=['POST', 'GET'])
def predict_api():
    print(" request.method :",request.method)
    if (request.method == 'POST'):
        data = request.get_json()
        return jsonify(model_load.predict([np.array(list(data.values()))]).tolist())
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)