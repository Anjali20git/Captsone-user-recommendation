from flask import Flask, jsonify,  request, render_template
#import sklearn.external.joblib as extjoblib
from Model import RecommendProducts

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html',Recommended_Products='Recommended Products')


@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        user_input = [x for x in request.form.values()]
        user_input = user_input[0]
        output=RecommendProducts(user_input)
        
        return render_template('index.html', Recommended_Products=[output.to_html(classes='output',header=False,index=False)])
    else :
        return render_template('index.html',Recommended_Products='Recommended Products')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)