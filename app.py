from flask import Flask, escape, request, render_template

import pickle

vector = pickle.load(open("vectorizer.pkl", 'rb')) 
model = pickle.load(open("finalized_model.pkl", 'rb'))

app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def home():
    if request.method == "POST":
        news = str(request.form['news']) 
        print(news)

        predict = model.predict(vector.transform([news]))[0]
        print(predict)
    
        return render_template("predicted.html", prediction_text="Probably your information is {}".format(predict))


    else:
        return render_template("prediction.html")

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
