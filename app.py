from flask import Flask,request,render_template
from src.pipeline.prediction_pipeline import UserData,PredictPipeline
applicaiton=Flask(__name__)
app=applicaiton

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prediction",methods=['GET','POST'])
def prediction():
    if request.method =='GET':
        return render_template('prediction.html')
    else:
        user_data=UserData(
            ph=int(request.form.get('Ph')),
            Hardness=int(request.form.get('Hardness')),
            Solids=int(request.form.get('Solids')),
            Chloramines=int(request.form.get('Chloramines')),
            Sulfate=int(request.form.get('Sulfate')),
            Conductivity=int(request.form.get('Conductivity')),
            Organic_carbon=int(request.form.get('Organic_carbon')),
            Trihalomethanes=int(request.form.get('Trihalomethanes')),
            Turbidity=int(request.form.get('Turbidity'))
        )
        features=user_data.get_user_data_as_dataframe()
        print(features)
        print("Before Prediction")
        prediction_pipeline_object=PredictPipeline()
        print("Mid Prediction")
        
        prediction = prediction_pipeline_object.predict(features)
        print("after Prediction")

        return render_template("prediction.html",prediction=prediction[0])

if __name__=="__main__":
    app.run(host="0.0.0.0")   

        
