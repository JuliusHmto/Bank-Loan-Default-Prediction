from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from src.loanDefault.config.configuration import ConfigurationManager
from src.loanDefault.components.data_transformation import DataTransformation
from src.loanDefault.pipeline.prediction import PredictionPipeline

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return render_template("train_success.html")


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            LoanNr_ChkDgt = int(request.form['LoanNr_ChkDgt'])
            Name = str(request.form['Name'])
            City = str(request.form['City'])
            State = str(request.form['State'])
            Zip = int(request.form['Zip'])
            Bank = str(request.form['Bank'])
            BankState = str(request.form['BankState'])
            NAICS = int(request.form['NAICS'])
            ApprovalDate = str(request.form['ApprovalDate'])
            ApprovalFY = str(request.form['ApprovalFY'])
            Term = int(request.form['Term'])
            NoEmp = int(request.form['NoEmp'])
            NewExist = int(float(request.form['NewExist']))
            CreateJob = int(request.form['CreateJob'])
            RetainedJob = int(request.form['RetainedJob'])
            FranchiseCode = int(request.form['FranchiseCode'])
            UrbanRural = int(request.form['UrbanRural'])
            RevLineCr = str(request.form['RevLineCr'])
            LowDoc = str(request.form['LowDoc'])
            ChgOffDate = str(request.form['ChgOffDate'])
            DisbursementDate = str(request.form['DisbursementDate'])
            DisbursementGross = float(request.form['DisbursementGross'])
            BalanceGross = float(request.form['BalanceGross'])
            ChgOffPrinGr = float(request.form['ChgOffPrinGr'])
            GrAppv = float(request.form['GrAppv'])
            SBA_Appv = float(request.form['SBA_Appv'])

       
         
            data = [LoanNr_ChkDgt, Name, City, State, Zip, Bank, BankState, NAICS, ApprovalDate, ApprovalFY, Term, NoEmp, NewExist, CreateJob, RetainedJob,
                    FranchiseCode, UrbanRural, RevLineCr, LowDoc, ChgOffDate, DisbursementDate, DisbursementGross, BalanceGross, ChgOffPrinGr, GrAppv, SBA_Appv]
            data = np.array(data).reshape(1, 26)

            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            
            clean_data = data_transformation.clean_and_preprocess_data(data)
            transformed_data = data_transformation.transform_data(clean_data, True)
            
            obj = PredictionPipeline()
            predict = obj.predict(transformed_data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return render_template("something_wrong.html")

    else:
        return render_template('index.html')


if __name__ == "__main__":
	app.run(host="0.0.0.0", port = 8080, debug=True)
	# app.run(host="0.0.0.0", port = 8080)