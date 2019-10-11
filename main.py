from flask import Flask, jsonify, request, render_template
from sklearn.externals import joblib
import numpy as np
import json

gbr = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api', methods=['POST'])
def make_prediction():
     data = request.get_json(force=True)
     #convert our json to a numpy array
     one_hot_data = input_to_one_hot(data)
     predict_request = gbr.predict([one_hot_data])
     output = [predict_request[0]]
     output = [float(np.round(output, 2))]
     print(data)
     return jsonify(results=output)


def input_to_one_hot(data):
    # initialize the target vector with zero values
    enc_input = np.zeros(32)
    # set the numerical input as they are
    enc_input[0] = data['Jamb']


    cols = ['Jamb', 'Bio_A', 'Bio_B', 'Bio_C', 'Chm_A', 'Chm_B', 'Chm_C', 'Eng_A',
       'Eng_B', 'Eng_C', 'Gender_F', 'Gender_M', 'Guardian_Father',
       'Guardian_Mother', 'Home_address_Rural', 'Home_address_Urban',
       'Internet_No', 'Internet_Yes', 'Maths_A', 'Maths_B', 'Maths_C',
       'Parent_status_Academician', 'Parent_status_Non_academician', 'Phy_A',
       'Phy_B', 'Phy_C', 'Place_of_stay_in_campus', 'Place_of_stay_off_campus',
       'Pstatus_A', 'Pstatus_T', 'Romantic_rel_No', 'Romantic_rel_Yes']

    Bio = ['Bio_A', 'Bio_B', 'Bio_C']

     # redefine the the user inout to match the column name
    redefinded_user_input = 'Bio_'+data['Bio']
    # search for the index in columns name list 
    Bio_column_index = cols.index(redefinded_user_input)
    enc_input[Bio_column_index] = 1

    Chm = ['Chm_A', 'Chm_B', 'Chm_C']
    redefinded_user_input = 'Chm_'+data['Chm']
    Chm_column_index = cols.index(redefinded_user_input)
    enc_input[Chm_column_index] = 1

    Eng = ['Eng_A','Eng_B', 'Eng_C']
    redefinded_user_input = 'Eng_'+data['Eng']
    Eng_column_index = cols.index(redefinded_user_input)
    enc_input[Eng_column_index] = 1

    Gender = ['Gender_F', 'Gender_M']
    redefinded_user_input = 'Gender_'+data['Gender']
    Gender_column_index = cols.index(redefinded_user_input)
    enc_input[Gender_column_index] = 1

    Guardian = ['Guardian_Father','Guardian_Mother']
    redefinded_user_input = 'Guardian_'+data['Guardian']
    Guardian_column_index = cols.index(redefinded_user_input)
    enc_input[Guardian_column_index] = 1

    Home_address = ['Home_address_Rural', 'Home_address_Urban']
    redefinded_user_input = 'Home_address_'+data['Home_address']
    Home_address_column_index = cols.index(redefinded_user_input)
    enc_input[Home_address_column_index] = 1

    Internet = ['Internet_No', 'Internet_Yes']
    redefinded_user_input = 'Internet_'+data['Internet']
    Internet_column_index = cols.index(redefinded_user_input)
    enc_input[Internet_column_index] = 1

    Maths = ['Maths_A', 'Maths_B', 'Maths_C']
    redefinded_user_input = 'Maths_'+data['Maths']
    Maths_column_index = cols.index(redefinded_user_input)
    enc_input[Maths_column_index] = 1

    Parent_status = ['Parent_status_Academician', 'Parent_status_Non_academician']
    redefinded_user_input = 'Parent_status_'+data['Parent_status']
    Parent_status_column_index = cols.index(redefinded_user_input)
    enc_input[Parent_status_column_index] = 1

    Phy = ['Phy_A','Phy_B', 'Phy_C']
    redefinded_user_input = 'Phy_'+data['Phy']
    Phy_column_index = cols.index(redefinded_user_input)
    enc_input[Phy_column_index] = 1

    Place_of_stay = ['Place_of_stay_in_campus', 'Place_of_stay_off_campus']
    redefinded_user_input = 'Place_of_stay_'+data['Place_of_stay']
    Place_of_stay_column_index = cols.index(redefinded_user_input)
    enc_input[Place_of_stay_column_index] = 1

    Pstatus = ['Pstatus_A', 'Pstatus_T']
    redefinded_user_input = 'Pstatus_'+data['Pstatus']
    Pstatus_column_index = cols.index(redefinded_user_input)
    enc_input[Pstatus_column_index] = 1

    Romantic_rel = ['Romantic_rel_No', 'Romantic_rel_Yes']
    redefinded_user_input = 'Romantic_rel_'+data['Romantic_rel']
    Romantic_rel_column_index = cols.index(redefinded_user_input)
    enc_input[Romantic_rel_column_index] = 1


    return enc_input



    
@app.route('/result',methods=['POST'])
def get_delay():
    result=request.form
    Jamb = result['Jamb']
    Bio = result['Bio']
    Chm = result['Chm']
    Eng = result['Eng']
    Gender = result['Gender']
    Guardian = result['Guardian']
    Home_address = result['Home_address']
    Internet = result['Internet']
    Maths = result['Maths']
    Parent_status = result['Parent_status']
    Phy = result['Phy']   
    Place_of_stay = result['Place_of_stay']
    Pstatus = result['Pstatus']
    Romantic_rel = result['Romantic_rel']

    user_input = {'Jamb':Jamb, 'Bio':Bio, 'Chm':Chm, 'Eng':Eng, 'Gender':Gender, 'Guardian':Guardian,
                'Home_address':Home_address, 'Internet':Internet, 'Maths':Maths, 
                'Parent_status':Parent_status, 'Phy':Phy, 'Place_of_stay':Place_of_stay,
                'Pstatus':Pstatus, 'Romantic_rel':Romantic_rel}
    
    print(user_input)
    a = input_to_one_hot(user_input)
    price_pred = gbr.predict([a])[0]
    if price_pred >= 5.0:
        return '<h1>Your Estimated CGPA is: 4.98</h1>'
    price_pred = round(price_pred, 2)    
    return json.dumps({'Your Estimated Cgpa is':price_pred});
    





if __name__ == '__main__':
    app.run(port=5000, debug=True)
