import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

risk_credit = pd.read_csv('credit_risk_dataset.csv')
x_credit_risk = pd.concat([risk_credit.iloc[:,0:8],risk_credit.iloc[:,9:12]], axis=1) # Variável preditiva
y_credit_risk = risk_credit.iloc[:,8]                                                 # Variável target
le_person_home_ownership = LabelEncoder()
le_loan_intent = LabelEncoder()
le_loan_amnt = LabelEncoder()
le_person_default_on_file = LabelEncoder()
x_credit_risk.iloc[:,2] = le_person_home_ownership.fit_transform(x_credit_risk.iloc[:,2])
x_credit_risk.iloc[:,4] = le_loan_intent.fit_transform(x_credit_risk.iloc[:,4])
x_credit_risk.iloc[:,5] = le_loan_amnt.fit_transform(x_credit_risk.iloc[:,5])
x_credit_risk.iloc[:,9] = le_person_default_on_file.fit_transform(x_credit_risk.iloc[:,9])
x_credit_risk['person_emp_length'] = x_credit_risk['person_emp_length'].fillna(x_credit_risk['person_emp_length'].mean()) # preenche os valores faltantes com a média dos valores presentes
x_credit_risk['loan_int_rate'] = x_credit_risk['loan_int_rate'].fillna(x_credit_risk['loan_int_rate'].mean()) # preenche os valores faltantes com a média dos valores presentes
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_credit_risk,y_credit_risk)
columns = x_credit_risk.columns
predict_input = pd.DataFrame([[27, 85000, 2, 3.0, 0, 0, 20000, 7.0, 0.23, 0, 5]],columns=columns) #exemplo de entrada
predict = rf_model.predict(predict_input)
print(predict)


# train_accuracy = rf_model.score(x_credit_risk, y_credit_risk)                   
# print(f"Acuracia nos dados de treinamento: {train_accuracy * 100:.2f}%")        
# baseline = y_credit_risk.value_counts(normalize=True).max()                      #checar a acuracia na base de treinamento
# print(f"Baseline (classe majoritaria): {baseline * 100:.2f}%")                  
                                                                                