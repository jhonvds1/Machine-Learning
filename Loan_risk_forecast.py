import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


adult_data = pd.read_csv('adult/adult.data.csv')
x_adult_data = adult_data.iloc[:,:14]
y_adult_data = adult_data.iloc[:,14]
le_state_gov = LabelEncoder()
le_bachelors = LabelEncoder()
le_never_married = LabelEncoder()
le_adm_clerical = LabelEncoder()
le_not_in_family = LabelEncoder()
le_white = LabelEncoder()
le_male = LabelEncoder()
le_united_states = LabelEncoder()
x_adult_data.iloc[:,1] = le_state_gov.fit_transform(x_adult_data.iloc[:,1])
x_adult_data.iloc[:,3] = le_bachelors.fit_transform(x_adult_data.iloc[:,3])
x_adult_data.iloc[:,5] = le_never_married.fit_transform(x_adult_data.iloc[:,5])
x_adult_data.iloc[:,6] = le_adm_clerical.fit_transform(x_adult_data.iloc[:,6])
x_adult_data.iloc[:,7] = le_not_in_family.fit_transform(x_adult_data.iloc[:,7])
x_adult_data.iloc[:,8] = le_white.fit_transform(x_adult_data.iloc[:,8])
x_adult_data.iloc[:,9] = le_male.fit_transform(x_adult_data.iloc[:,9])
x_adult_data.iloc[:,13] = le_united_states.fit_transform(x_adult_data.iloc[:,13])
#Base de dados sem valores faltantes ou inconsistentes
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_adult_data,y_adult_data)
train_accuracy = rf_model.score(x_adult_data,y_adult_data)
print(f"Acuracia nos dados de treinamento: {train_accuracy * 100:.2f}%")  




# predict = rf_model.predict(predict_input)
# print(predict)










