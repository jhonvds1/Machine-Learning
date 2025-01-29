import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


data_water = pd.read_csv('water_potability.csv')
data_water.fillna(data_water.mean(), inplace=True)
sns.heatmap(data_water.corr(),annot=True,cmap='coolwarm')
sns.scatterplot(x='ph', y='Turbidity' , data=data_water)
plt.title('Relação entre pH e Turbidez')
plt.show()
x = data_water[['ph','Turbidity','Conductivity','Trihalomethanes', 'Organic_carbon','Sulfate','Chloramines','Solids','Hardness']]
y = data_water['Potability']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(f'Acuracia: {accuracy_score(y_test, y_pred):.2f}')
print(classification_report(y_test, y_pred))




