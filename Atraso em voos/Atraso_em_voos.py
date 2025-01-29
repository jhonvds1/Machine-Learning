import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error




data_delays = pd.read_csv('Airline_Delay_Cause.csv')
pd.set_option('display.max_columns', None)
data_delays.dropna(inplace=True) #Apaga as linhas com valores faltantes, quantidade de dados muito grande, levaria muito tempo substituir pela média e não seria necessário.
plt.figure(figsize=(10,6))
sns.histplot(data_delays['arr_delay'], kde=True)
plt.xlabel('Atraso na chegada')
plt.ylabel('Contagem')
plt.title('Distribuição dos Atrasos na Chegada')
plt.show()


aeroportos_atrasados = data_delays.groupby('carrier_name')['arr_delay'].mean().sort_values(ascending=False)
aeroportos_atrasados.plot(kind='bar', figsize=(12,6))
plt.title('Aeroportos com mais atraso')
plt.xlabel('Aeroportos')
plt.show()

X = data_delays[['year', 'month', 'carrier', 'airport', 'arr_flights', 'arr_del15',
                 'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct', 
                 'arr_cancelled', 'arr_diverted']]
y = data_delays['arr_delay']
label_encoder = LabelEncoder()  # Codificando variáveis categóricas
X.loc[:,'carrier'] = label_encoder.fit_transform(X['carrier'])
X.loc[:,'airport'] = label_encoder.fit_transform(X['airport'])
scaler = StandardScaler() # Normalizando as variáveis numéricas
X.loc[:,['arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 
   'late_aircraft_ct', 'arr_cancelled', 'arr_diverted']] = scaler.fit_transform(
    X.loc[:,['arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 
        'late_aircraft_ct', 'arr_cancelled', 'arr_diverted']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Dividindo em treino e teste
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Erro quadratico medio: {mse:.2f}')




# year: O ano dos dados.
# month: O mês dos dados.
# carrier: Código da transportadora.
# carrier_name: Nome da transportadora.
# airport: Código do aeroporto.
# airport_name: Nome do aeroporto.
# arr_flights: Número de voos chegando.
# arr_del15: Número de voos atrasados ​​em 15 minutos ou mais.
# carrier_ct: Contagem da transportadora (atraso devido à transportadora).
# weather_ct: Contagem do clima (atraso devido ao clima).
# nas_ct: Contagem do NAS (National Airspace System) (atraso devido ao NAS).
# security_ct: Contagem de segurança (atraso devido à segurança).
# late_aircraft_ct: Contagem de aeronaves atrasadas (atraso devido à chegada tardia da aeronave).
# arr_cancelled: Número de voos cancelados.
# arr_diverted: Número de voos desviados.
# arr_delay: Atraso total na chegada.
# carrier_delay: Atraso atribuído à transportadora.
# weather_delay: Atraso atribuído ao clima.
# nas_delay: Atraso atribuído ao NAS.
# security_delay: Atraso atribuído à segurança.
# late_aircraft_delay: Atraso atribuído à chegada tardia da aeronave.








