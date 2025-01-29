import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv('train.csv') # Carrega o dataset
train = train.dropna(subset='Postal Code') # Exclui os valores nulos de Postal Code sem ter muito impacto, pois havia apenas 11 valores nulos
train['Order Date'] = pd.to_datetime(train['Order Date'],dayfirst=True) # Converte para o formato datetime
train['Month'] = train['Order Date'].dt.month # Cria uma coluna para o mês
monthly_sales = train.groupby('Month')['Sales'].sum() # Agrupa a coluna do mês com as vendas do mês e faz o somatório
plt.figure(figsize=(10,6))
monthly_sales.plot(kind='bar', color='skyblue')
plt.title('Vendas por Mês')
plt.xlabel('Mês')
plt.ylabel('Total de Vendas')              
plt.xticks(rotation=0)
plt.show()   # Exibe o gráfico com as configurações especificadas acima

category_sales = train.groupby('Category')['Sales'].sum().sort_values(ascending=False) # Agrupa a coluna categoria com as vendas na categoria e faz o somatório
plt.figure(figsize=(10,6))
category_sales.plot(kind='bar', color='lightcoral')
plt.title('Vendas por Categoria')
plt.xlabel('Categoria')
plt.ylabel('Total de Vendas')
plt.xticks(rotation=45)
plt.show()

state_sales = train.groupby('State')['Sales'].sum().sort_values(ascending=False) # Agrupa a coluna estados com as vendas em cada estado e faz o somatório
plt.figure(figsize=(10,6))
state_sales.plot(kind='bar',color='lightgreen')
plt.title('Vendas por Estado')
plt.xlabel('Estados')
plt.ylabel('Total de Vendas ')
plt.xticks(rotation=90)
plt.show()


        # Código abaixo funcional, é o mesmo código, mas com a definição de uma função para montar e exibir o gráfico
        # Para testar essa parte, somente precisa comentar o código acima começando na linha 11 até o final

# def plot_grouped_data(grouped_data, title, xlabel, ylabel, rotation=0, color="skyblue"):
#     """Função genérica para exibir gráficos agrupados."""
#     plt.figure(figsize=(10, 6))
#     grouped_data.plot(kind='bar', color=color)
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.xticks(rotation=rotation)
#     plt.show()



# # Vendas por Mês
# monthly_sales = train.groupby('Month')['Sales'].sum()
# plot_grouped_data(
#     monthly_sales,
#     title="Vendas por Mês",
#     xlabel="Mês",
#     ylabel="Total de Vendas",
#     rotation=0,
#     color="skyblue",
# )

# # Vendas por Categoria
# category_sales = train.groupby('Category')['Sales'].sum().sort_values(ascending=False)
# plot_grouped_data(
#     category_sales,
#     title="Vendas por Categoria",
#     xlabel="Categoria",
#     ylabel="Total de Vendas",
#     rotation=45,
#     color="lightcoral",
# )

# # Vendas por Estado
# state_sales = train.groupby('State')['Sales'].sum().sort_values(ascending=False)
# plot_grouped_data(
#     state_sales,
#     title="Vendas por Estado",
#     xlabel="Estado",
#     ylabel="Total de Vendas",
#     rotation=90,
#     color="lightgreen",
# )








