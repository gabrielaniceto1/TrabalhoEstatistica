import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

handle = "sanjeetsinghnaik/top-1000-highest-grossing-movies"
file_path = "Highest Holywood Grossing Movies.csv"

df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, handle, file_path)
print(df.head())


#normalizando as colunas chaves por serem de escalas muito distintas
#nesse caso, interessante para fazer graficos
df_normalizado = df.copy()

df_normalizado["Year"] = (
    (df["Year"] - df["Year"].min()) /
    (df["Year"].max() - df["Year"].min())
)

df_normalizado["World Wide Sales (in $)"] = (
    (df["World Wide Sales (in $)"] - df["World Wide Sales (in $)"].min()) /
    (df["World Wide Sales (in $)"].max() - df["World Wide Sales (in $)"].min())
)


#É possivel fazer essas metricas para qualquer coluna apenas trocando o que eu quero exibir

#Anos
print(f"Mediana dos anos: {df['Year'].median()}\nMédia dos anos: {df['Year'].mean()}\nModa dos anos: {df['Year'].mode()}")
print(f"Q1: {df['Year'].quantile(.25)}\nQ2: {df['Year'].quantile(.50)}\nQ3: {df['Year'].quantile(.75)}\nQ4: {df['Year'].quantile(1.00)}")
print(f"Amplitude: {df['Year'].max() - df['Year'].min()} anos")
print(f"Variância: {df['Year'].var()}")
print(f"Desvio padrão: {df['Year'].std()}")

#Vendas em dolares ao todo
print(
    f"Mediana das vendas mundiais: {df['World Wide Sales (in $)'].median()}\n"
    f"Média das vendas mundiais: {df['World Wide Sales (in $)'].mean()}\n"
    f"Moda das vendas mundiais:\n{df['World Wide Sales (in $)'].mode()}"
)

print(
    f"Q1: {df['World Wide Sales (in $)'].quantile(0.25)}\n"
    f"Q2: {df['World Wide Sales (in $)'].quantile(0.50)}\n"
    f"Q3: {df['World Wide Sales (in $)'].quantile(0.75)}\n"
    f"Q4: {df['World Wide Sales (in $)'].quantile(1.00)}"
)

print(f"Amplitude: {df['World Wide Sales (in $)'].max() - df['World Wide Sales (in $)'].min()}")
print(f"Variância: {df['World Wide Sales (in $)'].var()}")
print(f"Desvio padrão: {df['World Wide Sales (in $)'].std()}")

# pearson em relação aos dois. Não é bom se tem outliers
coeficientePearson, relevanciaPearson = pearsonr(df['Year'], df['World Wide Sales (in $)'])
print(f"Coeficiente de pearson: {coeficientePearson}\nRelevancia: {relevanciaPearson}")

#sperman. bom para outliers
coeficienteSperman, relevanciaSperman = spearmanr(df['Year'], df['World Wide Sales (in $)'])
print(f"Coeficiente de sperman: {coeficienteSperman}\nRelevancia: {relevanciaSperman}")

sns.boxplot(y=df_normalizado['Year'])
plt.title("Boxplot dos anos dos filmes")
plt.show()

sns.boxplot(y=df_normalizado['World Wide Sales (in $)'])
plt.title("Boxplot dos lucros dos filmes")
plt.show()