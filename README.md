# Objetivos

Criar uma rede neural utilizando o Keras e a avaliar a mesma com a validação cruzada. O dataset utlizado pode ser encontrado aqui: 
 <a href="https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data">Breast Cancer Wisconsin</a>.

O dataset do presente projeto possui 32 atributos, que podem ser explorados no link acima. A ideia desse projeto visa a construção de redes neurais
e não a exploração dos dados em si.

# Etapas de Treinamento e Validação
 
 Abaixo todas as bibliotecas necessárias para o projeto:
 
```
# Lib para manipulação de dados
import pandas as pd

# Libs da Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Integrando Keras e Scikit-Learn 
from scikeras.wrappers import KerasClassifier

# Criando arquitetura da rede neural
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
```

Carregando o Dataset no Google Colab e excluindo colunas não interessantes:

```
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Datasets/breast_cancer.csv')

# Excluindo coluna ID e a última coluna (Umnamed 32)
df.drop(['id', 'Unnamed: 32'], axis = 1, inplace=True)
```

O alvo (target) precisa ser transformado em numérico. Onde a classe maligno (M) foi substituída por 1 e a classe benigna (B) em 0:

```
# Maligno (M) = 1
# Benigno (B) = 0

df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
```

Antes de realizarmos a separação entre treino e teste, obtêm-se o modelo baseline, que seria chutar os valores como tumor maligno
