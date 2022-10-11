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

Antes de realizarmos a separação entre treino e teste, obtêm-se o modelo baseline, que seria chutar os valores como tumor benigno:

```
df['diagnosis'].value_counts()

0    357
1    212
Name: diagnosis, dtype: int64
```

Logo, se chutássemos apenas que todos os pacientes possuem câncer benigno, esse modelo baseline possuiria uma acurácia de 62.74%. A rede neural no mínimo precisa ser melhor do que este modelo baseline.

Abaixo, é realizada a separação dos atributos previsores e as classes, além, da separação entre treino e teste:

```
X = df.drop(['diagnosis'], axis = 1)
y = df['diagnosis']


# Separação entre treino e testes
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=99)
```

Irei utilizar a biblioteca scikeras, que é uma lib que faz a integração do keras e a scikit-learn. Para utilizar a mesma, é necessário criar uma função 
onde é montada a arquitetura da rede neural, para este caso, testei combinações do número de camadas ocultas e número de neurônios matemáticos por camada
para achar a melhor combinação dentre as duas e a melhor arquitetura foi uma rede neural densa com duas camadas ocultas, cada uma delas com 8 neurònios matemáticos. A camada de entrada possui 30 neurônios, já que há 30 atributos de entradas. A camada de saída possuí apenas um neurônio de saída, já que o problema em questão trata-se de uma classificação binária. Foi utilizada duas camadas de dropout, onde tal camada realiza uma regularização, onde alguns neurônios são desligados aleatoriamente, juntamente com suas conexões, durante o treinamento ape, foi escolhido o valor de 20 %, ou seja, 20 % dos neurônios de cada camada oculta serão resetados como zero. O motivo de utilizar a camada de droput
```
def nn_create():
  model = Sequential()
  model.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = X_treino.shape[1]))
  model.add(Dropout(0.2))
  model.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'random_uniform'))
  model.add(Dropout(0.2))
  model.add(Dense(units = 1, activation = 'sigmoid'))
  model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
  return model
  
  
clf = KerasClassifier(build_fn = nn_create, epochs = 30, batch_size = 10)
```
