#!/usr/bin/env python
# coding: utf-8

# **Recuerda que una vez abierto, Da clic en "Copiar en Drive", de lo contrario no podras almacenar tu progreso**
# 
# Nota: no olvide ir ejecutando las celdas de código de arriba hacia abajo para que no tenga errores de importación de librerías o por falta de definición de variables.

# In[ ]:


#configuración del laboratorio
# Ejecuta esta celda!
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
#for local 
#import sys ; sys.path.append('../commons/utils/')
get_ipython().system('wget https://raw.githubusercontent.com/jdariasl/ML_2020/master/Labs/commons/utils/general.py -O general.py --no-cache')
from general import configure_lab3
configure_lab3()
from lab3 import *
GRADER = part_2()


# # Laboratorio 3 - Parte 2. Comparación de metodos basados en árboles

# A continuación se leen los datos de un problema de clasificación. El problema corresponde a la clasifiación de dígitos escritos a mano, el cual fue abordado en el laboratorio anterior. Usaremos únicamente 4 de las 10 clases disponibles, usando solo los números pares. Los datos fueron preprocesados para reducir el número de características. La técnica usada será analizada más adelante en el curso.

# In[ ]:


digits = load_digits(n_class=10)
#--------- preprocesamiento usando solo los pares--------------------
pca = PCA(0.99, whiten=True)
data = pca.fit_transform(digits.data[np.isin(digits.target, [2,4,6,8,10])])
#---------- Datos a usar ----------------------
x = data
y =  digits.target[np.isin(digits.target, [2,4,6,8,10])]


# ### Ejercicio 1 Experimentos con Arboles de decisión

# Debe consultar todo lo relacionado con la creación, entrenamiento y uso en predicción de este modelo usando la librería scikit-learn. En este enlace, se puede leer la documentación http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html.
# 
# En el notebook, ya se encuentra cargada la libreria:
# 
# ```python
# from sklearn.tree import DecisionTreeClassifier
# ```
# 
# Las siguientes preguntas abiertas, buscan verificar que se está haciendo un contraste con la librería y la teoría, por lo tanto procura incluir conceptos asociados y **NO** solo enfocarse en las descripciones de la documentación.

# In[ ]:


#@title Pregunta Abierta
#@markdown  Entre el parametro max_features y max_depth definidos en la librería, ¿cual es un criterio de parada de crecimiento de los arboles? Explique en sus palabras
respuesta_1 = "" #@param {type:"string"}


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿Cuál es la diferencia entre min_samples_leaf y min_samples_split? Y explique la diferencia fundamental de estos dos con respecto a min_impurity_decrease.
respuesta_2 = "" #@param {type:"string"}


# En la siguiente celda se define una simulación para entrenar y validar un modelo usando los datos previamente cargados. Complete el código para usar como modelo de predicción un arbol de decisión.
# 
# 
# <b>Note</b> que existe una clase para modelos de clasificación y otra para modelos de regresión:
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
# 
# Vamos a tener en cuenta lo siguiente:
# 1. dentro del código, ya se encuentra sugerida la metodologia de validación
# 2. la función va aceptar un parametro booleano, dependiendo del valor de esta parametro, se ejecutara normalización de los datos.
# 3. **hacer uso explicito del nombre del parametro que se va usar**, por ejemplo, si se requeire asignar el parametro `max_features`  debemos llamar la libreria de esta manera: `DecisionTreeClassifier(max_features = 'auto')`
# 4. Vamos a configurar el arbol con la medida de impureza Gini

# In[ ]:


#ejercicio de código
def experimentar_dt( X, Y, depths,normalize):
    """funcion que realiza experimentos de arboles de decision
    X: matriz con las caractersiticas
    Y: matriz de numpy con etiquetas
    depths: list[int] lista con la profundidad de arboles a experimentar
    normalize bool: indica si se aplica normalización a los datos
    retorna: dataframe con:
        - profunidad de los arboles
        - eficiencia de entrenamiento
        - desviacion de estandar eficiencia de entrenamiento
        - eficiencia de prueba
        - desviacion estandar eficiencia de prueba
    """
    folds = 4
    skf = StratifiedKFold(n_splits=folds)
    resultados = pd.DataFrame()
    idx = 0
    for depth in depths:
        ## para almacenar los errores intermedios
        EficienciaTrain = []
        EficienciaVal = []
        for train, test in skf.split(X, Y):
            Xtrain = X[train,:]
            Ytrain = Y[train]
            Xtest = X[test,:]
            Ytest = Y[test]
            #Normalizamos los datos
            # si la bandera esta en True
            if normalize:
                scaler = StandardScaler()
                scaler.fit(Xtrain)
                Xtrain= scaler.transform(Xtrain)
                Xtest = scaler.transform(Xtest)
            #Haga el llamado a la función para crear y entrenar el modelo usando los datos de entrenamiento
            modelo = 
            modelo
            #predecir muestras de entrenamiento
            Ytrain_pred = modelo
            #predecir muestras de pruebas
            Yest = modelo
            #Evaluamos las predicciones del modelo con los datos de test
            EficienciaTrain.append(np.mean(Ytrain_pred.ravel() == Ytrain.ravel()))
            EficienciaVal.append(np.mean(Yest.ravel() == Ytest.ravel()))

        resultados.loc[idx,'profunidad del arbol'] = depth
        resultados.loc[idx,'eficiencia de entrenamiento'] = 
        resultados.loc[idx,'desviacion estandar entrenamiento'] = 
        resultados.loc[idx,'eficiencia de prueba'] =
        resultados.loc[idx,'desviacion estandar prueba'] = 
        idx= idx +1
        
    return (resultados)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio1", experimentar_dt)


# In[ ]:


# Realiza los experimentos sin normalizacion
depths = [5,10,20,30,50]
resultados_dt_no_norm = experimentar_dt(X = x , Y = y, depths = , normalize=)
resultados_dt_no_norm


# In[ ]:


# Realiza los experimentos con normalizacion
depths = [5,10,20,30,50]
resultados_dt_no_norm = experimentar_dt(X = x , Y = y, depths = , normalize=)
resultados_dt_no_norm


# In[ ]:


#@title Pregunta Abierta
#@markdown  ¿Tiene algún efecto la normalización o estándarización de las variables en el desempeño del modelo de árboles de decisión? Justifique.   
respuesta_3 = "" #@param {type:"string"}


# ### Ejercicio 2 Experimentos con Random Forest
# 
# En la siguiente celda se define una simulación para entrenar y validar un modelo usando los datos previamente cargados. Complete el código para usar como modelo de predicción un Random Forest. Debe consultar todo lo relacionado con la creación, entrenamiento y uso en predicción de este modelo usando la librería scikit-learn. Consultar aquí: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html.
# 
# En el notebook, ya se encuentra cargada la libreria:
# 
# ```python
# from sklearn.ensemble import RandomForestClassifier
# 
# ```
# 
# <b>Note</b> que al igual que en el caso anterior, existe una clase para modelos de clasificación y otra para modelos de regresión: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# 
# **Recordar hacer uso explicito del nombre del parametro que se va usar**, por ejemplo, si se requiere asignar el parametro `criterion`  debemos llamar la libreria  de esta manera: `RandomForestClassifier(criterion = 'gini')`.
# 
# Para nuestros experimentos vamos a configurar el RF para que el mínimo de muestras para considerar un nodo como hoja sea de 3.

# In[ ]:


#ejercicio de código
def experimentar_rf(X, Y, num_trees,numero_de_variables):
    """funcion que realiza experimentos de random forest
    X: matriz con las caractersiticas
    Y: matriz de numpy con etiquetas
    num_trees: list[int]: lista con el número de arboles usado para el RF
    numero_de_variables list[int]: lista con variables para la selección del mejor umbral en cada nodo 
    retorna: dataframe con:
        -  numero de arboles usados
        -  variables para la selección del mejor umbral
        - eficiencia de entrenamiento
        - desviacion de estandar eficiencia de entrenamiento
        - eficiencia de prueba
        - desviacion estandar eficiencia de prueba
    """
    folds = 4
    skf = StratifiedKFold(n_splits=folds)
    resultados = pd.DataFrame()
    idx = 0
    for trees in num_trees:
        for num_variables in numero_de_variables:
            ## para almacenar los errores intermedios
            EficienciaTrain = []
            EficienciaVal = []
            for train, test in skf.split(X, Y):
                Xtrain = X[train,:]
                Ytrain = Y[train]
                Xtest = X[test,:]
                Ytest = Y[test]
                #Haga el llamado a la función para crear y entrenar el modelo usando los datos de entrenamiento
                modelo =
                modelo
                #predecir muestras de entrenamiento
                Ytrain_pred = modelo
                #predecir muestras de pruebas
                Yest = modelo
                #Evaluamos las predicciones del modelo con los datos de test
                EficienciaTrain.append(np.mean(Ytrain_pred.ravel() == Ytrain.ravel()))
                EficienciaVal.append(np.mean(Yest.ravel() == Ytest.ravel()))

            resultados.loc[idx,'número de arboles'] = trees
            resultados.loc[idx,'variables para la selección del mejor umbral'] = num_variables
            resultados.loc[idx,'eficiencia de entrenamiento'] = 
            resultados.loc[idx,'desviacion estandar entrenamiento'] =
            resultados.loc[idx,'eficiencia de prueba'] =
            resultados.loc[idx,'desviacion estandar prueba'] = 
            idx= idx +1
        print(f"termina para {trees} arboles")
        
    return (resultados)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio2", experimentar_rf)


# Una vez completado el código realice los experimentos necesarios para llenar la siguiente tabla:

# In[ ]:


arboles = [5,10,20,50,100, 150]
variables_seleccion = [5,20,40]
resultados_rf = experimentar_rf(X=x, Y=y, num_trees =  ,numero_de_variables = )
resultados_rf


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿qué caractersitica tiene en especial los Random forest entrenados cuando la cantidad de variables para la selección del mejor umbral es igual a 40?  *desde el punto de vista teorico no de los resultados**
respuesta_4 = "" #@param {type:"string"}


# Vamos a comparar los resultados del RF y con el DT.

# In[ ]:


print("diferencia promedio entre entrenamiento y prueba del DT", 
      resultados_dt_norm['eficiencia de entrenamiento'].mean()-resultados_dt_norm['eficiencia de prueba'].mean())

print("diferencia promedio entre entrenamiento y prueba del RF", 
      resultados_rf['eficiencia de entrenamiento'].mean()-resultados_rf['eficiencia de prueba'].mean())


# de acuerdo a estos resultados, que modelo tuvo un mayor sobre entrenamiento?, ten presente esa diferencia para responder la siguiente pregunta abierta.

# In[ ]:


#@title Pregunta Abierta
#@markdown ¿esperaba la diferencia que se observa entre las eficiencias entre entrenamiento y pruebas para el Random forest y el arbol de decisón? justifique 
respuesta_5 = "" #@param {type:"string"}


# ### Ejercicio 3 Experimentos con Gradient Boosted Trees
# 
# En la siguiente celda se define una simulación para entrenar y validar un modelo usando los datos previamente cargados. Complete el código para usar como modelo de predicción un Gradient boosted Tree. Debe consultar todo lo relacionado con la creación, entrenamiento y uso en predicción de este modelo usando la librería scikit-learn. Consultar aquí: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
# 
# En el notebook, ya se encuentra cargada la libreria:
# 
# ```python
# from sklearn.ensemble import GradientBoostingClassifier
# 
# ```
# 
# Debemos configurar el árbol con un mínimo de tres (3) muestras para considerar una división de un nodo.
# 
# <b>Note</b> que al igual que en el caso anterior, existe una clase para modelos de clasificación y otra para modelos de regresión: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor
# 
# **Recordar hacer uso explicito del nombre del parametro que se va usar**, por ejemplo, si se requeire asignar el parametro ` loss`  debemos llamar la libreria  de esta manera: `GradientBoostingClassifier(loss = 'deviance')`

# In[ ]:


#@title Pregunta Abierta
#@markdown ¿qué diferencia existe entre el entrenamiento de un Random Forest y el gradient boosting de árboles?
respuesta_6 = "" #@param {type:"string"}


# In[ ]:


#ejercicio de código
def experimentar_gbt(num_trees, X, Y):
    """funcion que realiza experimentos de arboles de decision
    num_trees: list[int] lista con el número de arboles usado para el RF
    X: matriz con las caractersiticas
    Y: matriz de numpy con etiquetas
    retorna: dataframe con:
        - numero de arboles usados
        - eficiencia de entrenamiento
        - desviacion de estandar eficiencia de entrenamiento
        - eficiencia de prueba
        - desviacion estandar eficiencia de prueba
    """
    folds = 4
    skf = StratifiedKFold(n_splits=folds)
    resultados = pd.DataFrame()
    idx = 0
    for trees in num_trees:
        ## para almacenar los errores intermedios
        EficienciaTrain = []
        EficienciaVal = []
        for train, test in skf.split(X, Y):
            Xtrain = X[train,:]
            Ytrain = Y[train]
            Xtest = X[test,:]
            Ytest = Y[test]
            #Haga el llamado a la función para crear y entrenar el modelo usando los datos de entrenamiento
            modelo=
            modelo
            #predecir muestras de entrenamiento
            Ytrain_pred =  
            #predecir muestras de pruebas
            Yest = 
            #Evaluamos las predicciones del modelo con los datos de test
            EficienciaTrain.append(np.mean(Ytrain_pred.ravel() == Ytrain.ravel()))
            EficienciaVal.append(np.mean(Yest.ravel() == Ytest.ravel()))

        resultados.loc[idx,'número de arboles'] = trees
        resultados.loc[idx,'eficiencia de entrenamiento'] = 
        resultados.loc[idx,'desviacion estandar entrenamiento'] = 
        resultados.loc[idx,'eficiencia de prueba'] =
        resultados.loc[idx,'desviacion estandar prueba'] = 
        idx= idx +1
        
    return (resultados)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio3", experimentar_gbt)


# In[ ]:


# ejecuta para realizar los experimentos
arboles = [5,10,20,50,100, 150]
resultados_gbt = experimentar_gbt(arboles, x, y)
resultados_gbt


# Vamos a graficar la eficiencia para el RF y el GBT en función del número de arboles.

# In[ ]:


# se crea un df para agrupar los resultados
# y graficar las diferencias entre el GBT y el RF
rf_res = resultados_rf.groupby("número de arboles")['eficiencia de prueba'].mean().reset_index()
rf_res['Tipo'] = 'RF'
gbt_res = resultados_gbt.groupby("número de arboles")['eficiencia de prueba'].mean().reset_index()
gbt_res['Tipo'] = 'GBT'
data_to_plot= pd.concat([rf_res, gbt_res], ignore_index=True)
sns.relplot(data=data_to_plot, x= 'número de arboles', y = 'eficiencia de prueba', hue = 'Tipo', kind='line', aspect=1.5,height=3)


# ### Ejercicio 4 Tiempo de entrenamiento del RF y GBT
# 
# En nuestro último experimento, vamos a evaluar la influencia de las parametros del RF y del GBT en el tiempo de entrenamiento. 
# 
# Para ello vamos a crear una función para medir el tiempo de entrenamiento usando la instrucción `time.process_time()`.
# 
# Vamos crear la función, para poder evaluar la influencia de:
# 1. número de arboles
# 2. cantidad de variables a analizar por nodo
# 
# En el entrenamiento del RF y del GBT. 
# 
# **Notar**  
# 1. No vamos a dividir el conjunto, ya que el objetivo es evaluar el tiempo de entrenamiento y no la eficiencias del modelo
# 2. No calculamos las prediciones
# 3. **Recordar hacer uso explicito del nombre del parametro que se va usar**, por ejemplo, si se requeire asignar el parametro `criterion`  debemos llamar la libreria  de esta manera: `RandomForestClassifier(criterion = 'gini')`

# In[ ]:


def time_rf_gbt_training(X, Y, num_trees, numero_de_variables, metodo):
    """funcion que realiza experimentos, para determinar la influencia
    del numero de arboles y de caracteristicas en el tiempo de entrenamiento
    del RF
    X: conjunto de datos para realizar los experimentos
    Y: conjunto de etiquetas de clase
    num_trees: List[int] lista con el número de arboles a evaluar
    num_variables: List[int] lista con el número variables a evaluar
    metodo: 'rf' o 'gbt', de acuerdo a esto se verifica cual modelo a evaluar.
    retorna: dataframe con:
    - número de arboles
    - variables para la selección del mejor umbral
    - tiempo de entrenamiento (promedio)
    """
    resultados = pd.DataFrame()
    idx = 0
    
    for trees in num_trees:
        for variables in numero_de_variables:
            ## ejecutar 5 veces lo mismo
            ## para llegar a un tiempo más adecuado
            tiempos = []
            for i in range(5):
            ## llamar la funcion para inicar el conteo
                start 
                if metodo == 'rf':
                
                    modelo = 
                else:
                    modelo = 
                
                modelo
                ## obtener tiempo 
                end
                # append de la resta de fin y end
                tiempos.append()
            resultados.loc[idx,'número de arboles'] = trees
            resultados.loc[idx,'variables para la selección del mejor umbral'] = variables
            # obtenga el promedio
            resultados.loc[idx,'tiempo de entrenamiento'] = 
            resultados.loc[idx,'metodo'] = metodo
            idx = idx +1
    return(resultados)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio4", time_rf_gbt_training)


# Vamos a dejar fijo el número de variables en 20 y variar los árboles en: [5,10,15,25, 50], completa el código para ver la grafica

# In[ ]:


resultados_rf_time1 = time_rf_gbt_training(x, y, [], [], metodo = 'rf')
resultados_gbt_time1 = time_rf_gbt_training(x, y, [], [], metodo = 'gbt')

resultados_time = pd.concat([resultados_rf_time1, resultados_gbt_time1], ignore_index=True)

sns.relplot(data = resultados_time, x = 'número de arboles', y = 'tiempo de entrenamiento', hue = 'metodo', kind = 'line')


# Y por ultimo Vamos a dejar fijo el número de árboles en 20 y el número de varaibles [5,10,15,20,40], completa el código para ver la grafica

# In[ ]:


resultados_rf_time1 = time_rf_gbt_training(x, y, [], [], metodo = 'rf')
resultados_gbt_time1 = time_rf_gbt_training(x, y, [], [], metodo = 'gbt')

resultados_time = pd.concat([resultados_rf_time1, resultados_gbt_time1], ignore_index=True)

sns.relplot(data = resultados_time, x = 'variables para la selección del mejor umbral', y = 'tiempo de entrenamiento', hue = 'metodo', kind = 'line')


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿qué parametro de los evaluados tiene una mayor influencia en los tiempos de entrenamiento? ¿hay diferencia entre el RF y GBT? justifique
respuesta_7 = "" #@param {type:"string"}


# In[ ]:


GRADER.check_tests()


# In[ ]:


#@title Integrantes
codigo_integrante_1 ='' #@param {type:"string"}
codigo_integrante_2 = ''  #@param {type:"string"}


# ----
# esta linea de codigo va fallar, es de uso exclusivo de los profesores
# 

# In[ ]:


GRADER.grade()

