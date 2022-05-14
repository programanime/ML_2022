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
# for local 
# import sys ; sys.path.append('../commons/utils/')
get_ipython().system('wget https://raw.githubusercontent.com/jdariasl/ML_2020/master/Labs/commons/utils/general.py -O general.py --no-cache')
from general import configure_lab3
configure_lab3()
from lab3 import *
GRADER = part_1()


# # Laboratorio 3 - Parte 1. Comparación de metodos de clusterización
# 
# ### Ejercicio 1: Contextualización del problema
# 
# A continuación se leen los datos de un problema de clasificación. El problema corresponde a la clasificación de dígitos escritos a mano. Los datos fueron preprocesados para reducir el número de características y solo se van usar los digitos 0 al 4. La técnica usada será analizada más adelante en el curso. Al ejecutar la celda de código, tambien se podra visualizar una muestra de los datos usados.

# In[ ]:


digits = load_digits(n_class=5)
#--------- preprocesamiento--------------------
pca = PCA(0.99, whiten=True)
data = pca.fit_transform(digits.data)
#---------- Datos a usar ----------------------
x = data
y = digits.target

plot_digits(digits.data)


# Vamos realizar algunas pruebas para determinar que metodologia de validación es más adecuada. En el ejercicio de código, vamos usar las librerias de scikit-learn:
# 1. [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html?highlight=stratifiedkfold#sklearn.model_selection.StratifiedKFold)
# 2. [ShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html)
# 
# En la función debes seleccionar la función de sklearn adecuada para lograr el efecto deseado y completar el código donde es señalado

# In[ ]:


#ejercicio de código
def get_muestras_by_cv (method, X, Y):
    """función que devuelve el numero de muestras promedios
    dependiendo de la metodologia de validación (method).
    los conjuntos de datos deben dividirse **3 veces**

    method (int): solo puede tomar dos valores:
        1: se debe usar el metodo de validación Bootstrapping
        2: se debe usar el metodo valicacion cruzada estratificada
    X: matriz numpy con los valores de las caracteristicas
    Y: vector numpy con las etiquetas de con conjunto de caracteriticas X
    retorna: dataframe con las siguientes columnas:
        - etiqueta de clase
        - fold
        - # muestras en entrenamiento
    """
    idx = 0
    results = pd.DataFrame()
    
    if method == 1:
        #completa el codigo con el metodo adecuado
        cv = 
        for n, (train_index, test_index) in enumerate (cv.split()):
            y_train, y_test = Y[train_index], Y[test_index]
            
            # iterar sobre las etiquetas
            for label in np.unique(y_train):
                results.loc[idx, 'etiqueta de clase'] = label
                results.loc[idx, 'folds'] = n
                results.loc[idx, 'numero de muestras entrenamiento'] = (y_train == label).sum()
                idx = idx+1
        
    elif method == 2:
        #completa el codigo con el metodo adecuado
        cv = 
        for n, (train_index, test_index ) in enumerate (cv.split()):
            y_train, y_test = Y[train_index], Y[test_index]
            len(np.unique(y_train))
            # iterar sobre las etiquetas
            for label in np.unique(y_train):
            
                results.loc[idx, 'etiqueta de clase'] = label
                results.loc[idx, 'folds'] = n             
                results.loc[idx, 'numero de muestras entrenamiento'] = (y_train == label).sum()
                idx = idx+1
    else:
        print("el metodo no es valido!")
        
    return (results)
    


# In[ ]:


## la funcion que prueba tu implementacion
#ignora las graficas!!
GRADER.run_test("ejercicio1", get_muestras_by_cv)


# Aca vamos a crear un dataframe para observar el numero de muestras de entrenamiento logradas por cada uno de los metodos.

# In[ ]:


cv1 = get_muestras_by_cv(1,x,y)
cv1['validación'] = 'Bootstrapping'
cv2 = get_muestras_by_cv(2,x,y)
cv2['validación'] = 'Estratificada'
cvs = pd.concat([cv1, cv2], ignore_index=True)
cvs


# En la tabla es dificil analizar que esta pasando. Para interpretar mejor los resultados vamos a usar una grafica llamada [PointPlot](https://interactivechaos.com/es/manual/tutorial-de-seaborn/point-plot). Esta grafica de la libreria [Seaborn](https://seaborn.pydata.org/generated/seaborn.pointplot.html) nos permite ver la diferencia entre los metodos.
# 
# Debemos entender que representan los puntos y las barras.

# In[ ]:


display("distribución de clases")
sns.catplot(data= cvs , x = 'etiqueta de clase', y='numero de muestras entrenamiento', kind='point', hue='validación')


# En la grafica, las barras representan la variación del numero de muestras en cada fold/partición. Y el punto es el valor medio.

# In[ ]:


#@title Pregunta Abierta
#@markdown En sus palabras, usando el grafico anterior y su conocimiento, ¿por qué Boostrapping no es la metodologia más adecuada?
respuesta_1 = "" #@param {type:"string"}


# ### Ejercicio 2: Mezclas de gaussinas
# 
# En la siguiente celda vamos a definir una función que tome como entradas una matriz $X$ y una matriz $Y$, entrene un modelo GMM  (Modelo de mezclas gaussianas) por cada clase y retorne el listado de modelos para cada clase. Debe consultar todo lo relacionado con la creación, entrenamiento y uso en predicción de este modelo usando la librería scikit-learn. Consultar aquí: http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
# 
# En el notebook, ya se encuentra cargada la libreria que vamos a usar:
# 
# ```python
# from sklearn.mixture import GaussianMixture
# ```
# 

# In[ ]:


#ejercicio de código
def GMMClassifierTrain(M,tipo,X,Y):
    """metodo para entrenar un modelo de mezclas de gausianas
    M: Número de componentes
    tipo (str): Tipo de matriz de covarianza (los parametros que se
          aceptan son los nombres de la  misma libreria:
          'full', 'tied', 'diag', 'spherical')
    X: Matriz de caracteristicas
    Y : Matriz con las clases
   
    retorna: diccionario de con la estructura:
        {etiqueta_clase: modelo_gmm_entrenado}
    """ 
    GMMs = {}
    for cl in np.unique(Y):
        xclass = X[Y==cl, :]
        gmm  = 
        
    #Debe retornar un objeto que contenga todos los modelos entrenados  
    return GMMs 


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio2", GMMClassifierTrain)


# In[ ]:


#@title Pregunta Abierta
#@markdown En sus palabras, ¿por qué debemos entrenar un modelo por cada clase?
respuesta_2 = "" #@param {type:"string"}


# Adicionalmente construya una función que use los modelos entrenados en la función anterior para hacer la clasificación de un conjunto nuevo de muestras.
# 

# In[ ]:


#ejercicio de código
def GMMClassfierVal(GMMs,Xtest):
    """funcion que recibe los modelos GMM entrenados
        y realiza predicciones con base a cada salida
    GMMs: diccionario de con la estructura:
        {etiqueta_clase: modelo_gmm_entrenado}
    Xtest: matriz numpy con el cojunto de datos
        de las caracteristicas de prueba/test
    retorna: 
    - un numpy array con las estimaciones de de clase de cada 
        una de las muestras de Xtest
    - matriz las probabilidades(en log) de cada clase por cada
      muestra debe tener un shape [numero de muestras, numero de clases]
    """
    prob = np.zeros((Xtest.shape[0], len(GMMs)))
    
    #pista explora los metodos de la libreria, que metodo retorna probabilidades?
    for k,v in GMMs.items():
       
    
    # la etiqueta la asignas seleccionando le maximo de probabilidad
    Yest= 
    
    return Yest, prob


# In[ ]:


## la funcion que prueba tu implementacion
#ignora las graficas!!
GRADER.run_test("ejercicio3", GMMClassfierVal)


# ### Ejercicio 3: Experimentos
# Con el código completado, vamos a realizar experimentos. Complete la siguiente función para poder obtener los resultados de los experimentos. Es necesario:
# 1. retornar errores de entrenamiento y pruebas
# 2. retornar intervalos de confianza (desviacion estandar) para cada una de la configuración de experimentos
# 3. en el código, ya se sugiere la metodologia de validación (usando 3 folds) y se usa, como es usual el standard scaler para normalizar los datos.

# In[ ]:


#ejercicio de código
def experimentar(X, Y, covariance_types,num_components):
    """función que realiza experimentos del GMM
    
    X: matriz con las caractersiticas
    Y: matriz de numpy con etiquetas
    covariance_types: list[str] con las matrices de covarianza a probar
    num_components: list[int] con el numero de componente a probar
  
    retorna: dataframe con:
        - matriz de covarianza
        - numero de componentes
        - eficiencia de entrenamiento
        - desviacion de estandar eficiencia de entrenamiento
        - eficiencia de prueba
        - desviacion estandar eficiencia de prueba
    """
    # asignar los folds
    folds = 
    skf = StratifiedKFold(n_splits=folds)
    resultados = pd.DataFrame()
    idx = 0

    for cov_tipo in covariance_types:
        for M in num_components:
            ## para almacenar los errores intermedios
            EficienciaTrain = []
            EficienciaVal = []
            for train, test in skf.split(X, Y):
                Xtrain = X[train,:]
                Ytrain = Y[train]
                Xtest = X[test,:]
                Ytest = Y[test]
                #Normalizamos los datos
                scaler = StandardScaler()
                scaler.fit(Xtrain)
                Xtrain= scaler.transform(Xtrain)
                Xtest = scaler.transform(Xtest)
                #Haga el llamado a la función para crear y entrenar el modelo usando los datos de entrenamiento
                gmms = 
                #Validación se ignora la matriz de probabilidad
                Ytrain_pred, _ =
                Yest, _ = 
                #Evaluamos las predicciones del modelo con los datos de test
                EficienciaTrain.append(np.mean(Ytrain_pred.ravel() == Ytrain.ravel()))
                EficienciaVal.append(np.mean(Yest.ravel() == Ytest.ravel()))

            resultados.loc[idx,'matriz de covarianza'] = cov_tipo
            resultados.loc[idx,'numero de componentes'] = M
            resultados.loc[idx,'eficiencia de entrenamiento'] = 
            resultados.loc[idx,'desviacion estandar entrenamiento'] =
            resultados.loc[idx,'eficiencia de prueba'] =
            resultados.loc[idx,'desviacion estandar prueba'] =
            idx= idx +1
            print("termina para", cov_tipo, M)
        
    return (resultados)


# In[ ]:


## la funcion que prueba tu implementacion
# los prints del test los puedes ignorar (se hacen unos checks!)
GRADER.run_test("ejercicio4", experimentar)


# Ahora vamos a ejecutar nuestros experimentos:

# In[ ]:


matrices = ['full', 'tied', 'diag', 'spherical']
componentes = [1,2,3]
resultados = experimentar(x, y, matrices, componentes)


# In[ ]:


# para ver la tabla
resultados


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿a qué corresponde el parámetro init_params de la función GaussianMixture y cuál fue el valor usado en nuestros experimentos?
respuesta_3 = "" #@param {type:"string"}


# ### Ejercicio 4 Kmeans y experimentos
# 
# 
# Consultar todo lo relacionado al llamado del método KMeans de la librería scikit-learn en el siguiente enlace: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html. 
# 
# En el notebook, ya se encuentra cargada la libreria:
# 
# ```python
# from sklearn.cluster import KMeans
# ```
# 
# Vamos a resolver el mismo problema anterior pero con el K-means. La libreria nos ofrece suficiente utilidad para ya poder comenzar a realizar experimentos con ella. 
# 
# Por ellos vamos a implementar la función para realizar los experimentos. [Debemos configurar la inicialización del algoritmo](https://scikit-learn.org/stable/modules/clustering.html#k-means) usando 'k-means++'
# 
# 

# In[ ]:


#ejercicio de código
#Validamos el modelo
def experimentar_kmeans(numero_clusters,X, Y):
    """función que realiza experimentos de Kmeans
    numero_clusters: list[int] numero cluster para realizar experimentos
    X: matriz con las caractersiticas
    Y: matriz de numpy con etiquetas
    retorna: dataframe con:
        - numero_clusters
        - el error de entrenamiento
        - desviacion de estandar del error entrenamiento
        - error de prueba
        - desviacion estandar eror de prueba
    """
    folds = 4
    skf = StratifiedKFold(n_splits=folds)
    resultados = pd.DataFrame()
    idx = 0

    for n_cluster in numero_clusters:
        ## para almacenar los errores intermedios
        EficienciaTrain = []
        EficienciaVal = []
        for train, test in skf.split(X, Y):
            Xtrain = X[train,:]
            Ytrain = Y[train]
            Xtest = X[test,:]
            Ytest = Y[test]
            #Normalizamos los datos
            scaler = StandardScaler()
            scaler.fit(Xtrain)
            Xtrain= scaler.transform(Xtrain)
            Xtest = scaler.transform(Xtest)
            #verifica como pasar n_cluster a KMeans
            kmeans = 
            #¿que metodo debes llamar para entrenar kmeans?
            kmeans
            #predicciones en cada conjunto
            # ¿que metodo debes llamar para realizar preddiciones con kmeans?
            Ytrain_pred = kmeans
            Yest  = kmeans
            #Evaluamos las predicciones del modelo con los datos de test
            EficienciaTrain.append(np.mean(Ytrain_pred.ravel() == Ytrain.ravel()))
            EficienciaVal.append(np.mean(Yest.ravel() == Ytest.ravel()))

        resultados.loc[idx,'numero de clusters'] = n_cluster
        resultados.loc[idx,'eficiencia de entrenamiento'] = 
        resultados.loc[idx,'desviacion estandar entrenamiento'] =
        resultados.loc[idx,'eficiencia de prueba'] = 
        resultados.loc[idx,'desviacion estandar prueba'] =
        idx= idx +1
        print("termina para", n_cluster)
        
    return (resultados)


# In[ ]:


#@title Default title text
## la funcion que prueba tu implementacion
#ignora las graficas!!
GRADER.run_test("ejercicio5", experimentar_kmeans)


# In[ ]:


# ejecuta los experimentos y ve los resultados
resultados_kmeans = experimentar_kmeans([3,5,6,8],x, y)
resultados_kmeans


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿Qué es el método elbow y por qué en nuestro caso no es útil? Pista: tiene sentido buscar un número de grupos != 5 ?
respuesta_4 = "" #@param {type:"string"}


# tener presente que tipo de entrenamiento realiza el kmeans: ¿supervisado o no supervisado? 
# 
# Vamos a realizar una grafica usando un metodo que vamos a ver en futuros laboratorios.
# 
# La grafica se pueden entender como una representación en 2 dimensiones de nuestra base de datos (que tiene 40 variables). 
# 
# Cada punto representa uno de nuestros ejemplos.
# 
# **Recordar**: 
# 1. Completa el codigo, con el numero de clusters con la cual se obtuvo la mejor eficiencia en los experimentos.
# 2. Completa el codigo con los mejores parametros para GMM.

# In[ ]:


# estandarizacion

x_std = StandardScaler().fit_transform(x)

## Kmeans
numero_clusters = # aca el numero de cluster con la mejor eficiencia

predict_kmeans = KMeans(n_clusters=numero_clusters, init = 'k-means++').fit_predict(x_std)

## GMM. Reemplaza los valores de acuerdo a tus experimentos
gmms = GMMClassifierTrain(X=x_std,Y=y,M=,tipo= )
# ignoramos probabilidad
predict_gmm, _ = GMMClassfierVal(gmms, x_std)

#adicionamos etiquetas reales
real = pd.DataFrame(x[:, 0:2], columns=['PC1', 'PC2'])
real['clase'] = y
real['tipo de etiqueta'] = 'Real'

#adicionamos etiquetas de kmeans
kmeans = pd.DataFrame(x[:, 0:2], columns=['PC1', 'PC2'])
kmeans['clase'] = predict_kmeans
kmeans['tipo de etiqueta'] = 'Asignada por Kmeans'

#adicionamos etiquetas de gmms
gmm_df = pd.DataFrame(x[:, 0:2], columns=['PC1', 'PC2'])
gmm_df['clase'] = predict_gmm
gmm_df['tipo de etiqueta'] = 'Asignada por gmm'

# unimos y graficamos
data_to_plot = pd.concat([real, kmeans, gmm_df], ignore_index=True)
data_to_plot['clase'] = data_to_plot['clase'].astype('category')
g = sns.relplot(data = data_to_plot, x = 'PC1', y='PC2', hue= 'clase', col='tipo de etiqueta', style= 'clase',   markers = ['$0$', '$1$', '$2$', '$3$', '$4$'], s = 400, height = 10)


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿con base a los resultados que modelo es el más adecuado entre el k-means y el GMM? justifique.
respuesta_5 = "" #@param {type:"string"}


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

