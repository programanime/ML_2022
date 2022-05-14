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
in_colab = True


# In[ ]:


import os
if not in_colab:
    import sys ; sys.path.append('../commons/utils/'); sys.path.append('../commons/utils/data')
else: 
    os.system('wget https://raw.githubusercontent.com/jdariasl/ML_2020/master/Labs/commons/utils/general.py -O general.py')
    from general import configure_lab6
    configure_lab6()
from lab6 import *
GRADER, x,y = part_2()


# # Laboratorio 6 - Parte 2: Reducción de dimensión PCA y LDA

# Para el problema de clasificación usaremos la siguiente base de datos: https://archive.ics.uci.edu/ml/datasets/Cardiotocography
# 
# Analice la base de datos, sus características, su variable de salida y el contexto del problema.

# In[ ]:


print('Dimensiones de la base de datos de entrenamiento. dim de X: ' + str(np.shape(x)) + '\tdim de Y: ' + str(np.shape(y)))


# Este ejercicio tiene como objetivo implementar varias técnicas de extracción de características (PCA y LDA)

# **observación para las librerias sklearn**
# 
# Llamar explicitamente los parametros de las librerias de sklearn (e.j. si se quiere usar el parametro `kernel` del `SVC`, se debe llamar `SVC(kernel='rbf'`)

# En la siguiente celda se define una función para entrenar un SVM para resolver el problema. Esta función la vamos a usar como base para comparar nuestros metodos de selección de características.

# In[ ]:


def entrenamiento_sin_seleccion_caracteristicas(splits, X, Y):
    """
    Función que ejecuta el entrenamiento del modelo sin una selección particular
    de las características

      Parámetros:
        splits : numero de particiones  a realizar
      Retorna:
        - El vector de errores
        - El Intervalo de confianza
        - El tiempo de procesamiento
    """
    #Implemetamos la metodología de validación
    Errores = np.ones(splits)
    Score = np.ones(splits)
    times = np.ones(splits)
    j = 0
    kf = KFold(n_splits=splits)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        #Creamos el clasificador SVM.
        clf = SVC(kernel="linear", C=1)
        #Aquí se entran y se valida el modelo sin hacer selección de características
        tiempo_i = time.time()
        clf.fit(X_train,y_train)
        # Validación del modelo
        Errores[j] = accuracy_score(y_true=y_test, y_pred=clf.predict(X_test))
        times[j] = time.time()-tiempo_i
        j+=1

    return np.mean(Errores), np.std(Errores), np.mean(times)


# ### Ejercicio 1: Entrenamiento usando PCA para realizar extracción

# En este ejercicio vamos a aplicar PCA para realizar la extracción de caracteristicas. Para ello tener en cuenta:
# 
# 1. Vamos a usar el modulo [PCA de sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). El cual ya se encuentra importado (se pueda acceder a el como `PCA(....)`)
# 2. Tener en cuenta la respuesta de la siguiente pregunta abierta y completar el código de acuerdo a la respuesta usando la libreria y modulo de sklearn correspondiente (El cual tambien deberia ya estar importado en la configuración).
# 3. Usar el parametro adecuado para las particiones en la metodologia de validación
# 3. Usar la exactitud como medida de error del modulo [metrics de sklearn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
# 4. Vamos a calcular el costo computacional de aplicar la PCA.
# 5. Recordar que PCA se debe "ajustar" con el conjunto de entrenamiento. Pero la transformación se debe hacer para las particiones de entrenamiento y test.

# In[ ]:


#@title Pregunta Abierta
#@markdown ¿Cuando se aplica PCA ¿es necesario estandarizar los datos? Si, No y por qué? En qué consiste dicha estandarización?
respuesta_1 = '' #@param {type:"string"}


# In[ ]:


#@title Pregunta Abierta
#@markdown  La proyección de los datos que realiza PCA busca optimizar una medida, ¿Cuál? Explique.
respuesta_2 = '' #@param {type:"string"}


# In[ ]:


#ejercicio de código
def entrenamiento_pca_ext_caracteristicas(n_comp, n_sets, X, Y):
    """
    Esta función realiza la reducción de la dimensionalidad sobre el conjunto de
    datos de entrenamiento, de acuerdo con las particiones especificadas usando PCA

    Parámetros:
        n_comp, int, Número de componentes para reducción
        n_sets,int, Número de particiones
        X: numpy Array de características
        Y: numpy Array  Vector de etiquetas

    Retorna: 
        El valor medio de errores
        Intervalo de confianza del error
        El  valor medio del tiempo de ejecución
    """  
    #Implemetamos la metodología de validación 
    Errores = np...
    times = np...
    j = 0
    kf = KFold(...)
    for train_index, test_index in kf.split(X):  
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        # ¿es necesario estandarizacion de datos?
        ...
        X_train = ....
        X_test = ...
 
        #dejar el mismo nombre del objeto 
        pca = ...(n_components=...)
        # para calcular costo computacional
        tiempo_i = time.time()
        # es recomendable usar el metodo que ajusta y transforma
        X_train_pca = pca...(X=...)
        # aca solo usar el metodo de transformar (ya que en el anterior el pca se ajusto)
        X_test_pca = pca....(...)
        # entrenar el modelo usando las caractieristicas transformadas por PCA
        clf = SVC(kernel="linear", C=1)
        clf.fit(X=..., y=y_train)
        tiempo_o = time.time()-tiempo_i
        Errores[j] = ...(y_true=y_test, y_pred=....predict(...))
        times[j] = tiempo_o
        j+=1


    return np....(Errores), np....(Errores), np.mean(times)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio1", entrenamiento_pca_ext_caracteristicas)


# ### Ejercicio 2 : Experimentar con PCA
# 
# Usando las anteriores funciones vamos a realizar experimentos para evaluar la efectividad de PCA, para ello:
# 
# 1. Utilizar una metodología cross-validation con 5 particiones.
# 2. Usar como parametros para los experimentos el número de características a extraer
# 3. Usar la función `entrenamiento_pca_ext_caracteristicas` para realizar la extración de características.
# 3. Vamos a retornar un DataFrame con las siguientes columnas:
#     - CON_SEL (indicando si se uso selección de caracteristicas)
#     - NUM_VAR (número de selección de caracteristicas)
#     - T_EJECUCION: tiempo de ejecucción
#     - ERROR_VALIDACION
#     - IC_STD_VALIDACION
# 4. En la primera fila del dataframe vamos a incluir la evaluación del modelo SVM sin selección de características (usando la función creada en el primer ejercicio). 

# In[ ]:


#ejercicio de código
def experimentar_PCA(n_feats, X, Y):
    """
    Esta función realiza la comparación del desempeño de PCA utilizando diferente 
    número de caracteristicas y particionando el conjunto de datos en 5 conjuntos

    Parámetros:
        X (numpy.array), El arreglo numpy de características
        Y (numpy.array), El vector de etiquetas
        n_feats, Vector de números enteros que indica el número de características
                que debe utilizar el modelo

    Retorna:  
    - DataFrame con las columnas: CON_SEL, NUM_VAR, T_EJECUCION, ERROR_VALIDACION y IC_STD_VALIDACION. 

    """
    df = pd.DataFrame()
    idx = 0
    split_number = 5
    #Sin selección de características
    error,ic_error,t_ex = ...(split_number, X,Y)  
    df.loc[idx,'CON_SEL'] = 'NO'
    df.loc[idx,'NUM_VAR'] = X.shape[1] # se usan todas las caracteristicas
    df.loc[idx,'T_EJECUCION'] = ...
    df.loc[idx,'ERROR_VALIDACION'] = ...
    df.loc[idx,'IC_STD_VALIDACION'] = ...
    idx+=1
    print("termina experimento sin selección")
    #Con selección de características
    
    for f in n_feats:
        #Implemetamos la metodología de validación 
        ..., ..., ... = ...(n_comp=f, X=...,Y=... n_sets...)
        df.loc[idx,'CON_SEL'] = 'SI'
        df.loc[idx,'NUM_VAR'] = ...
        df.loc[idx, 'T_EJECUCION'] = ...
        df.loc[idx,'ERROR_VALIDACION'] = ...
        df.loc[idx, 'IC_STD_VALIDACION'] = ...
        idx+=1
    return df


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio2", experimentar_PCA)


# In[ ]:


experimentar_PCA(n_feats=[2,5,10,15,20], X= x, Y = y)


# In[ ]:


# aca realizamos una curva de varianza explicada del PCA
pca_varianza = PCA(n_components=x.shape[1]).fit(StandardScaler().fit_transform(x))
plt.plot(np.cumsum(pca_varianza.explained_variance_/np.sum(pca_varianza.explained_variance_)))
plt.title('Varianza acumulada')
plt.xlabel('Componentes principales')
plt.ylabel('Porcentaje de varianza acumulada')
plt.grid()


# ahora recordemos que el PCA tambien nos sirve para explorar y visualizar los datos en pocas dimensiones. En la siguiente celda vamos a visualizar nuestro conjunto de datos usando los dos primeros componentes principales

# In[ ]:


data_to_plot = StandardScaler().fit_transform(X=x)
fig, ax = plt.subplots()
pca = PCA(n_components=2)
x_pc2 = pca.fit_transform(data_to_plot)
scatter= ax.scatter(x= x_pc2[:,0], y = x_pc2[:,1], c = y, alpha = 0.5, label = y)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()


# In[ ]:


#@title Pregunta Abierta
#@markdown Aunque PCA nos nos ofrece una visualización aproximada de nuestro conjunto de datos, conociendo la varianza acumulada obtenida para 2 componentes principales ¿que tan cercana es la aproximación para este problema en específico?
respuesta_3 = '' #@param {type:"string"}


# En nuestros laboratorios anteriores, hemos usado el siguiente codigo:

# In[ ]:


digits = load_digits(n_class=5)
#--------- preprocesamiento--------------------
pca = PCA(n_components = 0.99, whiten=True)
print("shape antes pca", digits.data.shape)
data = pca.fit_transform(digits.data)
print("shape luego pca", data.shape)


# Aplicando los conceptos que manejamos en este laboratorio, responde la siguiente pregunta abierta

# In[ ]:


#@title Pregunta Abierta
#@markdown En nuestros laboratorios pasados y usando el código ejecutado anteriormente ¿Las 40 características representaban bien la varianza del conjunto original?
respuesta_4 = '' #@param {type:"string"}


# ### Ejercicio 3: Entrenamiento usando Discriminante de Fisher para extracción

# En este ejercicio vamos a aplicar PCA para realizar la extracción de caracteristicas. Para ello tener en cuenta:
# 
# 1. Vamos a usar el modulo [LinearDiscriminantAnalysis-LDA de sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html). El cual ya se encuentra importado (se pueda acceder a el como `LinearDiscriminantAnalysis(....)`)
# 2. ¿También se estandarizar los datos?
# 3. Usar 5 particiones en la metodologia de validación
# 3. Usar la exactitud/accuracy como medida de error del modulo [metrics de sklearn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
# 4. Vamos a calcular el costo computacional de aplicar la LDA.
# 5. Recordar que LDA se debe "ajustar" con el conjunto de entrenamiento. Pero la transformación se debe hacer para las particiones de entrenamiento y test.

# In[ ]:


#@title Pregunta Abierta
#@markdown Explicar en sus palabras la principal ventaja que tiene LDA sobre PCA para resolver problemas de clasificación.
respuesta_5 = '' #@param {type:"string"}


# In[ ]:


#ejercicio de código
def entrenamiento_lda_ext_caracteristicas(n_comp, X, Y):
    """
    Esta función realiza la reducción de la dimensionalidad sobre el conjunto de
    datos de entrenamiento, de acuerdo con las particiones especificadas usando PCA

    Parámetros:
        n_comp, int, Número de componentes para reducción
        X: numpy Array de características
        Y: numpy Array  Vector de etiquetas

    Retorna:
        tupla con:
        El  valor medio del tiempo de ejecución,
        El valor medio de errores
        Intervalo de confianza de los errores
    """
   

    #Implemetamos la metodología de validación 
    Errores = np.ones(5)
    times = np.ones(5)
    j = 0
    kf = KFold(...)
    for train_index, test_index in kf.split(X):  
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        # ¿es necesario estandarizacion de datos?
        ...
        X_train = ....
        X_test = ...
        # dejar el nombre del objeto igual (lda)
        lda = ...(n_components=...)
        # para calcular costo computacional
        tiempo_i = time.time()
        # es recomendable usar el metodo que ajusta y transforma
        X_train_lda = lda....(X_train, y_train)
        # aca solo usar el metodo de transformar (ya que en el anterior el pca se ajusto)
        X_test_lda = lda...(...)
        # entrenar el modelo usando las caractieristicas transformadas por PCA
        clf = SVC(kernel="linear", C=1)
        clf.fit(X=X_train_lda, y=...)
        tiempo_o = time.time()-tiempo_i
        Errores[j] = ...(y_true=y_test, y_pred=clf...(X_test_lda))
        times[j] = tiempo_o
        j+=1


    return np...(times), np...(Errores), np...(Errores)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio3", entrenamiento_lda_ext_caracteristicas)


# ### Ejercicio 4 : Experimentar con Discriminante de Fisher

# Usando las anteriores funciones vamos a realizar experimentos para evaluar la efectividad de PCA, para ello:
# 
# 1. Utilizar una metodología cross-validation con 5 particiones.
# 2. Usar como parametros para los experimentos el número de características a extraer
# 3. Usar la función `entrenamiento_pca_ext_caracteristicas` para realizar la extración de características.
# 3. Vamos a retornar un DataFrame con las siguientes columnas:
#     - CON_SEL (indicando si se uso selección de caracteristicas)
#     - NUM_VAR (número de selección de caracteristicas)
#     - ERROR_VALIDACION
#     - IC_STD_VALIDACION
#     - T_EJECUCION: tiempo de ejecucción
# 4. En la primera fila del dataframe vamos a incluir la evaluación del modelo SVM sin selección de características (usando la función creada en el primer ejercicio). 

# In[ ]:


#ejercicio de código
def experimentar_LDA(n_feats, X, Y):
    """
    Esta función realiza la comparación del desempeño de LDA utilizando diferente 
    número de feats y particionando el conjunto de datos en diferente en 5 subconjuntos

    Parámetros:
    X (numpy.array), El arreglo numpy de características
    Y (numpy.array), El vector de etiquetas
    n_feats, Vector de números enteros que indica el número de características
              que debe utilizar el modelo

    Retorna:  
    - DataFrame con las columnas: CON_SEL, NUM_VAR, ERROR_VALIDACION, IC_STD_VALIDACION, 
    y T_EJECUCION. 

    """

    df = pd.DataFrame()
    idx = 0
    split_number = 5
    #Sin selección de características
    ...,...,... = ...(split_number, X,Y)  
    df.loc[idx,'CON_SEL'] = 'NO'
    df.loc[idx,'NUM_VAR'] = X.shape[1] # se usan todas las caracteristicas
    df.loc[idx,'ERROR_VALIDACION'] = ...
    df.loc[idx,'IC_STD_VALIDACION'] = ...
    df.loc[idx,'T_EJECUCION'] = ...
    idx+=1
    print("termina experimento sin selección")
    #Con selección de características
    
    for f in n_feats:
        #Implemetamos la metodología de validación 
        ..., ..., ... = ...(n_comp=f, X=x,Y=y)
        df.loc[idx,'CON_SEL'] = 'SI'
        df.loc[idx,'NUM_VAR'] = ...
        df.loc[idx,'ERROR_VALIDACION'] = ...
        df.loc[idx, 'IC_STD_VALIDACION'] = ...
        df.loc[idx, 'T_EJECUCION'] = ...
        idx+=1
    return df


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio4", experimentar_LDA)


# In[ ]:


experimentar_LDA(n_feats=[1,2], X= x, Y = y)


# In[ ]:


#@title Pregunta Abierta
#@markdown  ¿que diferencias existen entre los métodos de selección de características y los métodos de extracción de características vistos en la anterior sesión? Explicar
respuesta_6 = '' #@param {type:"string"}


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

