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
    from general import configure_lab5_2
    configure_lab5_2()
from lab5 import *
GRADER, dataset = part_2()


# # Laboratorio 5 - Parte 2. Máquinas de Vectores de Soporte

# ### Ejercicio 1: Limipiar base de datos y completar código
# 
# En este ejercicio usaremos la regresión por vectores de soporte para resolver el problema de regresión de la base de datos AirQuality (https://archive.ics.uci.edu/ml/datasets/Air+Quality). Tener en cuenta que vamos a usar solo 2000 muestras.
# 
# En primera instancia vamos a transformar la matriz en un dataframe, para poderlo procesar de manera mas sencilla. Se crea una columna por cada variable que obtenemos.

# In[ ]:


dataset_df = pd.DataFrame(dataset, columns = [f'col_{c}' for c in range (1,14)])


# Para esta base de datos vamos a realizar una limpieza de datos. 
# Para ello vamos a completar la siguiente función para realizar:
#     
# 1. **Remover** todos registros cuya variable objetivo es faltante (missing Value). Estos registros están marcados como -200, es decir, donde haya un valor -200 eliminaremos el registro.
# 2. **imputar los valores perdidos/faltantes** en cada una de las características, vamos a usar el valor medio de la característica en especifico.
# 3. **Verificar** si quedaron valores faltantes
# 4. **retornar**X (12 primeras columnas) y Y(la 13 columna).
# 
# informacion de utilidad:
# 
# 1. Aca puede ser de utilidad recordar nuestra [sesión extra](https://jdariasl.github.io/ML_2020/Labs/Extra/Basic_Preprocessing_FeatureEngineering.html).
# 2. Para transformar columnas de pandas a arreglos de numpy se puede usar `.iloc` / `.loc` y . `.values`, por ejemplo  para devolver una matriz con los valores de las primeras dos columnas es posible hacerlo asi: `dataset_df.iloc[: , 0:2].values` o `dataset_df.loc[: , ['col_1', 'col_2']].values`
# 3. Para cambiar valores faltantes, podemos usar la [librería sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

# In[ ]:


# ejercicio de codigo
def clean_data(df):
    """funcion que limpia el dataset y obtiene X y Y
    
    df: es un pandas dataframe
    
    retorna:
    X: una matriz numpy con los valores a usar como conjunto de datos
       de entrada
    Y una matriz numpy con los valores usados como conjunto de datos
       de salida
    
    """
    
    # se copia el df para evitar cambios sobre el objeto
    database = df.copy()
    
    ##Verificar
    pct_valores_faltantes = (database==-200).mean()
    print(",".join([f"% VF en {a}: {pct_valores_faltantes[a]*100:.2f}% " for a in pct_valores_faltantes.index]))
    
    # identificar muetras cuya salida en un valor faltante
    idx_to_remove = []
    for idx in database.index:
        ## reemplazar el valor
        if database.iloc[idx,...] == ...:
            idx_to_remove.append(idx)
    
    #remover la muestras de los indices
    database = database.drop(idx_to_remove,axis = 0)

    print ("\nHay " + str(len(idx_to_remove)) + " valores perdidos en la variable de salida.")
    print ("\nDim de la base de datos sin las muestras con variable de salida perdido "+ str(np.shape(database)))

    ##Imputar
    print ("\nProcesando imputación de valores perdidos en las características . . .")
    imputer = ... (missing_values= ... , strategy= ...)
    "imputar solo las columnas de las variables de entrada"
    database.iloc[:,0:...] = imputer.fit_transform(database.iloc[:,...] )

    print ("Imputación finalizada.\n")

    ##Verificar
    pct_valores_faltantes = (database==-200).mean()
    
    print(",".join([f"% VF en {a}: {pct_valores_faltantes[a]*100:.2f}% " for a in pct_valores_faltantes.index]))
    
    if(pct_valores_faltantes.max() != 0):
        print ("Hay valores perdidos")
    else:
        print ("No hay valores perdidos en la base de datos. Ahora se puede procesar")

    X = database.iloc[:,0:12].values
    Y = database.iloc[:,12:13].values
    return (X,Y)


# In[ ]:


# ignorar los prints
GRADER.run_test("ejercicio1", clean_data)


# Ahora usemos la función para tener nuestras variables X, Y

# In[ ]:


X,Y = clean_data(dataset_df)


# ### Ejercicio 2: Experimentar SVM para regresión
# 
# Ahora vamos a crear la función para experimentar con la maquina de soporte vectorial. Para ellos vamos:
# 1. Usar la libreria de sklearn https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html. 
# 2. Vamos a variar tres parámetros del SVR: kernel,  gamma y el parametro de regularización.
# 3. Utilizar la metodología cross-validation con 4 folds.
# 4. Usar normalización de datos estandar implementada por sklearn
# 5. Extraer los vectores de soporte (observe los *atributos* del modelo SVR de sklearn). Recuerde que estos atributos son accesibles, una vez el modelo es entrenado
# 6. Utilizar el metrica para calcular el MAPE (usar sklearn).
# 
# **Notas**: 
# - Deberiamos poder acceder a las funciones de la libreria de sklearn directamente por el nombre sin necesidad de importarlas. Las funciones que deberios utilizar ya están precargadas en la configuración del laboratorio.
# - Llame todos los parametros de las funciones de sklearn de manera explicita. (i.e, si se quiere usar `max_iter` como parámetro para el SVR, debe crear el objeto: `SVR(max_iter = 100)`)

# In[ ]:


#ejercicio de código
def experiementarSVR(x, y, kernels, gammas,params_reg):
    """función que realizar experimentos sobre un SVM para regresión
    
    x: numpy.Array, con las caracteristicas del problema
    y: numpy.Array, con la variable objetivo
    kernels: List[str], lista con valores a pasar 
        a sklearn correspondiente al kernel de la SVM
    gammas: List[float], lista con los valores a pasar a
        sklean correspondiente el valor de los coeficientes para usar en el
        kernel
    params_reg: List[float], lista con los valores a a pasar a 
        sklearn para ser usados como parametro de regularización
    
    retorna: pd.Dataframe con las siguientes columnas:
        - 3 columnas con los tres parametros: kernel, gamma, param de regularizacion
        - error cuadratico medio en el cojunto test (promedio de los 5 folds)
        - intervalo de confianza del error cuadratico medio en el cojunto test 
            (desviacion estandar de los 5 folds)
        - # de Vectores de Soporte promedio para los 5 folds
        - % de Vectores de Soporte promedio para los 5 folds (0 a 100)
    """
    idx = 0
    kf = ...(n_splits=...)
    # crear una lista con la combinaciones de los elementos de cada list
    kernels_gammas_regs = list(itertools.product(kernels, gammas, params_reg))
    resultados = pd.DataFrame()
    
    for params in kernels_gammas_regs:
        kernel, gamma, param_reg = params
        print("parametros usados", params) # puede usar para ver los params
        errores_test = []
        pct_support_vectors = []
        num_support_vectors = []
        for train_index, test_index in kf...(...)
            
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]  
            # normalizar los datos
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            svm = ...(kernel = ...., gamma = ..., C = ...)
            # Entrenar el modelo
            svm.fit(X=X_train, y=...)
            # Validación del modelo
            ypred = svm.predict(X=...)
            
            # error y pct de vectores de soporte
            errores_test.append(...(y_true = y_test, y_pred = ypred))
            # contar muestras de entrenamiento
            n_train = X_train.shape[0]
            num_vs = len(svm...)
            pct_vs = (... / n_train ) *100
            pct_support_vectors.append(pct_vs)
            num_support_vectors.append(num_vs)
        
        resultados.loc[idx,'kernel'] = kernel
        resultados.loc[idx,'gamma'] = gamma
        resultados.loc[idx,'param_reg'] = param_reg
        resultados.loc[idx,'error de prueba (promedio)'] = np.mean(...)
        resultados.loc[idx,'error de prueba (intervalo de confianza)'] = np.std(errores_test)
        resultados.loc[idx,'# de vectores de soporte'] = np.mean(...)
        resultados.loc[idx,'% de vectores de soporte'] = np.mean(...)
        
        idx+=1
    return (resultados)


# In[ ]:


GRADER.run_test("ejercicio2", experiementarSVR)


# Para entrenar vamos a ignorar las dos primeras variables, estas corresponden a valores de fechas.

# In[ ]:


# vamos a realizar los experimentos
resultadosSVR = experiementarSVR(x =X[:,2:],y=Y,
                                 kernels=['linear', 'rbf'],
                                 gammas = [0.01,0.1],
                                 params_reg = [0.1, 1.0,10]
                                )


# In[ ]:


# ver los 5 primeros resultados
resultadosSVR.sort_values('error de prueba (promedio)',ascending=True).head(5)


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿Cual es el objetivo de las las funciones kernel? Contestar dentro del contexto de las máquinas de sporte vectorial
respuesta_1 = "" #@param {type:"string"}


# In[ ]:


#@title Pregunta Abierta
#@markdown Explique en sus palabras ¿qué representan los vectores de soporte en un problema de regresión?
respuesta_2 = "" #@param {type:"string"}


# Para analizar los resultados vamos a crear dos graficas para el mejor modelo encontrado:
# 1. vamos a graficar en el eje x el valor real, en el eje y el valor predicho. El modelo ideal deberia ser una recta que recuerda la identidad
# 2. en el eje x vamos a dejar un valor incremental y con colores vamos a diferenciar entre el valor real y el valor predicho
# 
# 

# In[ ]:


# dividir el conjunto
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#predicciones
# OJO: Reemplazar los valores!
Ypred =  predict_svr(X_train,y_train,X_test,kernel = ..., )

# plots

f, ax = plt.subplots(ncols=2, sharex=False, sharey=False, figsize = (22,6))
ax[0].scatter(y_test, Ypred)
ax[0].set_xlabel('valor real', fontdict = {'fontsize': 12})
ax[0].set_ylabel('valor predicho', fontdict = {'fontsize': 12})
ax[1].plot(y_test, label = 'valor real', color = 'b', alpha = 0.7)
ax[1].plot(Ypred, label = 'valor predicho', color = 'r', alpha = 0.5)
ax[1].legend()
ax[1].set_ylabel('Humedad relativa', fontdict = {'fontsize': 12})
plt.show()


# In[ ]:


#@title Pregunta Abierta
#@markdown usando las anteriores graficas, ¿como calificaria el modelo de manera cualitativa?.
respuesta_3 = "" #@param {type:"string"}


# ### Ejercicio 3: Experimentar SVM para clasificación

# En este ejercicio vamos a volver a resolver el problema de clasificación de dígitos. Vamos usar solo 5 clases y realizaremos un pre-procesamiento:
# 1. mediante PCA (una tecnica proxima a practicar en el laboratorio)
# 2. Vamos a convertirlo en problema biclase (vamos a diferenciar entre 0, 1 y el resto)

# In[ ]:


Xcl, Ycl = load_digits(n_class=5,return_X_y=True)
#--------- preprocesamiento--------------------
pca = PCA(0.99, whiten=True)
Xcl = pca.fit_transform(Xcl)
# cambiar problema de clases
unique, counts  = np.unique(Ycl, return_counts=True)
print("distribución original (claves las etiquetas, valores el número de muestras): \n", dict(zip(unique, counts )))
Ycl = np.where(np.isin(Ycl, [0,1]), Ycl, 2)
unique, counts  = np.unique(Ycl, return_counts=True)
print("Nueva distribución  (claves las etiquetas, valores el número de muestras): \n", dict(zip(unique, counts )))


# Ahora vamos a crear la función para experimentar con la maquina de soporte vectorial. Para ellos vamos:
# 
# 1. Usar la libreria de sklearn https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
# 2. Vamos a variar tres parámetros del SVC: kernel,  gamma y el parametro de regularización.
# 3. Utilizar la metodología cross-validation con 4 folds más adecuada para problemas de clasificación.
# 4. Usar normalización de datos estandar implementada por sklearn
# 5. Extraer los vectores de soporte (observe los *atributos* del modelo SVC de sklearn). Recuerde que estos atributos son accesibles una vez el modelo es entrenado
# 6. vamos a probar la dos estragegias del SVC:
#     - One Vs One
#     - One Vs Rest
# 7. Utilizar como error el score de exactitud de la clasificación de sklearn.
# 
# **Notas**: 
# - Deberiamos poder acceder a las funciones de la libreria de sklearn directamente por el nombre sin necesidad de importarlas. Las funciones que deberios utilizar ya están precargadas en la configuración del laboratorio.
# - Llame todos los parametros de las funciones de sklearn de manera explicita. (i.e, si se quiere usar `max_iter` como parámetro para el SVC, debe crear el objeto: `SVC(max_iter = 100)`)

# * sklearn tiene unos "wrappers", que implementan estrategias para la clasificación multiclase, uno  de estos wrappers, implementa la estrategia one-vs-rest: https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html.
# 
# * Un wrapper, es un esquema de diseño común para "envolver" librerias/funciones con caracteristicas similares y poder modificar ciertos comportamientos.

# In[ ]:


#ejercicio de código
def experiementarSVC(x, y, kernels, gammas,params_reg, estrategia = 'ovo'):
    """función que realizar experimentos sobre un SVM para clasificación
    
    x: numpy.Array, con las caracteristicas del problema
    y: numpy.Array, con la variable objetivo
    kernels: List[str], lista con valores a pasar 
        a sklearn correspondiente al kernel de la SVM
    gammas: List[float], lista con los valores a pasar a
        sklean correspondiente el valor de los coeficientes para usar en el
        kernel
    params_reg: List[float], lista con los valores a a pasar a 
        sklearn para ser usados como parametro de regularización
    estrategia: str, valor que puede ser ovo (para one vs one) o ovr 
        (para one vs rest)
    
    retorna: pd.Dataframe con las siguientes columnas:
        - 3 columnas con los tres parametros: kernel, gamma, param de regularizacion
        - error cuadratico medio en el cojunto entrenamiento (promedio de los 4 folds)
        - error cuadratico medio en el cojunto test (promedio de los 4 folds)
        - % de Vectores de Soporte promedio para los 4 folds (0 a 100)
    """
    idx = 0
    kf = ...(n_splits=...)
    # crear una lista con la combinaciones de los elementos de cada list
    kernels_gammas_regs = list(itertools.product(kernels, gammas, params_reg))
    resultados = pd.DataFrame()
    
    for params in kernels_gammas_regs:
        kernel, gamma, param_reg = params
        print("parametros usados", params) # puede usar para ver los params
        errores_train = []
        errores_test = []
        pct_support_vectors = []
        
        for train_index, test_index in kf...(...,...):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]  
            # normalizar los datos
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            svm = SVC(kernel = ..., gamma = ..., C = ...)
            
            # si es estrategia "envolver" a la svm
            if estrategia =='ovr':
                svm = ... (svm)               
            
            # Entrenar el modelo
            svm...(X=X_train, y=y_train)
            # calculo de errores
            y_train_pred = svm...(X=X_train)
            y_test_pred = svm...(X=X_test)
            # error y pct de vectores de soporte
            errores_train.append(accuracy_score(y_true = y_train, y_pred = y_train_pred))
            errores_test.append(accuracy_score(y_true = y_test, y_pred = y_test_pred))
            # contar muestras de entrenamiento
            n_train = X_train.shape[0]
            if estrategia == 'ovr':
                # en esta estrategia se realizar una SVM por cada clase
                # por lo tanto tenemos que acceder a cada una de la SVM
                # lee la documentacion
                num_vs = np.mean([len(svc...) for svc in svm....])
                pct_vs = (num_vs/n_train)*100
                
            else:
                # cuando es ovo solo tenemos una SVM
                pct_vs = (len(svm...)/n_train)*100
            pct_support_vectors.append(pct_vs)
        
        resultados.loc[idx,'kernel'] = kernel
        resultados.loc[idx,'gamma'] = gamma
        resultados.loc[idx,'param_reg'] = param_reg
        resultados.loc[idx,'estrategia'] = ...
        resultados.loc[idx,'error de entrenamiento'] = np.mean(errores_train)
        resultados.loc[idx,'error de prueba'] = np.mean(errores_test)
        resultados.loc[idx,'% de vectores de soporte'] = np.mean(...)
        idx+=1
    return (resultados)


# In[ ]:


GRADER.run_test("ejercicio3", experiementarSVC)


# Veamos la estrategia OVR

# In[ ]:


# vamos a realizar los experimentos
resultadosSVC_ovr = experiementarSVC(x = Xcl,y=Ycl,
                                 kernels=['linear', 'rbf'],
                                 gammas = [0.01,0.1],
                                 params_reg = [0.001, 0.01,0.1, 1.0,10],
                                estrategia = 'ovr')


# In[ ]:


# ver los mejores modelos
resultadosSVC_ovr.sort_values('error de prueba', ascending=False).head(5)


# Ahora vamos a ver la estrategia OVO

# In[ ]:


# vamos a realizar los experimentos
resultadosSVC_ovo = experiementarSVC(x = Xcl,y=Ycl,
                                 kernels=['linear', 'rbf'],
                                 gammas = [0.01,0.1],
                                 params_reg = [0.001, 0.01,0.1, 1.0,10],
                                estrategia = 'ovo')


# In[ ]:


# ver los mejores modelos
resultadosSVC_ovo.sort_values('error de prueba', ascending=False).head(5)


# In[ ]:


#@title Pregunta Abierta
#@markdown Explique en sus palabras ¿qué representan los vectores de soporte en un problema de clasificación?
respuesta_4 = "" #@param {type:"string"}


# In[ ]:


#@title Pregunta Abierta
#@markdown Según el tipo de problema (enfocarse en la distribución de clases) ¿La métrica usada es la adecuada? ¿Cual otra métrica del modulo de sklearn podría ser usada?
respuesta_5 = "" #@param {type:"string"}


# In[ ]:


# ver la relación de parametro de regularización y los vectores de soporte
import seaborn as sns
ax= sns.relplot(data = resultadosSVC_ovo, x = 'param_reg', y = '% de vectores de soporte', kind = 'line', hue ='kernel', aspect = 1.5)
ax.set(xscale="log")


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿qué relación observa entre el valor del parametro de regularización y los vectores de soporte? ¿como explica esta relación?
respuesta_6 = "" #@param {type:"string"}


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

