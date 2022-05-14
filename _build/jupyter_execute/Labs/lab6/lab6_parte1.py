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
GRADER, x,y = part_1()


# # Laboratorio 6 - Parte 1: Reducción de dimensión y Selección de características

# Este ejercicio tiene como objetivo implementar varias técnicas de selección de características y usar regresion logisctica para resolver un problema de clasificación multiclase.

# Para el problema de clasificación usaremos la siguiente base de datos: https://archive.ics.uci.edu/ml/datasets/Cardiotocography
# 
# Analice la base de datos, sus características, su variable de salida y el contexto del problema.

# In[ ]:


print('Dimensiones de la base de datos de entrenamiento. dim de x: ' + str(np.shape(x)) + '\tdim de y: ' + str(np.shape(y)))


# **observación para las librerias sklearn**
# 
# Llamar explicitamente los parametros de las librerias de sklearn (e.j. si se quiere usar el parametro `kernel` del `SVC`, se debe llamar `SVC(kernel='rbf'`)

# ### Ejercicio 1: Entrenamiento sin selección de características
# 
# En nuestro primer ejercicio debemos completar la función para entrenar una SVM para resolver un problema de clasificación. Debemos completar siguiendo las recomendaciones:
# 
# 1. Mantener los parámetros sugeridos de la regresión logística. 
# 2. Asignar el parametro de StratifiedKFold a los splits
# 3. Usar la area bajo la curva ROC como medida de error del modulo [metrics de sklearn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics). Tener en cuenta que esta función recibe un score, que es diferente a la predicción (explorar que metodo debemos usar para nuestro caso de problema y usar un estrategia One vs One).
# 4. Esta función la vamos a usar como base para comparar nuestros metodos de selección de características.

# In[ ]:


#ejercicio de código
def entrenamiento_sin_seleccion_caracteristicas(X, Y, splits):
    """
    Función que ejecuta el entrenamiento del modelo sin una selección particular
    de las características

      Parámetros:
        X: matriz con las caracteristicas
        Y: Matriz con la variable objetivo
        splits : numero de particiones  a realizar
      
      Retorna:
      1. El modelo entreando
      2. El vector de errores
      3. El Intervalo de confianza
      4. El tiempo de procesamiento
    """
    #Implemetamos la metodología de validación
    Errores = np.ones(splits)
    Score = np.ones(splits)
    times = np.ones(splits)
    j = 0
    kf = StratifiedKFold(n_splits= ...)
    for train_index, test_index in kf.split(...):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        #Creamos el clasificador SVM.
        clf = LogisticRegression(solver="liblinear", random_state=0)
        #Aquí se entran y se valida el modelo sin hacer selección de características
        ######
        # Entrenamiento el modelo.
        #Para calcular el costo computacional
        tiempo_i = time.time()
        clf...(X_train,y_train)
        # Validación del modelo
        Errores[j] = ...(y_true=y_test, y_score=clf...(...), ...=...)
        times[j] = time.time()-tiempo_i
        j+=1
    
    return clf, np.mean(Errores), np.std(Errores), np.mean(times)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio1", entrenamiento_sin_seleccion_caracteristicas)


# In[ ]:


#@markdown En sus palabras ¿Cual es el proceso ejecutado por la función roc_auc_score que calcula la área sobre la curva roc?
respuesta_1 = '' #@param {type:"string"}


# ## Ejercicio 2: Entrenamiento con selección de características
# 
# La siguiente función "wrapper" nos permite hacer una selección de características utilizando la [librería recursive feature elimination de Sci-kit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html).
# 
# Esta libreria es un metodo de seleccion carcterisitcas wrapper, que usa los coeficientes derivados de  un estimador entrenado para estimar que caracteristicas tienen mayor poder predictivo.
# 
# Para completar debemos tener en cuenta lo siguiente:
# 
# 1. Para el número de caractersiticas usar el parametro feature_numbers
# 2. Establecer el paso = 1 para ir eliminando las caracteristicas
# 3. Asumir que el estimador se crea externamente de la función
# 4. Entender los campos del RFE disponibles despues de entrenarlo para obtener:
#     1. La mascara para saber que características fueron seleccionadas
#     2. El ranking de las caracteristicas

# In[ ]:


#ejercicio de código
def recursive_feature_elimination_wrapper(estimator, feature_numbers, X,Y):
    """
    Esta función es un envoltorio del objeto RFE de sklearn

    Parámetros:
    estimator(sklearn.linear_model._logistic.LogisticRegression), El estimador LR
    feature_numbers(int), El número de características a considerar
    X (numpy.array), El arreglo numpy de características
    Y (numpy.array), El vector de etiquetas

    Retorna:
    El modelo entrenado ()
    La máscara de características seleccionada, array [longitud de caracterisitcas de X]
    El rankeo de características, array [longitud de caracterisitcas de X]
    El objeto RFE entrenado sobre el set reducido de características
    El tiempo de ejecución
    """
    rfe = ...(estimator=..., n_features_to_select=..., step=...)
    tiempo_i = time.time()
    rfe...(X=X, y=Y)
    time_o = time.time()-tiempo_i
    feature_mask = rfe...
    features_rank = rfe...
    estimator = rfe...

    return rfe, feature_mask, features_rank, estimator, time_o


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio2", recursive_feature_elimination_wrapper)


# In[ ]:


#@title Preguntas Abierta
#@markdown Explicar ¿Que diferencia tiene el metodo implementado con un metodo de filtro de selección de caracteristicas? 
respuesta_2 = '' #@param {type:"string"}


# ## Ejercicio 3:  Comparación de los resultados del modelo
# 
# Ahora en la siguiente función, vamos a usar la función planteada para realizar experimentos con la selección de características. Para ello:
# 1. Utilizar una metodología cross-validation estratificada.
# 2. Considerando n características, retorne el modelo entrenado, el vector de errores, el intervalo de confianza y el tiempo de ejecución.
# 3. Vamos a retornar un DataFrame con las siguientes columnas:
#     - CON_SEL (indicando si se uso selección de caracteristicas)
#     - NUM_VAR (número de selección de caracteristicas)
#     - NUM_SPLITS  (número de particiones realizadas)
#     - T_EJECUCION: tiempo de ejecucción
#     - ERROR_VALIDACION
#     - IC_STD_VALIDACION
# 
# 4. En la primera fila del dataframe vamos a incluir la evaluación del modelo SVM sin selección de características (usando la función creada en el primer ejercicio) y sin particionar el set de datos.

# In[ ]:


#ejercicio de código
def experimentar(n_feats, n_sets, X, Y):
    """
    Esta función realiza la comparación del desempeño de RFE utilizando diferente 
    número de feats y particionando el conjunto de datos en diferente número de 
    subconjuntos

    Parámetros:
    X (numpy.array), El arreglo numpy de características
    Y (numpy.array), El vector de etiquetas
    n_feats, Vector de números enteros que indica el número de características
              que debe utilizar el modelo
    n_sets, Vector de números enteros que indica el número de particiones

    Retorna:  
    - DataFrame con las columnas: DESCRIPCION, T_EJECUCION ERROR_VALIDACION, 
    y IC_STD_VALIDACION 

    """
    df = pd.DataFrame()
    idx = 0
    for split_number in n_sets: 
    #Sin selección de características
        # se ignorar las otras salidas
        _,err,ic,t_ex = entrenamiento_sin_seleccion_caracteristicas(...)  
        
        df.loc[idx,'CON_SEL'] = 'NO'
        df.loc[idx,'NUM_VAR'] = X.shape[1]
        df.loc[idx,'NUM_SPLITS'] = ...
        df.loc[idx,'T_EJECUCION'] = ...
        df.loc[idx,'ERROR_VALIDACION'] = ...
        df.loc[idx,'IC_STD_VALIDACION'] = ...
        idx+=1
    print("termina experimentos sin selección")
    #Con selección de características
    for f in n_feats:
        for split_number in n_sets:
            #Implemetamos la metodología de validación 
            Errores = np.ones(split_number)
            Score = np.ones(split_number)
            times = np.ones(split_number)
            kf = ...(n_splits=split_number)
            j = 0
            for train_index, test_index in kf.split(...,...):
                
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                lr =  LogisticRegression(solver="liblinear", random_state=0)
                
                # se ignorar las otras salidas
                rfe, _, _, _, t = recursive_feature_elimination_wrapper(estimator=...,
                                                                        feature_numbers=...,
                                                                        X=...,
                                                                        Y=...)
                Errores[j]=roc_auc_score(y_true=y_test, y_score=rfe.predict_proba(...), multi_class='ovo')
                times[j] = t
                j+=1

            df.loc[idx,'CON_SEL'] = 'SI'
            df.loc[idx,'NUM_VAR'] = f
            df.loc[idx,'NUM_SPLITS'] = split_number
            df.loc[idx, 'T_EJECUCION'] = np.mean(...)
            df.loc[idx,'ERROR_VALIDACION'] = np.mean(Errores)
            df.loc[idx, 'IC_STD_VALIDACION'] = np.std(Errores)
            idx+=1
    return df


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio3", experimentar)


# Ejecuta la celda de codigo para realizar los experimentos

# In[ ]:


dfr = experimentar(n_feats = [3, 5, 10,15,20], n_sets = [3, 6], X= x, Y=y)


# Utilizando el *DataFrame* definido en la función anterior, retorne la mejor configuración del "Número de características" que resultó ,as beneficiosa. Identifique en términos de "Costo computacional" y "Eficiencia del Modelo".

# In[ ]:


# podemos cambiar las forma de organizar el df para observar que diferencias hay
# cuando cambia  la prioridad alguno de los dos parámetros
dfr.sort_values(['T_EJECUCION', "ERROR_VALIDACION"], ascending=[False, True])


# Veamos como se relaciona el tiempo de ejecución con los splits y la selección de caracteristicas

# In[ ]:


import seaborn as sns
d_toplot = pd.melt(dfr,id_vars=['CON_SEL', 'NUM_VAR', 'NUM_SPLITS'], value_vars=['ERROR_VALIDACION', 'T_EJECUCION'])
sns.relplot(data = d_toplot, x = 'NUM_VAR', y = 'value', hue = 'CON_SEL', style = 'NUM_SPLITS', col = 'variable', kind='scatter', facet_kws = {'sharey' : False}, aspect=1.2,s=150)


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿Que relación observa entre tiempo de ejecución, el desempeño del modelo y el número de características? Explicar con base a los resultados y el conocimiento desde la teoría
respuesta_3 = '' #@param {type:"string"}


# Ahora use el mejor modelo para entrenar nuevamente el modelo y saber que caracteristicas tienen el mejor poder predictivo. Use el # de caracteristicas que resulto mejor cuando se realizo alguna selección de caracteristicas.

# In[ ]:


# observemos el mejor modelo cuando se realizo selección de caracteristicas
dfr[dfr['CON_SEL'] == 'SI'].sort_values(["ERROR_VALIDACION"], ascending=[False]).head(1)


# In[ ]:


lr =  LogisticRegression(solver="liblinear", random_state=0)
rfe, feature_mask, _, _, _ = recursive_feature_elimination_wrapper(lr, ...)

mask_explicada = "\n".join([f"feature {i+1} : {m}" for i,m in zip(range(x.shape[1]), feature_mask)])

print(f"esta es la mascara (deberia ser solo valores True y False) \n {mask_explicada}")


# In[ ]:


#@title Pregunta Abierta
#@markdown Utilizando los resultados obtenidos, sí en algún momento se debe prescindir de algunas de las variables para realizar el diagnóstico, ¿cuáles se podrían sugerir como candidatas menos importantes al personal médico ?
respuesta_4 = '' #@param {type:"string"}


# Recordemos el [concepto de importancia de variables de los metodos basados en arboles](https://jdariasl.github.io/ML_2020/Clase%2010%20-%20%C3%81rboles%20de%20Decisi%C3%B3n%2C%20Voting%2C%20Bagging%2C%20Random%20Forest.html).
# 
# El siguiente código implementa el calculo de la importancia de variables en nuestro problema.

# In[ ]:


# entrenar random forest
from sklearn.ensemble import RandomForestClassifier
feature_names = [f"feature {i+1}" for i in range(x.shape[1])]
forest = RandomForestClassifier(random_state=0)
forest.fit(x, y)
#obtener la importnacia de variables
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
# Graficar las importancia
fig, ax = plt.subplots(figsize = (12,6))
forest_importances = pd.Series(importances, index=feature_names)
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Importancia de variables")
ax.set_ylabel("Valor medio en dismunución de impureza")
fig.tight_layout()


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿Observa alguna relación entre la importancia de las variables y los resultados obtenidos en los experimentos? ¿Desde el conocimiento teórico, esperábamos ver esa relación?
respuesta_5 = '' #@param {type:"string"}


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

