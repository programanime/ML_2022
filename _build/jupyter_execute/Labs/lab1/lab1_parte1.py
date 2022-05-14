#!/usr/bin/env python
# coding: utf-8

# **Recuerda que una vez abierto, Da clic en "Copiar en Drive", de lo contrario no podras alamancenar tu progreso**
# 
# Nota: no olvide ir ejecutando las celdas de código de arriba hacia abajo para que no tenga errores de importación de librerías o por falta de definición de variables.
# 

# In[ ]:


#configuración del laboratorio
# Ejecuta esta celda!
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
#for local
#import sys ; sys.path.append('../commons/utils/'); sys.path.append('../commons/utils/data')
get_ipython().system('wget https://raw.githubusercontent.com/jdariasl/ML_2020/master/Labs/commons/utils/general.py -O general.py')
from general import configure_lab1_p1
configure_lab1_p1()
from lab1 import *
GRADER_LAB_1_P1, db, x, y = part_1()
y = y.reshape(np.size(y), 1)


# # Laboratorio 1 - Parte 1 Regresión polinomial múltiple
# 

# ### Ejercicio 1: Contextualización del problema
# 
# El problema de regresión que abordaremos consiste en predecir el valor de la humedad absoluta en el aire, a partir de varias variables sensadas en el aire (Para más información sobre la base de datos y la contextualización del problema, consulte: http://archive.ics.uci.edu/ml/datasets/air+quality).

# In[ ]:


# tienes ya cargadas las siguientes variables:
print("conjunto de datos", x)
print("variable a predecir", y)


# In[ ]:


#Ejercicio de Codigo
def num_muestras_carac(X):
    """Esta funcion es encargada retornar el numero de muestras
        y caracteristicas del conjunto de datos X

        X: matriz numpy
        retorna:
            numero de caracteristicas (int/float)
            numero de muestras (int/float)
    """
    
    return ()


# In[ ]:


## la funcion que prueba tu implementacion
GRADER_LAB_1_P1.run_test("ejercicio1", num_muestras_carac)


# In[ ]:


#@title Pregunta Abierta
#@markdown  ¿Explique en sus palabras que es un problema de regresión?
respuesta_1 = "" #@param {type:"string"}


# ### Ejercicio 2
# 
# Analice los siguientes métodos de la teoría  de modelos de  *regresión polinomial múltiple*:
# 
# 1. error cuadrático medio (**ECM**)
# 2. modelo de regresión múltiple (**regression**)
# 3. calculo del costo de la regresión (**cost**)
# 4. extension de matriz (**extension_matriz**)
# 
# La siguiente celda contiene la implementación de estas 4 funciones. Analizar y entender su funcionamiento

# In[ ]:


def ECM(Y_est,Y):
    """funcion para calcular el error cuadratico medio
    Y_est: debe contener los valores predichos por el modelo evaluar
    Y: debe contener los valores reales
    retorna: error cuadratico medio
    """
    N = np.size(Y)
    ecm = np.sum((Y_est.reshape(N,1) - Y.reshape(N,1))**2)/(N)
    return ecm 

def regression(X, W):
    """calcula la regresión multiple
    X: los valores que corresponden a las caractersiticas
    W: son los pesos usadados para realizar la regresión
    retorna: valor estimado
    """    
    Yest = np.dot(X,W)    #con np.dot se realiza el producto matricial. Aquí X es dim [Nxd] y W es dim [dx1]
    return Yest           #Esta variable contiene la salida de f(X,W)


def cost(W,X,Y):
    """calcula el costo de la regresion
    W: son los pesos usadados para realizar la regresión
    X: los valores que corresponden a las caractersiticas
    Y: el valor de salida esperadas

    retorna: valor de costo
    """    
    
    m = len(Y)
    y_est = regression(X,W)
    cost = (1/2*m) * np.sum(np.square(y_est-Y))
    return cost

def extension_matriz(X):
    """funcion que realiza la extension de la matriz X
    X: los valores que corresponden a las caractersiticas sin extender
    Y: el valor de salida esperadas
    
    retorna: X_ext: matriz con unos extendidos, Y: maitrz con dimensiones ajustadas
    """
    #Obtenemos las dimensiones antes de exteneder la matriz
    caracterisitcas, muestras = num_muestras_carac(X)
    #Extendemos la matriz X
    unos = np.array([np.ones(muestras)])
    X_ext = np.concatenate((unos.T, X), axis=1)
    X_ext = X_ext.reshape(muestras, caracterisitcas+1)
    return (X_ext)


# In[ ]:


#@title Pregunta Abierta
#@markdown  ¿cual es el objetivo de la extension_matriz? recordar que estamos "ajustando" en una regresión
respuesta_2 = "" #@param {type:"string"}


# Ahora vamos a completar el código de la regla de actualización de los parámetros del algoritmo de <font color='blue'>gradiente_descedente</font>: 
# 
# $$w_j(iter) = w_j(iter-1) - \eta \frac{\partial E(w)}{\partial w_j}$$ 
# 
# recordar que 
# 
# $$ \frac{\partial E(w)}{\partial w_j} = \frac{\partial E({\bf{w}})}{\partial w_j} = \frac{1}{N}\sum_{i=1}^{N}\left( f({\bf{x}}_i,{\bf{w}}) - y_i\right) \frac{\partial }{\partial w_j} f({\bf{x}}_i, {\bf{w}})$$
# 
# recuerda que debe usar las funciones ya implementadas y no usar **ninguna otra libreria**, adicional a las librerias ya pre-cargadas como numpy (la puedes llamar con np.)

# In[ ]:


## Ejercicio de codigo
def gradiente_descendente(X, Y, eta, iteraciones):
    """Gradiente descendente para regresión lineal múltiple
    X: Matriz de datos
    Y: vector con los valores a predecir
    W: Vector de parámetros del modelo
    eta: Taza de aprendizaje

    retorna: W el valor de de los parametros de regresión polinomica
             costos: array con el costo por iteracion
    """
    # nuevamente usamos la función
    # para saber el numero de muestras y caractersiticas
    X_ext = extension_matriz(X)
    caracterisitcas, N = num_muestras_carac(X_ext)
    #Inicializamos el vector de parámetros con ceros y
    W = np.zeros((1,caracterisitcas))
    W = W.reshape(np.size(W), 1)    
    # incializamos vector para almacenar costos
    costos = np.zeros(iteraciones)

    for i in range(iteraciones):
        ## Aca debes completar la funcion! recuerda que solo debes usar numpy (np.funcion_a_usar)
        # o las funciones definidas anteriormente
        # usa la funcion que hace la regresion y que definimos antes
        y_est = 
        f_xw_min_yi = 
        temp =
        # acutaliza
        W = 
        costos[i] = cost(W,X_ext,Y)
        
    return W, costos


# In[ ]:


## la funcion que prueba tu implementacion
GRADER_LAB_1_P1.run_test("ejercicio2", gradiente_descendente)


# ### Ejercicio 3: Entrenamiento
# 
# Con la función implementada vamos a entrenar un modelo y calcular su error de entrenamiento. Antes de realizar esto, debemos separar nuestro conjunto de datos.

# In[ ]:


# esto para lograr reproductibilidad
# de nuestro modelo
random.seed(1)
# usamos nuestra funcion para obtener el numero de muestras
_, N = num_muestras_carac(x)
ind=np.random.permutation(N)
Xtrain = x[ind[0:int(math.ceil(0.7*N))],:]
Xtest = x[ind[int(math.ceil(0.7*N)):N],:]
Ytrain = y[ind[0:int(math.ceil(0.7*N))]]
Ytest = y[ind[int(math.ceil(0.7*N)):N]]


# In[ ]:


#@title Pregunta Abierta
#@markdown  ¿que representa Xtrain, que diferencia hay con Xtest? Explicar en terminos teoricos(no usar el código como justificación)
respuesta_3 = "" #@param {type:"string"}


# Ahora entrena ejecutando la siguiente linea de codigo y verifiquemos el comportamiento del costo

# In[ ]:


W, costo = gradiente_descendente(Xtrain, Ytrain, eta = 0.0001, iteraciones=5)
# graficar iteraciones y el costo
plt.plot(range(5), costo)


# El costo es la medida interna de nuestro algoritmo de optimización sin embargo, para este tipo de problemas al final debemos evaluar que tan bien estamos modelando nuestra salida. Vamos a evaluar nuestro modelo calculando el error cuadrático medio. Para ello vamos crear a una función. Recuerda usar las funciones definidas anteriormente.

# In[ ]:


## Ejercicio de Código
def evaluar_modelo (W, X_to_test, Y_True):
    """ funcion que evalua un modelo de regresión usando el error cuadratico medio

    W: es un matriz con los parametros del modelo entrenados
    X_to_test: conjunto de datos para usar en el evaluamiento del modelo
    Y_True: valores reales para usar en el evaluamiento del modelo

    retorna: el error cuadratico medio
    """
    ## Comienza a completar tu codigo. recuerda usar la funciones ya definidas
    X_to_test_ext = extension_matriz(X_to_test)
    y_est =
    error =

    return(error)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER_LAB_1_P1.run_test("ejercicio3", evaluar_modelo)


# In[ ]:


# y ahora usala para calcular el error, para evaluar el modelo
error_train = evaluar_modelo(W, X_to_test = Xtrain,  Y_True = Ytrain)
print("error en entrenamiento del modelo", error_train)
error_test = evaluar_modelo(W, X_to_test = Xtest,  Y_True = Ytest)
print("error en la evaluación del modelo", error_test)


# In[ ]:


#@title Pregunta Abierta
#@markdown  ¿Que tan bueno es tu modelo? Justifica tu respuesta. Nota si el error tiene notación cientifica
respuesta_4 = "" #@param {type:"string"}


# Hasta ahora lo que hemos realizado es un regresión lineal no polinómica. Nuestro siguiente objetivo es tomar esta regresión y transformarla en polinómica. Comprenda el funcionamiento de la función propuesta

# In[ ]:


#Potencia de polinomio
def potenciaPolinomio(X,grado):
    """calcula la potencia del polinomio
    X: los valores que corresponden a las caractersiticas
    grado: esl grado para realizar la potencia al polinomio
    retorna: el valor de X despues elevarlo al grado del polinimoo indicado
    """
    X2 = X.copy()
    
    if grado != 1:
        for i in range(2,grado+1):
            Xadd = X**i
            X2 = np.concatenate((X2, Xadd), axis=1)
    
    return X2


# ahora debemos usar esta función para completar la siguiente.
# **PISTAS**
# - Usa las funciones previamente construidas
# - Para completar `gradiente_descendente_poly` Tener presente que buscamos realizar este proceso: aplicar la `potenciaPolinomio` ->  aplicar gradiente descendente
# - Para completar `evaluar_modelo_poly` Tener presente que buscamos realizar este proceso: aplicar la `potenciaPolinomio`  -> evaluar el modelo

# In[ ]:


## Ejercicio de codigo
def gradiente_descendente_poly (X, Y, eta, iteraciones, grado):
    """Gradiente descendente para regresión lineal múltiple
    X: Matriz de datos extendida
    Y: vector con los valores a predecir
    W: Vector de parámetros del modelo
    eta: Taza de aprendizaje
    iteraciones: numero de iteraciones maximo para el gradiente
    grado: el valor del polinomio a usar
    
    retorna: W el valor de de los parametros de regresión polinomica
             costo: array con el valor del costo por cada iteracion
            
    """
    ## completa el codigo
    X2 = 
    W, costo =
    return (W, costo)

def evaluar_modelo_poly (W, X_to_test, Y_True, grado):
    """ funcion que evalua un modelo de regresión usando el error cuadratico medio

    W: es un matriz con los parametros del modelo entrenados
    X_to_test: conjunto de datos para usar en el evaluamiento del modelo
    Y_True: valores reales para usar en el evaluamiento del modelo
    grado: grado del polinimio a usar

    retorna: el error cuadratico medio
    """
    ## Comienza a completar tu codigo. recuerda usar la funciones ya definidas
    X2 = 
    error = 
    
    return(error)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER_LAB_1_P1.run_test("ejercicio4", gradiente_descendente_poly)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER_LAB_1_P1.run_test("ejercicio5", evaluar_modelo_poly)


# Entrenemos y evaluemos el comportamiento del costo con la regresion polinomica ¿El cambio fue bueno?

# In[ ]:


# entrenamos
W, costo_poly = gradiente_descendente_poly(Xtrain, Ytrain, eta = 0.0001, iteraciones=5, grado = 2)
plt.plot(range(5), costo_poly)


# In[ ]:


# completa los parametros para evaluar el modelo
error_test = evaluar_modelo_poly(W, X_to_test = Xtest,  Y_True = Ytest, grado = 2)
print("error en la evaluación del modelo", error_test)


# ### Ejercicio 4: Experimentar
# 
# En nuestro primer experimento vamos a evaluar el rendimiento del modelo usando varias tasas de aprendizaje y grados de polinimios. Vamos a dejar por ahora un numero de iteraciones fijas = 5. Para ello completa la siguiente función.
# 
# 

# In[ ]:


## ejercicio de codigo
def experimentar (Xtrain, Xtest, Ytrain, Ytest, tasas, grados):
    """ funcion para realizar experimentos.
    Xtrain: conjunto de datos
    Xtest:
    Ytrain:
    Ytest:
    tasas: Es una lista con los valores númericos de tasas de aprendizaje 
        para realizar los experimentos
    grados: Es una lista con los valores númericos de grados 
        para realizar los experimentos
    retorna: un dataframe con el resultados de los experimentos
    """
    numero_iter = 5

    resultados = pd.DataFrame()
    idx = 0 # indice
    for eta in tasas:
        for grado in grados:
            
            # ignorar el costo
            W, _ =
            error = 
        
            resultados.loc[idx,'grado'] = grado
            resultados.loc[idx,'tasa de aprendizaje'] = eta
            resultados.loc[idx,'ecm'] = error
            idx = idx+1

    return (resultados)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER_LAB_1_P1.run_test("ejercicio6", experimentar)


# In[ ]:


## Ahora usa para verlos los resultados
tasas_aprendizaje = [1e-6, 1e-5, 1e-3, 1e-2]
grados_polinomio = [1,2]
resultados_ex1 = experimentar(Xtrain, Xtest, Ytrain, Ytest, tasas_aprendizaje, grados_polinomio)


# In[ ]:


#para ver los resultados
resultados_ex1


# Si has implementado todo correctamente, parecieria que nuestros entrenamientos no esta logrando buenos resultados (hasta parece haber errores infinitos! o no determinados!). Ahora Entiende la siguiente función. 
# 

# In[ ]:


#Normalizamos los datos
def normalizar(Xtrain, Xtest):
    """ función que se usa para normalizar los datos con
    un metodo especifico
    Xtrain: matriz de datos entrenamiento a normalizar
    Xtest: matriz de datos evaluación a normalizar
    retorna: matrices normalizadas
    """
    
    media = np.mean(Xtrain, axis = 0)
    desvia = np.std(Xtrain, axis = 0)
    Xtrain_n = stats.stats.zscore(Xtrain)
    Xtest_n = (Xtest - media )/desvia
    # si hay una desviacion por cero, reemplazamos los nan
    Xtrain_n = np.nan_to_num(Xtrain_n)
    Xtest_n = np.nan_to_num(Xtest_n)
    return (Xtrain_n, Xtest_n)


# Ahora vuelve a realizar los mismos experimentos pero esta vez usa los valores de salida de la función anterior.

# In[ ]:


Xtrain_n, Xtest_n = normalizar(Xtrain, Xtest)


# In[ ]:


resultados_ex2 = experimentar(Xtrain_n, Xtest_n, Ytrain, Ytest, tasas_aprendizaje, grados_polinomio)
#para ver los resultados
resultados_ex2


# In[ ]:


# ejecuta esta linea de codigo para graficar tus resultados
# aca usamos una libreria llamada seaborn
import seaborn as sns
s = sns.catplot(data = resultados_ex2, x = 'tasa de aprendizaje',
            y = 'ecm',hue ='grado', kind = 'bar', )


# Ten en cuenta el resutaldo de los  dos experimentos y  responde la  siguiente pregunta abierta

# In[ ]:


#@title Pregunta Abierta
#@markdown  ¿Qué proceso hace la normalización sobre los datos? Consulte por qué es necesaria la normalización en el modelo de regresión
respuesta_5 = "" #@param {type:"string"}


# In[ ]:


#@title Pregunta Abierta
#@markdown  ¿cuáles son los tipos de normalización más comunes. ¿Cuál de ellos se aplicó en el laboratorio?
respuesta_6 = "" #@param {type:"string"}


# Finalmente, en nuestro tercer experimento, vamos ver el efecto de las iteraciones sobre el error. completa la siguiente función. Esta vez la tasa de aprendizaje es constante

# In[ ]:


## ejercicio de codigo
def experimentar_2 (Xtrain, Xtest, Ytrain, Ytest, iteraciones, grados):
    """ funcion para realizar experimentos.
    Xtrain: conjunto de datos
    Xtest:
    Ytrain:
    Ytest:
    tasas: Es una lista con los valores númericos de tasas de aprendizaje 
        para realizar los experimentos
    rangos: Es una lista con los valores númericos de grados 
        para realizar los experimentos
    retorna: un dataframe con el resultados de los experimentos
    """
    eta = 1e-2
    resultados = pd.DataFrame()
    idx = 0 # indice
    for iter in iteraciones:
        for grado in grados:
            # ignora el costo
            W , _= 
            error =
        
            resultados.loc[idx,'iteraciones'] = iter
            resultados.loc[idx,'grado'] = grado
            resultados.loc[idx,'ecm'] = error
            idx = idx+1
    return (resultados)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER_LAB_1_P1.run_test("ejercicio7", experimentar_2)


# In[ ]:


num_iters = [1,5,10,50, 100,200, 1000, 2000]
grados_polinomio = [1,2]
# usamos la funcion para evaliar los resultados
resultados_ex3 = experimentar_2(Xtrain_n, Xtest_n, Ytrain, Ytest, num_iters, grados_polinomio )


# In[ ]:


# ejecuta esta linea de codigo para ver raficamente tus resultados
# aca usamos una libreria llamada seaborn
import seaborn as sns
sns.relplot(data = resultados_ex3, x = 'iteraciones',
            y = 'ecm',col ='grado', kind = 'line')


# In[ ]:


#@title Pregunta Abierta
#@markdown  ¿con base a los resultados anteriores, qué efecto tiene el numero de iteraciones en el error?
respuesta_7 = "" #@param {type:"string"}


# In[ ]:


GRADER_LAB_1_P1.check_tests()


# In[ ]:


#@title Integrantes
codigo_integrante_1 ='' #@param {type:"string"}
codigo_integrante_2 = ''  #@param {type:"string"}


# ----
# esta linea de codigo va fallar, es de uso exclusivo del los profesores
# 

# In[ ]:


GRADER_LAB_1_P1.grade()

