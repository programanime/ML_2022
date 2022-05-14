#!/usr/bin/env python
# coding: utf-8

# **Recuerda que una vez abierto, Da clic en "Copiar en Drive", de lo contrario no podras alamancenar tu progreso**
# 
# Nota: no olvide ir ejecutando las celdas de código de arriba hacia abajo para que no tenga errores de importación de librerías o por falta de definición de variables.

# In[ ]:


#configuración del laboratorio
# Ejecuta esta celda!
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
# for local 
#import sys ; sys.path.append('../commons/utils/')
get_ipython().system('wget https://raw.githubusercontent.com/jdariasl/ML_2020/master/Labs/commons/utils/general.py -O general.py')
from general import configure_lab2
configure_lab2()
from lab2 import *
GRADER, x, y = part_1()


# # Laboratorio 2 - Parte 1. KNN para un problema de clasificación

# ### Ejercicio 1: Contextualización del problema
# 
# 
# Usaremos el dataset iris para el problema de clasificación. En el UCI Machine Learning Repository se encuentra más información en el siguiente [link](https://archive.ics.uci.edu/ml/datasets/iris) .

# In[ ]:


print("muestra de los 5 primeros renglones de x:\n", x[0:5, :])
print("muestra de los 5 primeros renglones de y:\n", y[0:5])
print ("¿el resultado de esta instrucción que información nos brinda?", x.shape[0])
print ("¿el resultado de esta instrucción que información nos brinda?", x.shape[1])
print ("¿el resultado de esta instrucción que información nos brinda?", len(np.unique(y)))


# En un problema de clasificación de más de una clase, tener un desbalance de muestras puede ser perjudicial para el proceso de entrenamiento. Vamos a crear una función para verificar el número de muestras por clases.

# In[ ]:


#Ejercicio de código
def muestras_por_clases (Y):
    """Funcion que calcula el número de muestras por cada clase
    Y: vector de numpy con las etiquetas de las muestras del conjunto X
    retorna: diccionario [int/float:int/float] 
        con la estructura:{etiquetaclase1: número de muestras clase1, etiquetaclase2: número de muestras clase2}
    """
    dicto = {}
    ## Pista se puede asginar keys a diccionario: dict[etiqueta] = valor
    for 
      

    return (dicto)



# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio1", muestras_por_clases)


# In[ ]:


# con esta linea de codigo puedes ver la dsitribucion de forma grafica
fig, ax = plt.subplots()
ax.bar(muestras_por_clases(y).keys(), muestras_por_clases(y).values())
ax.set_title("número de muestras por clase")
ax.set_xlabel("etiqueta de clase")
ax.set_ylabel("# muestras por clase")
ax.set_xticks(list(muestras_por_clases(y).keys()))
plt.show()


# In[ ]:


#@title Pregunta Abierta
#@markdown  ¿dependiendo de los resultados de la informacion anterior, como calificaria la distribución de clases desde el punto de vista de la factibilidad de usarlo este dataset para el problema planteado?
respuesta_1 = "" #@param {type:"string"}


# ### Ejercicio 2: Completar código KNN
# 
# Recordemos los conceptos vistos en teoria para los modelos basados en los K-vecimos más cercanos. En este ejercicio vamos a escribir la función que implementa este modelo. Pero primero vamos a definir la función que nos ayudara calcular el error de clasificación.

# In[ ]:


def ErrorClas(Y_lest, Y):
    """funcion que calcula el error de clasificación
    Y_lest: numpy array con la estimaciones de etiqueta
    Y: etiquetas reales
    retorna: error de clasificación (int)
    """
    error = 1 - np.sum(Y_lest == Y)/len(Y)
    
    return error


# Ahora si es hora del ejercicio. Ten en cuenta lo siguiente:
# 
# <b>Pistas</b>
# 
# 1. Para el cáculo de la distancia entre vectores existen varias opciones:
#     1. usar la función la distancia entre matrices `scipy.spatial.distance.cdist`([Ejemplo](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist))--esta puede ser usada directamente como `cdist(...)`. Entiende la salida de esta función. Al usarla, se logra un rendimiento superior.
#     2. usar la función la distancia euclidiana `scipy.spatial.distance.euclidean`([Ejemplo](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html))--pueder acceder a ella directamente como `euclidean`. Aca debe pensar en un algoritmo elemento a elemento, por lo tanto menos eficiente.
# 2. También serán de utilidad las funciones `np.sort` y `np.argsort`.
# 3. ten presente que la moda es una operación que calcula el valor más común. En el [notebook ya se encuentra cargada esta operacion](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html), es posible usarla de esta manera : `mode(y)`

# In[ ]:


#ejercicio de codigo
def KNN_Clasificacion(k, X_train, Y_train, X_test):
    """ Funcion que implementa el modelo de K-Vecino mas cercanos
        para clasificación
    k (int): valor de vecinos a usar
    X_train: es la matriz con las muestras de entrenamiento
    Y_train: es un vector con los valores de salida pra cada una de las muestras de entrenamiento
    X_test: es la matriz con las muestras de validación
    retorna: las estimaciones del modelo KNN para el conjunto X_test 
             esta matriz debe tener un shape de [row/muestras de X_test] 
             y las distancias de X_test respecto a X_train, estan matrix
             debe tener un shape de [rows de X_test, rows X_train]
             lo que es lo mismo [muestras de X_test, muestras de X_train]
    """
    if k > X_train.shape[0]:
        print("k no puede ser menor que las muestras de entrenamiento")
        return(None)
    distancias = 
    Yest = np.zeros(X_test.shape[0])
    
    for 
        
    
    return (Yest, distancias) 
  


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio2", KNN_Clasificacion)


# ### Ejercicio 3: Experimentos de KNN
# 
# Ahora vamos a probar nuestro algoritmo. Pero antes de esto vamos a tener que dividir nuestro conjunto de datos, vamos a usar una función llamada train_test_split de la libreria sklearn. [Aca puedes ver la ayuda](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). Entiende su funcionamiento. Vamos a usarla para crear una función con una propoción fija  de 80%-20% entre nuestro conjunto de entrenamiento y de pruebas.
# 
# **NOTA**
# 
# cuando se este usando SKlearn, se quiere incentivar la lectura de la documentación, por lo tanto el calificador va buscar siempre que se llamen de manera explicita los parametros que se estan usando. Ejemplo si se va usar un parametro llamado `shuffle` se debe llamar como `funcion_a_usar(shuffle = True)`.

# In[ ]:


#ejercicio de codigo
def train_test_split_fix(X,Y):
    """funcion que divide el conjunto de datos en
        entrenamiento y pruebas
        usando un proporcion fija de 20 %
        para el conjunto de pruebas.

    X: matriz de numpy con las muestras y caractersiticas
    Y: matriz de numpy con las las etiquetas reales
    retorna:
        Xtrain: conjunto de datos para entrenamiento
        Xtest: conjunto de datos para pruebas
        Ytrain: conjunto  de etiquetas para entrenamiento
        Ytest: conjunto de etiquetas para prueba 
    """
    Xtrain, Xtest, Ytrain, Ytest = ( ...)

    return (Xtrain, Xtest, Ytrain, Ytest)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio3", train_test_split_fix)


# Vamos a proceder a experimentar. Para ello vamos a crear una función que realiza los experimentos usando las funciones previamente construidas. En el código se hace uso de la función [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), para normalizar los datos.

# In[ ]:


#Ejercicio de código
def experimentar (ks, X, Y):
    """Función que realiza los experimentos con knn usando
       una estrategia de validacion entrenamiento y pruebas
       
    ks: List[int/float] lista con los valores de k-vecinos a usar
    X: matriz de numpy conjunto con muestras y caracteristicas
    Y: vector de numpy con los valores de las etiquetas
    
    retorna: dataframe con los resultados
    """

    # dividimos usando la función
    Xtrain, Xtest, Ytrain, Ytest = train_test_split_fix(X,Y)

    # se llama el objeto
    scaler = StandardScaler()
    # Se calculan los parametros
    scaler.fit(Xtrain)
    # se usa el objeto con los parametros calculados
    # realizar la normalización
    Xtrain= scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    resultados = pd.DataFrame()
    idx = 0
    for k in ks:
        # iteramos sobre la lista de k's
        resultados.loc[idx,'k-vecinos'] = k
        Yest, dist =
        errorTest = 
        resultados.loc[idx,'error de prueba'] = 
        idx+=1

    return (resultados)


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿qué tipo de normalización ejecuta la función `StandardScaler`?
respuesta_2 = "" #@param {type:"string"}


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio4", experimentar)


# Ahora ejecuta los experimentos con k = 2,3,4,5,6,7,10

# In[ ]:


resultados = experimentar ([2,3,4,5,6,7,10], x, y,)
resultados


# ### Ejercicio 4: ventana de Parzen
# 
# Ahora vamos a utilizar el metodo de ventana de parzen. Recordar de las clases teoricas, que para aplicar este metodo, debemos usar una función kernel. En la siguiente celda se proponen dos funciones para:
# 1. calculo de un kernel gausiano
# 2. calculo de la ventana de parzen, es decir el termino: $ \sum_{i=1}^{N} K(u_i)$, siendo $\;\; u_i = \frac{d({\bf{x}}^*,{\bf{x}}_i)}{h}$ y la función $K$ el kernel gausiano

# In[ ]:


def kernel_gaussiano(x):
    """Calcula el kernel gaussiano de x
    x: matriz/vector de numpy
    retorna: el valor de de kernel gaussiano
    """
    return np.exp((-0.5)*x**2)

def ParzenWindow(x,Data,h):
    """"ventana de parzen
    x: vector con representando una sola muestra
    Data: vector de muestras de entrenamiento
    h: ancho de la ventana de kernel
    retorna: el valor de ventana de parzen para una muestra
    """
    h = h
    Ns = Data.shape[0]
    suma = 0
    for k in range(Ns):
        u = euclidean(x,Data[k,:])
        suma += kernel_gaussiano(u/h)
    return suma


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿qué objetivo tiene la función kernel? Contestar en el contexto del método de la ventana de parzen 
respuesta_3 = "" #@param {type:"string"}


# Una vez entendidos los anteriores metodos, los vamos a usar para resolver el ejercicio de código.

# In[ ]:


#Ejercicio de código
def parzenClass(h, X_train, Y_train, X_test):
    """ Funcion que implementa metodo de ventana de parzen para
        para clasificación
    h (float): ancho de h de la ventana
    X_train: es la matriz con las muestras de entrenamiento
    Y_train: es un vector con los valores de salida pra cada una de las muestras de entrenamiento
    X_test: es la matriz con las muestras de validación
  
    retorna: - las estimaciones del modelo parzen para el conjunto X_test 
              esta matriz debe tener un shape de [row/muestras de X_test]
             - las probabilidades de la vetana [row/muestras de X_test, número de clases]  
    """
        
    Yest = np.zeros(X_test.shape[0])
    clases = np.unique(Y_train)
    fds_matrix = np.zeros((X_test.shape[0], len(clases)))
    
    
    ## pista: recuerde el termino que acompaña al sumatoria (N)
    
    for n, sample in enumerate (X_test):
      
        for label in clases:
           
           
       
    
    

    #Debe retornar un vector que contenga las predicciones para cada una de las muestras en X_val, en el mismo orden.  
    return Yest, fds_matrix


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio5", parzenClass)


# ### Ejercicio 5 - Experimentos con Parzen
# 
# Ahora vamos a realizar los experimentos, recordar usar la misma metodologia de validación, usando la función previamente creada.

# In[ ]:


#ejercicio de codigo
def experimentarParzen (hs, X, Y):
    """Función que realiza los experimentos con knn usando
       una estrategia de validacion entrenamiento y pruebas
       
    hs: List[int/float] lista con los valores de h a usar
    X: matriz de numpy conjunto con muestras y caracteristicas
    Y: vector de numpy con los valores de las etiquetas
    
    retorna: dataframe con los resultados, debe contener las siguientes columnas:
        - el ancho de ventana, el error medio de prueba, la desviacion estandar del error
    """
    
    
    resultados = pd.DataFrame()
    idx = 0
    Xtrain, Xtest, Ytrain, Ytest = ...(X,Y)
    scaler = StandardScaler()
    #normalizamos los datos
    scaler.fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    # iteramos sobre los valores de hs
    for h in hs:
        
        Yest, probabilidades = 
        errorTest = 
    
        resultados.loc[idx,'ancho de ventana'] = h 
        resultados.loc[idx,'error de prueba'] = 

        idx+=1
    return (resultados)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio6", experimentarParzen)


# In[ ]:


hs = [0.05, 0.1, 0.5, 1, 2, 5, 10]
experimentos_parzen = experimentarParzen(hs,x,y)
experimentos_parzen


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿En el método de ventana de parzen, porqué no hay necesidad de definir el número de vecinos cercanos? 
respuesta_4 = "" #@param {type:"string"}


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿por qué el KNN y la ventana de parzen son modelos no parámetricos? 
respuesta_5 = "" #@param {type:"string"}


# In[ ]:


GRADER.check_tests()


# In[ ]:


#@title Integrantes
codigo_integrante_1 ='' #@param {type:"string"}
codigo_integrante_2 = ''  #@param {type:"string"}


# ----
# esta linea de codigo va fallar, es de uso exclusivo del los profesores
# 

# In[ ]:


GRADER.grade()

