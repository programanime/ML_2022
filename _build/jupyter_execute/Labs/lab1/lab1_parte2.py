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
from general import configure_lab1_p2
configure_lab1_p2()
from lab1 import *
GRADER, x, y = part_2()
y = y.reshape(np.size(y), 1)


# # Laboratorio 1 - Parte 2. Regresión logística
# 
# 

# ### Ejercicio 1: Contextualización del problema
# 
# En esta sesión de laboratorio, vamos a resolver un problema de clasificación. Los variables que vamos a usar ya se encuentran cargadas:
# 
# 

# In[ ]:


# tienes ya cargadas las siguientes variables:
print("conjunto de datos, muestra \n",x[range(10), :] )
print("")
print(" muestra de etiquetas a predecir \n", y[range(10)])


# In[ ]:


#Ejercicio de Codigo
def clases_muestras_carac(X, Y):
    """Esta funcion es encargada retornar el numero clases, muestras 
        y caracteristicas del conjunto de datos X y Y

        X: matriz numpy con el conjunto de datos para entrenamiento
        Y: matriz numpy con el conjunto de etiquetas
        retorna:
            numero de clases (int/float)
            numero de muestras (int/float)
            numero de caracteristicas (int/float)
    """
    ##Pista: es de utilidad el metodo np.unique ?
    N,nf = 
    clases = 
    
    return (clases,N,nf)


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio1", clases_muestras_carac)


# En los problemas de clasificación, que lo permiten, es de utilidad visualizar los datos. De esta manera se puede determinar que modelos o algortimos pueden tener mejor rendimiento. En la siguiente función, se debe: 
# 1. graficar los datos usando la función [scatter](https://matplotlib.org/gallery/shapes_and_collections/scatter.html) de matplotlib. [Recuerda consultar la documentación de la función](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter)
# 2. Asginar un color Map diferente al de defecto. [Aca pueder los valores posibles](https://matplotlib.org/stable/tutorials/colors/colormaps.html).

# In[ ]:


#Ejercicio de Codigo
def scatter_plot(X, Y):
    """Esta funcion es encargada de graficar usando un scatter plot
       un problema de clasificacion.

        X: matriz numpy con el conjunto de datos para entrenamiento.
           esta debera ser usada para los ejes del grafico. puede asumir
           que solo va tener dos columnas
        Y: matriz numpy con el conjunto de etiquetas. Debera se usada
           para mostrar en diferentes colores, las etiquetas de cada una
           de las muestras
        retorna:
            No retorna nada, el grafico debe aparecer
    """
    ## puedes acceder con plt a la funcion adecuacada
    ## Pista: recuerda como indexar matrices
    ## Consulta como pasar el Color Map
    plt.
    # para mostrar el grafico
    plt.show()
   
    return (None)


# In[ ]:


## la funcion que prueba tu implementacion
# ignora los graficos que se muestran 
GRADER.run_test("ejercicio2", scatter_plot)


# In[ ]:


# usarla para ver el grafico
# se debe ver dos nubes de puntos diferenciadas
scatter_plot(x,y)


# In[ ]:


#@title Pregunta Abierta
#@markdown  ¿El problema es linealmente separable? justifique su respuesta 
respuesta_1 = "" #@param {type:"string"}


# ### Ejercicio 2: entrenamiento
# 
# En este laboratorio se va a realizar un procedimiento análogo al laboratorio anterior, pero con el modelo de *regresión logística* que sirve para resolver problemas de clasificación (en principio biclase).
# 
# ¿Cómo se relacionan los siguientes conceptos a la luz del modelo de regresión logística? 
# 
# 1. Función de activación 
# 2. Extensión de matriz
# 2. Modelo de regresión logística
# 3. Potencia del polinomio 
# 4. El cálculo del error en clasificación 
# 5. El gradiente descendente
# 
# Vamos a completar la función del sigmoide:

# In[ ]:


#Ejercicio de Código
def sigmoidal(z):
    """Función de activación Sigmoidal

    z: es la varible a la que se le va aplicar el sigmoide.
       es un array numpy de uan sola dimension
    retorna: el valor del sigmiode

    """
    #Complete la siguiente línea con el código para calcular la salida de la función sigmoidal
    # ¿Que metodo de np...{} es de utilidad?
    s = np.
    
    return s


# In[ ]:


## la funcion que prueba tu implementacion
GRADER.run_test("ejercicio3", sigmoidal)


# La siguiente celda se sugiere la implementación de 3 métodos que nos ayudaran a implementar los otros conceptos. 
# 
# 1. Modelo de regresión logística(Esta función usa nuestra `sigmoidal`)
# 2. Extensión de matriz
# 2. Potencia del polinomio 
# 4. Cálculo del error en clasificación
# 
# Debemos comprender que hacen estas funciones para determinar donde poder usarlas mas adelante. Luego de ellos, ejecuta la celda para cargarlas.

# In[ ]:


def logistic_regression(X, W):
    """calcula la regresión logistica
    X: los valores que corresponden a las caractersiticas
    W: son los pesos usadados para realizar la regresión
    retorna: valor estimado por la regresion
    """
    #Con np.dot se realiza el producto matricial. Aquí X (extendida) tiene dim [Nxd] y W es dim [dx1]
    Yest = np.dot(X,W)
    Y_lest = sigmoidal(Yest)
   
    return Y_lest    #Y estimado: Esta variable contiene ya tiene la salida de sigm(f(X,W))

def extension_matriz(X):
    """funcion que realiza la extension de la matriz X
    X: los valores que corresponden a las caractersiticas sin extender
    retorna: X_ext: matriz con unos extendidos
    """
    #Obtenemos las dimensiones antes de exteneder la matriz
    muestras,caracterisitcas =X.shape
    #Extendemos la matriz X
    unos = np.array([np.ones(muestras)])
    X_ext = np.concatenate((unos.T, X), axis=1)
    X_ext = X_ext.reshape(muestras, caracterisitcas+1)
    return (X_ext)


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

def cost_logistic(Y_lest, Y):
    """calculo del error logistico
       Si es diferente el Y_estimado con el Y_real cuenta como un error
       Y_lest: numpy array con los valores reales estimados
       Y:  numpy array  valor con los valores reales de las etiquetas
       retorna: costo logistico  -- numpy array
    """
    f1 = Y*np.log(Y_lest)
    f2 = (1-Y)*np.log(1-Y_lest)
    error = -np.sum(f1+f2)/Y.shape[0]
    return error


# In[ ]:


# ten en cuenta la salida de esta prueba para responder la pregunta abierta siguiente
vector = np.array([[1,4]])
print(vector)
potenciaPolinomio(vector, 3)


# In[ ]:


#@title Pregunta Abierta
#@markdown  Analizando lo anterior y asumiendo que el vector es representado como [x1, x2] ¿Cual seria la representación de `potenciaPolinomio(vector, 3)`? (usa ** para representar potencia, i.e x1**2 simboliza x1 elevado al cuadrado)
respuesta_2 = "" #@param {type:"string"}


# recordando lo aprendido anteriormente, dividimos nuestro cojunto de datos y normalizamos.
# **SOLO EJECUTAR UNA VEZ**

# In[ ]:


#Dejamos algunas muestras para el proceso de entrenamiento y otras para evaluar qué tan bueno fue el aprendizaje del modelo
random.seed(1)
N = x.shape[0]
ind=np.random.permutation(N)
Xtrain = x[ind[0:int(math.ceil(0.7*N))],:]
Xtest = x[ind[int(math.ceil(0.7*N)):N],:]
Ytrain = y[ind[0:int(math.ceil(0.7*N))]]
Ytest = y[ind[int(math.ceil(0.7*N)):N]]
# normalizamos
Xtrain, Xtest = normalizar(Xtrain, Xtest)


# Ahora vamos a completar el código de la regla de actualización de los parámetros del algoritmo de **gradiente_descedente** 
# 
# 
# $$w_j(iter) = w_j(iter-1) - \eta \frac{\partial E(w)}{\partial w_j}$$ 
# 
# recordar que aca queremos reducir el costo logistico, pero la actualización de los pesos sigue siendo equivalente a la regresión vista anteriormente, solo que agregando la función $g$ que corresponde al sigmoide.
# 
# $$ \frac{\partial E(w)}{\partial w_j} = \frac{\partial J({\bf{w}})}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N}\left( g(f({\bf{x}}_i,{\bf{w}})) - t_i\right)x_{ij}$$
# 
# Debemos tener presente:
# - Usar las funciones ya implementadas y no usar **ninguna otra libreria** adicional a las librerias ya pre-cargadas como numpy.
# - Dentro de nuestra función **vamos incluir una transformación polinómica**, por lo tanto el proceso a implementar puede ser descrito en los siguientes pasos:
#     - Aplicar la transformación polinómica
#     - Extender la matriz
#     - Inicializar $w$
#     - Y por cada iteración realizar el cálculo para actualizar $w$

# In[ ]:


#ejercicio de codigo
def gradiente_descendente_logistic_poly(X,Y,grado,eta, iteraciones):
    """Gradiente descendente para regresión lineal múltiple
    X: Matriz de datos extendida
    Y: vector con los valores a predecir
    W: Vector de parámetros del modelo
    eta: Taza de aprendizaje
    grado: grado para usar en la transformacion polinomica
    iteraciones: numero de iteraciones maxima

    retorna: W el valor de de los parametros de regresión polinomica
    """
    # aplicamos la transformación
    X2 = potenciaPolinomio(X,grado)
    
    # realizamos la extensión
    X2_ext = 
    
    #Tomamos el número de variables del problema leugo de la transformacion
    d = X2_ext.shape[1]
    #Tomamos el número de muestras de la base de datos
    N = X2_ext.shape[0]
    #Inicializamos w
    W = np.zeros(d)
    W = W.reshape(np.size(W),1)
    costos = np.zeros(iteraciones)
      
    for iter in range(iteraciones):
       
        #Aquí debe completar el código con la regla de actualización de los parámetros W para regresión
        #logística. Tenga en cuenta los nombres de las variables ya creadas: W, X, Y
 
        costo = cost_logistic(,)
        W =  
        #adicionamos el costo por cada iteracion
        costos[iter] = costo


    print("costo despues de finalizar las iteraciones", costo)
    return W, costos


# In[ ]:


## la funcion que prueba tu implementacion
# ignorar los print que se ejecutan
GRADER.run_test("ejercicio4", gradiente_descendente_logistic_poly)


# Veamos de manera preliminar, si nuestro algortimo de gradiente esta optimizando el costo. 
# Vamos entrenar con 100 iteraciones y vamos a graficar el costo logistico y verificar como cambia de acuerdo a las iteraciones.
# ¿Que pasa si cambias el grado?

# In[ ]:


iteraciones = 100
w, costos_logistico = gradiente_descendente_logistic_poly(Xtrain,Ytrain,grado = 1, eta = 10, iteraciones = iteraciones)
print("este las dimensiones de w son:", w.shape)
plt.plot(range(iteraciones), costos_logistico)


# In[ ]:


#@title Pregunta Abierta
#@markdown  ¿con base a los resultados anteriores, qué efecto tiene el grado en el comportamiento del costo? justifique
respuesta_3 = "" #@param {type:"string"}


# Como habiamos visto antes, el costo es nuestra medida de optimización, pero al final debemos evaluar que tambien esta realizando la tarea de clasificar. Entiende la siguiente función. Prestemos especial atención al bloque `if-else`. 

# In[ ]:


def evaluar_modelo (W, X_to_test, Y_True, grado):
    """ funcion que evalua un modelo de clasificación

W: es un matriz con los parametros del modelo entrenados
    X_to_test: conjunto de datos para usar en el evaluamiento del modelo
    Y_True: valores reales para usar en el evaluamiento del modelo
    grado: valor del polinomio a usar

    retorna: el error de clasificación.
    """
    X2 = potenciaPolinomio(X_to_test,grado)
    # realizamos la extensión
    X2_ext = extension_matriz(X2)
    Y_EST = logistic_regression(X2_ext,W)
    #Se asignan los valores a 1 o 0 según el modelo de regresión logística definido
    for pos, tag in enumerate (Y_EST):
        
        if tag >= 0.5:
            Y_EST[pos] = 1
        else:
            Y_EST[pos] = 0
            
    error = 0
    
    for ye, y in zip(Y_EST, Y_True):
        if ye != y:
            error += 1
    error_clasificacion =  error/np.size(Y_True)
    return(error_clasificacion)


# Probemos la función al evaluar el w obtenido. Tener presente que para la evaluación usamos el X y Y adecuado. Recuerda usar el grado de acuerdo a la ultima ejecución.

# In[ ]:


# recuerda que si entrenaste con grado = 2 debes asignar el mismo valor.
error_test = evaluar_modelo(w, Xtest, Ytest, grado = 1)
print("error en el conjunto de pruebas", error_test)


# ### Ejercicio 3: Experimentar

# En nuestro primer experimento vamos a evaluar el rendimiento del modelo usando varias tasas de aprendizaje y grados de polinimios. Vamos a dejar por ahora un numero de iteraciones fijas = 30. Para ello completa la siguiente función. Recuerda usar las funciones anteriores.

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
    numero_iter = 30

    resultados = pd.DataFrame()
    idx = 0 # indice
    for eta in tasas:
        for grado in grados:
            # ignorar el costo
            W, _ = 
            error_entrenamiento = 
            error_prueba = 
            resultados.loc[idx,'grado'] = grado
            resultados.loc[idx,'tasa de aprendizaje'] = eta
            resultados.loc[idx,'error_entreamiento'] = error_entrenamiento
            resultados.loc[idx,'error_prueba'] = error_prueba
            idx = idx+1

    return (resultados)


# In[ ]:


## la funcion que prueba tu implementacion
# ignorar los prints de "costo despues de finalizar"!
GRADER.run_test("ejercicio5", experimentar)


# In[ ]:


tasas = [10.0, 1.0, 0.1, 0.001]
grados = [1,2,3]
resultados = experimentar (Xtrain, Xtest, Ytrain, Ytest, tasas, grados)


# In[ ]:


# para ver los resultados
resultados


# In[ ]:


#@title Pregunta Abierta
#@markdown  ¿con base a los resultados anteriores, qué efecto tiene la  tasa de aprendizaje en los errores de entrenamiento y de prueba? justifique
respuesta_4 = "" #@param {type:"string"}


# Vamos entrenar nuevamente pero solo con los mejores parámetros. Si hay parametros empatados, el modelo que tenga menos parámetros deberia ser el mejor (recuerda que cuando aumentamos el grado del polinomio aumentamos el número de parametros del modelo).

# In[ ]:


# puedes usar el siguiente código para ordenar los resultados y ver los 3 primeros
# resultados, usa esta salida, para ver cuales fueron los mejores parámetros
resultados.sort_values(by = ['error_prueba', 'grado'], ascending = True).head(3)


# In[ ]:


#@title Pregunta Abierta
#@markdown ¿por qué se uso el error de prueba para ordenar la tabla de resultados en lugar del error de entrenamiento?
respuesta_5 = "" #@param {type:"string"}


# Entrenemos con los mejores parámetros

# In[ ]:


# ignoramos el costo!
W,_= gradiente_descendente_logistic_poly(Xtrain,Ytrain,grado =  ,eta =  , iteraciones = 20)
print("estos son los pesos para el modelo entrenando \n", W)


# Debemos recordar que el modelo creado se puede resumir en una ecuación usando los pesos de $w$. En el  siguiente ejemplo la ecuación estaria dada como: 2.0 + 3.0*x1 + 4.0*x2, para el caso del polinomio grado 1.
# 
# ![vectorization](https://raw.githubusercontent.com/jdariasl/ML_2020/master/Labs/commons/images/frontera_logistic.jpg)
# 
# 
# Usando los valores del ultimo $w$ entrenado podriamos construir la función que define nuestra frontera.
# 

# In[ ]:


#@title Pregunta Abierta
#@markdown Escribe el modelo completo con sus variables y coeficientes de f(**x**,**w**) con la mejor frontera de decisión que encontró. Recuerda tener presente el grado del polinomio.
respuesta_6 = "0.0x1..." #@param {type:"string"}


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

