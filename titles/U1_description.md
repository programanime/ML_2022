# INTRODUCCIÓN AL MACHINE LEARNING

## ¿Qué es Machine Learning?  <br/>

**Definiciones básicas y aplicaciones típicas**:  [Clase_02.pdf](https://drive.google.com/file/d/1k8XDXGFka_LWfty32M8BcOT7rFtbpRkS/view?usp=sharing) <br/>Se introduce el concepto de Machine Learning y se discuten algunas aplicaciones típicas.

Se presentan los tipos de aprendizaje automático más comunes: supervisado, no supervisado, auto supervisado y por refuerzo, así como la nomenclatura matemática básica que se usa para representar los problemas supervisados y no supervisados.

Se explica la diferencia entre los problemas de aprendizaje supervisado conocidos como clasificación y regresión, así como sus formas de representación gráfica. 


## Modelos básicos de aprendizaje  <br/> [Clase 03.pdf](https://github.com/jdariasl/ML_2020/blob/5ab6a9aedaec7836a8ddd2b0ab2484d44ab4602d/Clase%2002%20-%20Regresi%C3%B3n%20lineal%20y%20regresi%C3%B3n%20log%C3%ADstica.ipynb)

**Regresión múltiple**: <br/> Se repasa el modelo de regresión múltiple y de los tres componentes básicos del modelo de acuerdo con las ideas discutidas en la sesión anterior, desde una perspectiva de alto nivel.

**Algoritmo de gradiente descendente**:  <br/> Se continúa con el análisis de la regresión múltiple desde perspectivas de mediano y bajo nivel, y para esta última se explica el funcionamiento del algoritmo de gradiente descendente.

**Regresión logística**:  <br/> Con base en un modelo de regresión y en el algoritmo de gradiente descedente se introduce el modelo de regresión logística que sirve para resolver problemas de clasificación y se explica su función de costo conocida como <em>cross-entropy</em>.

**Análisis del entrenamiento de una regresión logística**:  <br/> Se presentan algunos resultados de la ejecución del proceso de entrenamiento de un modelo de regresión logística, para ayudar a comprender el proceso de minimización de la función de costo a partir del algoritmo de gradiente descendente, la modificación iterativa de la frontera de decisión hasta alcanzar la ubicación óptima (de acuerdo con la función de costo) y su correspondiente interpretación en el espacio de búsqueda.

**Fronteras no lineales con regresión logística**:  <br/> Se discute cómo se pueden obtener fronteras de decisión no lineales usando el modelo de regresión logística y los ajustes que deben realizarse para su implementación.

## Modelos discriminativos vs modelos generativos   <br/>

**Funciones discriminantes Gausianas**: <br/> Se discuten los modelos generativos los cuales utilizan un principio de funcionamiento diferente para resolver los problemas de clasificación y se presenta el modelo de funcionaes discriminantes Gausianas, además se explica el criterio de Máxima Verosimilitud a partir del cual se estiman los parámetros del modelo.

**Modelos discriminante lineal, discriminante cuadrático y Naïve Bayes**:  <br/> Se presentan varios modelos derivados de las funciones discriminantes Gausianas a partir de diferentes configuraciones de la matriz de covarianza del modelo.
