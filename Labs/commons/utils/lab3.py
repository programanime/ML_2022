"""
Este archivo es generado automaticamente.

###### NO MODIFICAR #########

# cualquier alteración del archivo
# puede generar una mala calificación o configuracion
# que puede repercutir negativamente en la 
# calificación del laboratorio!!!!!

###### NO MODIFICAR #########
"""
import seaborn as sns
from imports import *
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, load_wine
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (KFold, LeaveOneOut, ShuffleSplit,
                                     StratifiedKFold)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier



def classes_wines():
    return (['clase 1', 'clase 2', 'clase 3'])


def features_wines():
    return (

        [
            'Alcohol',
        'Malic acid',
        'Ash',
        'Alcalinity of ash  ',
        'Magnesium',
        'Total phenols',
        'Flavanoids',
        'Nonflavanoid phenols',
        'Proanthocyanins',
        'Color intensity',
        'Hue',
        'OD280/OD315 of diluted wines',
        'Proline'
        ]
    )


def plot_digits(data, size= (10, 10), figsize = (8, 8)):
    fig, ax = plt.subplots(size[0], size[1], figsize=figsize,
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)

def plot_digit(data):
    fig, ax = plt.subplots(1, 1, figsize=(3,3),
                           subplot_kw=dict(xticks=[], yticks=[]))

    im = ax.imshow(data.reshape(8, 8), cmap='binary')
    im.set_clim(0, 16)

def plot_digit_plano(data, y, numero_digitos):
    data_to_plot = pd.DataFrame(data[0:numero_digitos, 0:2], columns=['Variable reducida # 1', 'Variable reducida # 2'])
    data_to_plot['clase'] = y[0:numero_digitos]
    g = sns.relplot(data = data_to_plot, x = 'Variable reducida # 1', 
                    y='Variable reducida # 2', 
                    hue= 'clase', style= 'clase',
                    s = 400, height = 4)
    g.fig.suptitle("Muestras representadas en plano")


@unknow_error
def test_get_muestras_by_cv(func):

    Y = np.random.choice(3,100)
    cv1 = func(metodo = 1, X=np.ones((100,2)),Y=Y)
    cv2= func(metodo = 2, X=np.ones((100,2)),Y=Y) 
    cv3= func(metodo = 3, X=np.ones((100,2)),Y=Y)  

    met_1 = cv2['numero de muestras entrenamiento'].mean() <= cv1['numero de muestras entrenamiento'].mean()

    met_2 = cv2['numero de muestras entrenamiento'].mean() <= cv3['numero de muestras entrenamiento'].mean()

    met_3 = cv3['numero de muestras entrenamiento'].mean() == 33

    tests = {'recuerda dividir en 3 folds': (cv1.shape[0] == len(np.unique(Y))*3 ) and (cv2.shape[0] == len(np.unique(Y))*3) ,
             'recuerda que metodo corresponde a la metodologia de validacion': met_1 and met_2 and met_3
             }

    test_res = ut.test_conditions_and_methods(tests)
    code_to_look = ['ShuffleSplit', 'StratifiedKFold', 'LeaveOneOut', "split(X=X,y=Y)", "split(X=X)"]
    res2 = ut.check_code(code_to_look, func)

    return (res2 and test_res)

@unknow_error
def test_GMMClassifierTrain(func):
    y1 = np.random.choice(3,20)
    g1 = func(M=2, tipo= 'full', X=np.random.rand(20,2), Y = y1)
    g2 = func(M=2,  tipo= 'full', X=np.random.rand(20,2), Y = np.random.choice(3,20),)
    g3 = func(M= 2,  tipo= 'full', X=np.random.rand(20,2), Y = np.random.choice(3,20))
    g4 = func(M=2, tipo=  'spherical',X= np.random.rand(10,2), Y= np.random.choice(2,10))
    t1 =  len(np.array([np.mean(m.means_) for m in g1.values()])) == len(np.unique([np.mean(m.means_) for m in g1.values()]))

    code_to_look = ['n_components=', 'covariance_type=']
    res2 = ut.check_code(code_to_look, func)

    tests = {'debes entrenar un GMM por cada clase (valor unico de Y)': t1,
             'la clave del dict debe ser la etiqueta de Y':  (list(g1.keys()) == np.unique(y1)).all(),
             'no debes dejear codigo estatico.':  g1 != g2 }

    return (ut.test_conditions_and_methods(tests) and res2)

@unknow_error
def test_GMMClassfierVal(func):
    yy = np.random.choice(2, 20)
    xx = np.random.rand(20, 3)
    xx2 = np.random.rand(10, 3)
    yy2 = np.zeros(10)
    gmms = {0: GaussianMixture().fit(xx[yy==0]),
            1: GaussianMixture().fit(xx[yy==1])}
    gmms2 = {0: GaussianMixture().fit(xx2)}

    yy_res, probs = func(GMMs = gmms, Xtest = xx)
    _, probs2 = func(GMMs = gmms2, Xtest = xx2)

    tests = {'debes retornar las probabilidades por cada clase': len(np.unique(probs.sum(axis =1))) == len(probs.sum(axis =1)),
             'la salida debe la etiqueta de las clases predichas':  (np.unique(yy_res) == np.unique(yy)).all(),
             'el shape de la matriz de probs es incorrecto': yy_res.shape[0] == xx.shape[0],
             'evita dejar codigo estatico':  probs.shape == (xx.shape[0], len(np.unique(yy))) and probs2.shape == (xx2.shape[0], len(np.unique(yy2)))
             }
    return (ut.test_conditions_and_methods(tests))

@unknow_error
def test_experimentar(func):
    yy = np.random.choice(2, 30)
    xx = np.vstack([np.random.rand(15, 3), 2*np.random.rand(15, 3)])
    mts = ['full', 'tied', 'diag', 'spherical']
    ms = [1,2,3]
    cols = ['matriz de covarianza','numero de componentes',
            'eficiencia de entrenamiento',
            'desviacion estandar entrenamiento',
            'eficiencia de prueba',
            'desviacion estandar prueba']

    errs = ['eficiencia de entrenamiento',
            'eficiencia de prueba']

    
    code_to_look = ['folds=3']
    res2 = ut.check_code(code_to_look, func)       

    res = ut.test_experimento_oneset(func,  shape_val=(len(mts)*len(ms), 6), 
                                    col_error = errs,
                                    col_val=cols,
                                    X = xx, Y=yy,
                                    covariance_types = mts,
                                    num_components = ms)
    return (res and res2)

@unknow_error
def test_experimentar_kmeans(func):
    code_to_look = [['KMeans', ".fit", ".predict", 'init="k-means++"'], 
                    ['KMeans', ".fit", ".predict", "init='k-means++'"]]
    res2 = ut.check_code(code_to_look, func)
    yy = np.random.choice(2, 30)
    xx = np.vstack([np.random.rand(15, 3), 2*np.random.rand(15, 3)])
    nc = [1,2,3]
    cols = ['numero de clusters',
            'eficiencia de entrenamiento',
            'desviacion estandar entrenamiento',
            'eficiencia de prueba',
            'desviacion estandar prueba']

    errs = ['eficiencia de entrenamiento',
            'eficiencia de prueba']

    res = ut.test_experimento_oneset(func, shape_val=(len(nc), 5), 
                                    col_error = errs,
                                    col_val=cols,
                                    X = xx, Y=yy,
                                    numero_clusters = nc)
    
    return (res and res2)

def part_1 ():
    #cargamos la bd iris desde el dataset de sklearn
    GRADER = Grader("lab3_part1")
    GRADER.add_test("ejercicio1", Tester(test_get_muestras_by_cv))
    GRADER.add_test("ejercicio2", Tester(test_GMMClassifierTrain))
    GRADER.add_test("ejercicio3", Tester(test_GMMClassfierVal))
    GRADER.add_test("ejercicio4", Tester(test_experimentar))
    GRADER.add_test("ejercicio5", Tester(test_experimentar_kmeans))
    return(GRADER)


cols_errs = ['eficiencia de entrenamiento','eficiencia de prueba']

def generate_data():
    yy = np.random.choice(2, 30)
    xx = np.vstack([np.random.rand(15, 3), 2*np.random.rand(15, 3)])
    return (xx, yy)

@unknow_error
def test_experimentar_dt(func):
    xx, yy = generate_data()
    depths = [3,5,10]
    cols = ['profunidad del arbol', 'eficiencia de entrenamiento',
            'desviacion estandar entrenamiento', 'eficiencia de prueba',
            'desviacion estandar prueba']
    res = ut.test_experimento_oneset(func,  shape_val=(len(depths), 5), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    X = xx, Y=yy,
                                    depths = depths,
                                    normalize = 'ninguna')

    res_df_2 = ut.test_experimento_oneset(func,  shape_val=(len(depths), 5), 
                                col_error = cols_errs,
                                col_val=cols,
                                X = xx, Y=yy,
                                depths = depths,
                                normalize = 'estandar')

    res_df_3 = ut.test_experimento_oneset(func,  shape_val=(len(depths), 5), 
                            col_error = cols_errs,
                            col_val=cols,
                            X = xx, Y=yy,
                            depths = depths,
                            normalize = 'min-max')

    shared = ['DecisionTreeClassifier', 
              'max_depth=',
              "MinMaxScaler()",
              "StandardScaler()",
              ".fit", 
              "X=Xtrain,y=Ytrain",
              ".predict(X=Xtrain)",
              ".predict(X=Xtest)",
              'criterion="entropy"']
    
    code_to_look = [ shared + ['criterion="entropy"'],
                    shared + ["criterion='entropy'"]
                    
                    ]
    res2 = ut.check_code(code_to_look, func)

    return (res and res2 and res_df_2 and res_df_3)

@unknow_error
def test_experimentar_rf(func):
    xx, yy = generate_data()
    trees = [3,5,10]
    num_vars = [1,2,3]
    cols = ['número de arboles', 'variables para la selección del mejor umbral',
       'eficiencia de entrenamiento', 'desviacion estandar entrenamiento',
       'eficiencia de prueba', 'desviacion estandar prueba']
   
    res = ut.test_experimento_oneset(func,  shape_val=(len(trees)*len(num_vars), 6), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    X = xx, Y=yy,
                                    num_trees = trees,
                                    numero_de_variables = num_vars)
    code_to_look = ['RandomForestClassifier', 
                    'n_estimators=', 
                    'max_features=',  
                    ".fit", 
                    ".predict", 
                    'min_samples_leaf=3',
                    "X=Xtrain,y=Ytrain",
                    ".predict(X=Xtrain)",
                    ".predict(X=Xtest)"]
    res2 = ut.check_code(code_to_look, func)
    return (res and res2)

@unknow_error
def test_experimentar_gbt(func):
    xx, yy = generate_data()
    trees = [3,5,10]
    cols = ['número de arboles', 'eficiencia de entrenamiento',
       'desviacion estandar entrenamiento', 'eficiencia de prueba',
       'desviacion estandar prueba']
    res = ut.test_experimento_oneset(func,  shape_val=(len(trees), len(cols)), 
                                    col_error = None,
                                    col_val=cols,
                                    X = xx, Y=yy,
                                    num_trees = trees)
    code_to_look = ['GradientBoostingClassifier', 
                    'n_estimators=',  
                    ".fit", 
                    ".predict",
                     "min_samples_split=3", 
                     "X=Xtrain,y=Ytrain",
                    ".predict(X=Xtrain)",
                    ".predict(X=Xtest)"]
    res2 = ut.check_code(code_to_look, func)
    return (res and res2)

@unknow_error
def test_time_rf_gbt_training(func):
    xx, yy = generate_data()
    trees = [3,5,10]
    num_vars = [1,2,3]
    cols = ['número de arboles', 'variables para la selección del mejor umbral',
       'tiempo de entrenamiento', 'metodo']
    res = ut.test_experimento_oneset(func,  shape_val=(len(trees)*len(num_vars), len(cols)), 
                                    col_error = ['tiempo de entrenamiento'],
                                    col_val=cols,
                                    X = xx, Y=yy,
                                    num_trees = trees,
                                    numero_de_variables = num_vars, 
                                    metodo = 'rf')
    code_to_look = ['RandomForestClassifier', 'n_estimators=', "max_features=", 
    "time.process_time()",  ".fit", "GradientBoostingClassifier", "X=Xtrain,y=Ytrain",]
    res2 = ut.check_code(code_to_look, func)
    return (res and res2)


def part_2 ():
    #cargamos la bd iris desde el dataset de sklearn
    GRADER = Grader("lab3_part2")
    GRADER.add_test("ejercicio1", Tester(test_experimentar_dt))
    GRADER.add_test("ejercicio2", Tester(test_experimentar_rf))
    GRADER.add_test("ejercicio3", Tester(test_experimentar_gbt))
    GRADER.add_test("ejercicio4", Tester(test_time_rf_gbt_training))
    return(GRADER)