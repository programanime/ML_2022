"""
Este archivo es generado automaticamente.

###### NO MODIFICAR #########

# cualquier alteración del archivo
# puede generar una mala calificación o configuracion
# que puede repercutir negativamente en la 
# calificación del laboratorio!!!!!

###### NO MODIFICAR #########
"""

from imports import *
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR, SVC
import warnings
import os
import itertools
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')


def generate_data(is_class = False, cols = 2):
    yy = np.random.choice(2, 30) if is_class else 2*np.random.rand(60).reshape((int(60/cols),cols))
    xx = np.vstack([np.random.rand(15, 3), 2*np.random.rand(15, 3)])
    return (xx, yy)

@unknow_error
def test_create_rnn_model (func):
    code_to_look = [['look_back,1', "name='rnn_layer'", '.add(rnn_layer)',
                     "loss='mean_absolute_error'"], 
                   ['look_back,1', 'name="rnn_layer"', '.add(rnn_layer)',
                     'loss="mean_absolute_error"']]
    res2 = ut.check_code(code_to_look, func, msg = "**** recordar usar las funciones sugeridas ***", debug = False)
    return (res2)



@unknow_error
def test_many_to_one_custom(func):
    print("este es el test de dorian y dani")
    data1 = pd.DataFrame(data = {'T (degC)': {0: 112, 1: 118, 2: 132, 3: 129, 4: 121, 5: 135}})
    data2 = pd.DataFrame(data = {'T (degC)': {0: 100, 1: 101, 2: 102, 3: 103, 4: 104, 5: 105, 6:106}})
    w1 = func (data1, 2)
    w2 = func (data2, 4)

    for entradas, salidas in w1.train_df.take(1):
        t1 =  entradas.shape == (2, 2, 1)
        t2 =  salidas.shape == (2, 1, 1)
    
    for entradas, salidas in w2.train_df.take(1):
        t3 =  entradas.shape == (1, 4, 1)
        t4 =  salidas.shape == (1, 1, 1)
    
    tests = {'No se esta construyendo adecuadamente los valores': np.all([t1,t2,t3,t4]),
             'Se debe pasar los dataframe de entrenamiento': w1.train_df.shape == (4,1),
             'Se debe pasar los dataframe de prueba': w2.test_df.shape == (2,1)
             }
    
    return (ut.test_conditions_and_methods(tests))
    
    

@unknow_error
def test_experimentar_rnn(func):
    xx, _ = generate_data(False)
    xx = pd.DataFrame(data = {'T (degC)': xx[:, 0]}, index = range(len(xx[:, 0])))
    looksbacks = [1,2]
    neu = [1,2]
    cols =['lags', 'neuronas por capa', 
        'Métrica rendimiento en entrenamiento',
        'Métrica de rendimiento prueba']

    cols_errs = ['Métrica rendimiento en entrenamiento', 'Métrica de rendimiento prueba']

    code_to_look = ['MAX_EPOCHS=25',
                    'fit(x=window.train',
                    '.predict(x=window.train', 
                    '.predict(x=window.test', 
                    'create_rnn_model',
                    'many_to_one_custom',
                    "mean_absolute_error", 
                    "y_pred=testYPred",
                    'y_pred=trainYPred'] 

    res2 = ut.check_code(code_to_look, func, msg = "**** recordar usar las funciones sugeridas ***", debug = False)

    res = ut.test_experimento_oneset(func,  shape_val=(len(looksbacks)*len(neu), len(cols)), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    data = xx,
                                    look_backs = looksbacks,
                                    hidden_neurons= neu)
    return (res and res2)


@unknow_error
def test_experimentar_LSTM(func):
    xx, _ = generate_data(False)
    xx = pd.DataFrame(data = {'T (degC)': xx[:, 0]}, index = range(len(xx[:, 0])))
    looksbacks = [1,2]
    neu = [1,2]
    cols =['lags', 'neuronas por capa', 
          'Métrica rendimiento en entrenamiento',
          'Métrica de rendimiento prueba']

    cols_errs = ['Métrica rendimiento en entrenamiento', 'Métrica de rendimiento prueba']
    code_to_look = ['MAX_EPOCHS=50',
                    'fit(x=window.train',
                    '.predict(x=window.train', 
                    '.predict(x=window.test', 
                    'create_lstm_model',
                    'many_to_one_custom',
                    "mean_absolute_error", 
                    "y_pred=testYPred"]

    res2 = ut.check_code(code_to_look, func, msg = "**** recordar usar las funciones sugeridas ***", debug = False)

    res = ut.test_experimento_oneset(func,  shape_val=(len(looksbacks)*len(neu), len(cols)), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    data = xx,
                                    look_backs = looksbacks,
                                    hidden_neurons= neu)
    return (res and res2)


def part_1 ():
    GRADER = Grader("lab5_part1", num_questions = 4)
    os.system("pip install statsmodels==0.12")
    GRADER.add_test("ejercicio1", Tester(test_many_to_one_custom))
    GRADER.add_test("ejercicio2", Tester(test_create_rnn_model))
    GRADER.add_test("ejercicio3", Tester(test_experimentar_rnn))
    GRADER.add_test("ejercicio4", Tester(test_experimentar_LSTM))

    return(GRADER)


def predict_svr(x_train, y_train, x_test,y_test, kernel, gamma, param_reg):
    params = {'kernel': kernel, 'gamma': gamma, 'C': param_reg}
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print("*** calculando predicciones ***")
    md = SVR(**params).fit(X_train,y_train)
    ypred= md.predict(x_test)
    error = mean_absolute_percentage_error(y_true = y_test, y_pred = ypred)
    print("metrica", error)
    return(ypred)


@unknow_error
def test_clean_data(func):
    db = np.loadtxt('AirQuality.data',delimiter='\t')  # Assuming tab-delimiter
    db = db.reshape(9357,13)
    to_remove = -200*np.ones(( 1, db.shape[1]))
    to_impute = np.hstack([to_remove[:,0:12], np.array([[10]])])
    db = np.vstack((db[0:3,:], to_remove, to_impute))
    db_df = pd.DataFrame(db, columns = [f'col_{c}' for c in range (1,14)])
    xx, yy = func(db_df)
    tests = {'No se están removiendo valores faltantes en variable de respuesta': yy.shape[0] == db.shape[0] - 1,
             'No se estan imputando los valores': ut.are_np_equal(np.round(np.median(xx, axis = 0)), np.round(xx[-1])),
             'NO se estan removiendo todos los valores faltantes': not((xx==-200).any()),
             'Cuidado estas retornando diferentes shapes de X. Leer las instrucciones.': xx.shape[1] == 12
             }

    test_res = ut.test_conditions_and_methods(tests)

    return (test_res)


@unknow_error
def experiementarSVR(func):
    xx, yy = generate_data(False, cols=1)
    ks = ['linear','rbf']
    gs = [1.0, 0.1]
    cs = [100]
    cols= ['kernel', 
        'gamma', 
        'param_reg', 
        'Métrica de de rendimiento',
       'Desviación estandar en métrica de de rendimiento', 
       '# de vectores de soporte', 
       '% de vectores de soporte']

    cols_errs =['Métrica de de rendimiento', 'Desviación estandar en métrica de de rendimiento']

    res, df_r = ut.test_experimento_oneset(func,  
                                    shape_val=(len(ks)*len(gs)*len(cs), len(cols)), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    x = xx,
                                    y=yy,
                                    kernels = ks,
                                    gammas= gs,
                                    params_reg = cs,
                                    return_df = True)

    code_to_look = ['KFold', 'kernel=kernel', 'gamma=gamma', 'C=param_reg', 'SVR',
                    'StandardScaler()', '.fit(X=X_train', 
                    '.predict(X=X_test', "n_splits=5",
                    '.support_', 'mean_absolute_percentage_error'] 
    res2 = ut.check_code(code_to_look, func, debug = False)

    cond =( (df_r['% de vectores de soporte'].max() > 100.0) or 
            (df_r['% de vectores de soporte'].min() < 0.0) or
            df_r['% de vectores de soporte'].max() <= 1.01
            )

    if (cond ):
        print("*** recordar retornar el porcentaje de vectores de soporte ***")
        return (False)

    if ( (df_r[cols_errs[0]] == df_r[cols_errs[1]]).all()):
        print("*** recordar retornar el intervalo de confianza ***")
        return (False)

    return (res and res2)


@unknow_error
def experiementarSVC(func):
    xx, yy = generate_data(True)
    ks = ['linear','rbf']
    gs = [1.0, 0.1]
    cs = [100]
    cols= ['kernel', 
           'gamma', 
           'param_reg', 
           'estrategia', 
          'Métrica de de rendimiento entrenamiento',
          'Métrica de de rendimiento prueba',
          '% de vectores de soporte']
    
    cols_errs =['Métrica de de rendimiento entrenamiento', 'Métrica de de rendimiento prueba',]

    res, df_r = ut.test_experimento_oneset(func,  
                                    shape_val=(len(ks)*len(gs)*len(cs), len(cols)), 
                                    col_error = cols_errs,
                                    col_val=cols,
                                    x = xx,
                                    y=yy,
                                    kernels = ks,
                                    gammas= gs,
                                    params_reg = cs,
                                    return_df = True)
    
    code_to_look = ['StratifiedKFold', 'kernel=kernel', 'gamma=gamma', 'C=param_reg', 'SVC',
                    'StandardScaler()', '.fit(X=X_train', 
                    '.predict(X=X_test',  '.predict(X=X_train',
                    '.support_', 'accuracy_score', 'OneVsRestClassifier', 'estimators_'] 
    res2 = ut.check_code(code_to_look, func, debug = False)

    cond =( (df_r['% de vectores de soporte'].max() > 100.0) or 
            (df_r['% de vectores de soporte'].min() < 0.0) or
            df_r['% de vectores de soporte'].max() <= 1.01
            )

    if (cond):
        print("*** recordar retornar el porcentaje de vectores de soporte ***")
        return (False)


    return (res and res2)



def part_2 ():
    GRADER = Grader("lab5_part2", num_questions = 4)
    db = np.loadtxt('AirQuality.data',delimiter='\t')  # Assuming tab-delimiter
    db = db.reshape(9357,13)
    db = db[0:2000, :]
    print("Dim de la base de datos original: " + str(np.shape(db)))
    GRADER.add_test("ejercicio1", Tester(test_clean_data))
    GRADER.add_test("ejercicio2", Tester(experiementarSVR))
    GRADER.add_test("ejercicio3", Tester(experiementarSVC))

    return(GRADER, db)