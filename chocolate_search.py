
import chocolate as choco
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.multioutput import MultiOutputRegressor

from sklearn.svm import SVR


from xgboost import XGBRegressor

from sklearn.neighbors import KNeighborsRegressor

def run_chocolate(X_train, X_test, Y_train, Y_test, y_scaler_nl,run,space,iterations,clear_db):

    # CREATE and Connect to sqlite database in current directory
    conn = choco.SQLiteConnection(url="sqlite:///"+run+".db")
    
    #repeat each model run three times and take average
    #cv = choco.Repeat(repetitions=3, reduce=np.mean, rep_col="_repetition_id")
    
    #search strategy - Bayes attempts to "learn" patterns from ALL previous runs
    sampler = choco.Bayes(conn, space, clear_db=False)#, crossvalidation=cv)
    #sampler = choco.Grid(conn, space, clear_db=clear_db)
    #lets run 10 times and see what if we get a better answer
    for i in range(0,iterations):
        #examine db and pick next experiment
        token, params = sampler.next()
        #run experiment
        loss = _score(X_train, X_test, Y_train, Y_test, y_scaler_nl, params)
        #print("finished iteration",str(i),"loss",str(loss))
        #add new result to database
        sampler.update(token, loss)
        
#build a function to create a model with arbitrary params, train, fit, test and return a score

def _score(X_train, X_test, Y_train, Y_test, y_scaler_nl, params):
    
    #create ML model
    model = build_Regressor(params)
    
    #train
    model.fit(X_train, Y_train)
    
    #predict
    Y_test_pred_scaled = model.predict(X_test)
    
    #de-scale
    Y_test_pred_unscaled = y_scaler_nl.inverse_transform(Y_test_pred_scaled)

    # Chocolate minimizes the loss
    return mean_squared_error(Y_test_pred_unscaled, Y_test)

def build_Regressor(params):   
    if params['algo'] == 'xgb': 
        return MultiOutputRegressor(XGBRegressor(n_estimators=int(params['n_estimators']),
                                           learning_rate=params['learning_rate'],
                                            gamma=0, 
                                            subsample=0.75,
                                            colsample_bytree=1,
                                            max_depth=int(params['max_depth'])
                                            ))
    if params['algo'] == 'svrrbf': 
        return MultiOutputRegressor(SVR(kernel ="rbf", C = params['C'],gamma=params['gamma']))
    if params['algo'] == 'svrpoly': 
        return MultiOutputRegressor(SVR(kernel ="poly", C = params['C'],gamma=params['gamma'],
                                       degree=int(params['degree']),
                                    coef0 = params['coef0']))
    if params['algo'] == 'nneighbors':
        return KNeighborsRegressor(n_neighbors=int(params['n_neighbors']),weights=params['weights'],algorithm=params['algorithm'])
