from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor,plot_tree
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

class MyModels:
    
    def __init__(self, model_name, params=None, search_type='grid'):
        self.model_name = model_name.lower()
        self.model = None
        self.best_params_ = None
        self.rmse_train_ = None
        self.rmse_test_ = None
        self.search_type = search_type
        self.model_summary = None
        
        if params is None:
            if self.model_name == 'random forest':
                self.model = RandomForestRegressor()
                self.params_ = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.model_name == 'xgboost':
                self.model = xgb.XGBRegressor()
                self.params_ = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20],
                    'learning_rate': [0.01, 0.1, 1.0]
                }
            elif self.model_name == 'adaboost':
                self.model = AdaBoostRegressor()
                self.params_ = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                }
            elif self.model_name == 'decision tree':
                self.model = DecisionTreeRegressor()
                self.params_ = {
                    'max_depth': [5, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            else:
                print('Invalid model name. Please choose from "random forest", "xgboost", "adaboost", or "decision tree".')
        else:
            self.params_ = params
            #print("Parameters for the model are :",self.params_)
            if self.model_name == 'random forest':
                self.model = RandomForestRegressor()
                
            elif self.model_name == 'xgboost':
                self.model = xgb.XGBRegressor()
                
            elif self.model_name == 'adaboost':
                self.model = AdaBoostRegressor()
                
            elif self.model_name == 'decision tree':
                self.model = DecisionTreeRegressor()
                
            else:
                print('Invalid model name. Please choose from "random forest", "xgboost", "adaboost", or "decision tree".')
        
        if self.search_type == 'grid':
            self.search_method_ = GridSearchCV
        elif self.search_type == 'random':
            self.search_method_ = RandomizedSearchCV
        else:
            print('Invalid search type. Please choose "grid" or "random".')
    
    def train(self, X_train, y_train):
        if self.model is None:
            return
        
        print(self.search_type,"search")
        print("Parameters for the model are :",self.params_)
        
        if self.search_type == 'grid':
            grid_search = self.search_method_(
                estimator=self.model, 
                param_grid=self.params_, 
                cv=5, 
                n_jobs=-1, 
                verbose=2
            )
        else:
            grid_search = self.search_method_(
                estimator=self.model, 
                param_distributions=self.params_, 
                cv=5, 
                n_jobs=-1, 
                verbose=2
            )
            
        grid_search.fit(X_train, y_train)
        
        self.model_summary = grid_search
        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.rmse_train_ = mean_squared_error(y_train, self.model.predict(X_train), squared=False)
        
    def predict(self, X_test):
        if self.model is None:
            return
        
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        if self.model is None:
            return
        
        y_pred = self.model.predict(X_test)
        self.rmse_test_ = mean_squared_error(y_test, y_pred, squared=False)
        return self.rmse_test_
    
    def summary_df(self):
        if self.model is None:
            return None
        
        return pd.DataFrame({
            'Model Name': [self.model_name],
            'Search Type':[self.search_type],
            'Best Parameters': [self.best_params_],
            'Validation Performance': [self.rmse_train_],
            'Test Performance': [self.rmse_test_],
            'Optimal Parameters': [self.model.get_params()]
            })
        

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the Boston housing dataset
X, y = load_boston(return_X_y=True)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1_random_forest = MyModels('random forest')
model1_random_forest.train(X_train, y_train)
model1_random_forest.predict(X_test)
model1_random_forest.evaluate(X_test, y_test)
Model1 = model1_random_forest.summary_df()

model2_xgboost = MyModels('xgboost',search_type='random')
model2_xgboost.train(X_train, y_train)
model2_xgboost.predict(X_test)
model2_xgboost.evaluate(X_test, y_test)
Model2 = model2_xgboost.summary_df()

model3_ada = MyModels('adaboost')
model3_ada.train(X_train, y_train)
model3_ada.predict(X_test)
model3_ada.evaluate(X_test, y_test)
Model3 = model3_ada.summary_df()

model4_dt = MyModels('decision tree')
model4_dt.train(X_train, y_train)
model4_dt.predict(X_test)
model4_dt.evaluate(X_test, y_test)
Model4 = model4_dt.summary_df()

model5_xgboost = MyModels('xgboost')
model5_xgboost.train(X_train, y_train)
model5_xgboost.predict(X_test)
model5_xgboost.evaluate(X_test, y_test)
Model5 = model5_xgboost.summary_df()


Final_Compare = pd.concat([Model1,Model2,Model3,Model4,Model5])



Final_Compare.to_csv('C:\\Users\\Divyansh\\OneDrive\\Desktop\\compare.csv')

new_param = {
    'n_estimators': [50, 100, 200,300],
    'max_depth': [5, 10, 20,30],
    'learning_rate': [0.4, 0.3, 1.0],
    #'min_child_weight':[1,2,3],
    #'gamma':[0,0.1,0.2,0.3,0.4],
    #'alhpa':[0,0.1,0.2,0.3,0.4]
}

model3_xgboost = MyModels('xgboost',params=new_param)
model3_xgboost.train(X_train, y_train)
model3_xgboost.predict(X_test)
model3_xgboost.evaluate(X_test, y_test)
aaa2 = model3_xgboost.summary_df()

#####################################################################################################
####################################################################################################
########################### Inheritance #############################################################

class DT_CLass(MyModels):
    def __init__(self,model_name,Graph, params=None, search_type='grid'):
        super().__init__(model_name, params, search_type)
        self.Graph = Graph
        
        
    def model_fitting(self,X_train, y_train):
        super().train(X_train, y_train)
        
    def visualize_graph(self,X_train, y_train):
        clf = DecisionTreeRegressor()
        clf.fit(X_train, y_train)
        plot_tree(clf, filled=True)
    
    
        
   
vvv = DT_CLass(model_name = "decision tree",Graph= "Yes")

vvv.model_fitting(X_train, y_train)

vvv.visualize_graph(X_train, y_train)
vvv.model_summary

######################################################################################################
######################################################################################################
######################################################################################################

### All the class variables are public
class Car():
    def __init__(self,windows,doors,enginetype):
        self.windows=windows
        self.doors=doors
        self.enginetype=enginetype
        
### All the class variables are protected
class Car():
    def __init__(self,windows,doors,enginetype):
        self._windows=windows
        self._doors=doors
        self._enginetype=enginetype

### All the class variables are private
class Car():
    def __init__(self,windows,doors,enginetype):
        self.__windows=windows
        self.__doors=doors
        self.__enginetype=enginetype