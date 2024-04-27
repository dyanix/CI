import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from geneticalgorithm import geneticalgorithm as ga


np.random.seed(42)
x = np.random.uniform(low = 0.1,high = 1.0,size=(1000,5))
y = np.random.uniform(low = 0.1,high = 0.9,size = 1000)

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)


def create_model(hidden_layer_sizes,activation,solver,alpha):
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,activation = activation,solver=solver,alpha=alpha,max_iter=1000,random_state=42)

    return model


def fitness_func(params):
    hidden_layer_sizes = int(params[0])
    activation = 'relu' if params[1] < 0.5 else 'logistic'
    solver = 'sgd' if params[2] < 0.5 else 'adam'
    aplpha = params[3]


    model = create_model(hidden_layer_sizes,activation,solver,aplpha)
    model.fit(X_train,y_train)
    score = model.score(X_test,y_test)
    return score



param_ranges = np.array([

    [1,100],[0,1],[0,1],[0.0001,1]


])

ga_instance = ga(
    function=fitness_func,
    dimension=4,
    variable_type='real',
    variable_boundaries=param_ranges
)

ga_instance.run()

best_params = ga_instance.output_dict['variable']
print("Best Parameter",best_params)

best_hidden_layer_sizes = int(best_params[0])
best_activation = 'relu' if best_params[1] < 0.5 else 'logistic'
best_solver = 'sgd' if best_params[2] < 0.5 else 'adam'

best_alpha = best_params[3]



final_model = create_model(best_hidden_layer_sizes,best_activation,best_solver,best_alpha)
final_model.fit(X_train,y_train)


train_score = final_model.score(X_train,y_train)
test_score = final_model.score(X_test,y_test)

print("Training set score:",train_score)

print("Test Set Score:",test_score)



