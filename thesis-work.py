import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import randrange, uniform
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from gplearn.genetic import SymbolicRegressor
from sklearn.decomposition import PCA
import math

# Global Variables
FEATURES = 2
NUMBER_OF_GENERATION = 20
ROWS = 300
POPULATION_SIZE = 400
TEST_SIZE = .2
NUMBER_OF_REGIONS = 3
# Equation 1
# formula = lambda X: X[:, 0]**2 - X[:, 1]**2 + X[:, 1] - 1 
# Equation 2
formula = lambda X: 0.3 * X[:,0] * np.sin(2*math.pi*X[:,1])
NUMBER_OF_RUNS = 30

def train_pca_gp(seed_value):

    # Generating random data
    rng = check_random_state(seed_value)
    X = rng.uniform(-1, 1, ROWS).reshape(ROWS//FEATURES, FEATURES)
    Y = formula(X)

    # Dividing it into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = TEST_SIZE, random_state = 0)

    # Convert it to PCA
    pca = PCA(n_components = 1)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Training the system
    est_gp = SymbolicRegressor(population_size=POPULATION_SIZE,
                            generations=NUMBER_OF_GENERATION*3, stopping_criteria=0.01,
                            p_crossover=0.7, p_subtree_mutation=0.1,
                            p_hoist_mutation=0.05, p_point_mutation=0.1,
                            max_samples=0.9, verbose=1,
                            parsimony_coefficient=0.01, random_state=0)
    est_gp.fit(X_train_pca,y_train)

    # Use the converted pca test set
    x0 = X_test_pca[:,0]
    predicted_formula_result_y = est_gp.predict(np.c_[x0.ravel()]).reshape(x0.shape)
    fitness = mean_squared_error(y_test, predicted_formula_result_y)
    print('Final Fitness: ',str(fitness))

def train_dv_without_pca_gp(seed_value):
    # Generating random data
    rng = check_random_state(seed_value)
    X = rng.uniform(-1, 1, ROWS).reshape(ROWS//FEATURES, FEATURES)
    Y = formula(X)

    # Dividing it into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = TEST_SIZE, random_state = 0)

    # Convert it to PCA
    pca = PCA(n_components = 1)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Applying DV

    # sort X_train_pca and y_train by index
    sorted_indexes = np.argsort(X_train_pca,axis=0)
    sorted_x_train_pca = X_train_pca[sorted_indexes]
    sorted_y_train = y_train[sorted_indexes]

    # Finding Change of Slope
    slope1 = []
    slope2 = []
    for itr in range(1,len(sorted_x_train_pca)):
        slope1.append((sorted_y_train[itr]-sorted_y_train[itr-1])/(sorted_x_train_pca[itr]-sorted_x_train_pca[itr-1]))
    for itr in range(1,len(slope1)):
        slope2.append((slope1[itr]-slope1[itr-1])/(sorted_x_train_pca[itr]-sorted_x_train_pca[itr-1]))

    # normalize slope2 
    normalized_slope2 = (slope2-min(slope2))/(max(slope2)-min(slope2))

    # Calculating Quantiles
    normalized_slope2 = np.reshape(normalized_slope2,len(normalized_slope2))
    quantile_ranges = pd.qcut(normalized_slope2,NUMBER_OF_REGIONS,labels=False,retbins=True)
    quantile_ranges = quantile_ranges[1]

    # Adding the difficult Vectors
    difficult_points = {}
    for q_ind in range(NUMBER_OF_REGIONS):
        low = quantile_ranges[q_ind]
        high = quantile_ranges[q_ind+1]
        difficult_points[q_ind] = []
        for n_ind in range(0,len(normalized_slope2)):
            if normalized_slope2[n_ind] >= low and normalized_slope2[n_ind] <= high:
                difficult_points[q_ind].append(n_ind)

    # Hard to evolve points
    hard_to_evolve_x = X_train_pca[difficult_points[0]]
    hard_to_evolve_y = y_train[difficult_points[0]]

    # Medium to evolve points
    medium_to_evolve_x = X_train_pca[difficult_points[1]]
    medium_to_evolve_y = y_train[difficult_points[1]]

    # Easy to evolve points
    easy_to_evolve_x = X_train_pca[difficult_points[2]]
    easy_to_evolve_y = y_train[difficult_points[2]]

    # Training the system
    est_gp = SymbolicRegressor(population_size=POPULATION_SIZE,
                               generations=NUMBER_OF_GENERATION, stopping_criteria=0.01,
                               p_crossover=0.7, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.9, verbose=1,
                               parsimony_coefficient=0.01, 
                               random_state=0, 
                               init_depth=(2,6)
                              )
    est_gp.fit(hard_to_evolve_x,hard_to_evolve_y)
    #print(est_gp._program)
    est_gp.set_params(generations=NUMBER_OF_GENERATION*2, warm_start=True)
    est_gp.fit(medium_to_evolve_x,medium_to_evolve_y)
    #print(est_gp._program)
    est_gp.set_params(generations=NUMBER_OF_GENERATION*3, warm_start=True)
    est_gp.fit(easy_to_evolve_x,easy_to_evolve_y)
    #print(est_gp._program)
    # list_of_est_program.append(str(est_gp._program))

    # Use the converted pca test set
    x0 = X_test_pca[:,0]
    predicted_formula_result_y = est_gp.predict(np.c_[x0.ravel()]).reshape(x0.shape)
    fitness = mean_squared_error(y_test, predicted_formula_result_y)
    print('Final Fitness: ',str(fitness))

def train_std_gp(seed_value):

    # Generating random data
    rng = check_random_state(seed_value)
    X = rng.uniform(-1, 1, ROWS).reshape(ROWS//FEATURES, FEATURES)
    Y = formula(X)

    # Dividing it into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = TEST_SIZE, random_state = 0)

    # Training the system
    est_gp = SymbolicRegressor(population_size=POPULATION_SIZE,
                            generations=NUMBER_OF_GENERATION*3, stopping_criteria=0.01,
                            p_crossover=0.7, p_subtree_mutation=0.1,
                            p_hoist_mutation=0.05, p_point_mutation=0.1,
                            max_samples=0.9, verbose=1,
                            parsimony_coefficient=0.01, random_state=0)
    est_gp.fit(X_train,y_train)
    x0 = X_test[:,0]
    x1 = X_test[:,1]
    predicted_formula_result_y = est_gp.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)
    fitness = mean_squared_error(y_test, predicted_formula_result_y)
    print('Final Fitness: ',str(fitness))

def train_dv_gp(seed_value):

    # Generating random data
    rng = check_random_state(seed_value)
    X = rng.uniform(-1, 1, ROWS).reshape(ROWS//FEATURES, FEATURES)
    Y = formula(X)

    # Dividing it into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = TEST_SIZE, random_state = 0)

    # Convert it to PCA
    pca = PCA(n_components = 1)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Applying DV

    # sort X_train_pca and y_train by index
    sorted_indexes = np.argsort(X_train_pca,axis=0)
    sorted_x_train_pca = X_train_pca[sorted_indexes]
    sorted_y_train = y_train[sorted_indexes]

    # Finding Change of Slope
    slope1 = []
    slope2 = []
    for itr in range(1,len(sorted_x_train_pca)):
        slope1.append((sorted_y_train[itr]-sorted_y_train[itr-1])/(sorted_x_train_pca[itr]-sorted_x_train_pca[itr-1]))
    for itr in range(1,len(slope1)):
        slope2.append((slope1[itr]-slope1[itr-1])/(sorted_x_train_pca[itr]-sorted_x_train_pca[itr-1]))

    # normalize slope2 
    normalized_slope2 = (slope2-min(slope2))/(max(slope2)-min(slope2))

    # Calculating Quantiles
    normalized_slope2 = np.reshape(normalized_slope2,len(normalized_slope2))
    quantile_ranges = pd.qcut(normalized_slope2,NUMBER_OF_REGIONS,labels=False,retbins=True)
    quantile_ranges = quantile_ranges[1]

    # Adding the difficult Vectors
    difficult_points = {}
    for q_ind in range(NUMBER_OF_REGIONS):
        low = quantile_ranges[q_ind]
        high = quantile_ranges[q_ind+1]
        difficult_points[q_ind] = []
        for n_ind in range(0,len(normalized_slope2)):
            if normalized_slope2[n_ind] >= low and normalized_slope2[n_ind] <= high:
                difficult_points[q_ind].append(n_ind)

    # Hard to evolve points
    hard_to_evolve_x = X_train_pca[difficult_points[0]]
    hard_to_evolve_y = y_train[difficult_points[0]]

    # Medium to evolve points
    medium_to_evolve_x = X_train_pca[difficult_points[1]]
    medium_to_evolve_y = y_train[difficult_points[1]]

    # Easy to evolve points
    easy_to_evolve_x = X_train_pca[difficult_points[2]]
    easy_to_evolve_y = y_train[difficult_points[2]]

    # Training the system
    est_gp = SymbolicRegressor(population_size=POPULATION_SIZE,
                               generations=NUMBER_OF_GENERATION, stopping_criteria=0.01,
                               p_crossover=0.7, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.9, verbose=1,
                               parsimony_coefficient=0.01, 
                               random_state=0, 
                               init_depth=(2,6)
                              )
    est_gp.fit(hard_to_evolve_x,hard_to_evolve_y)
    #print(est_gp._program)
    est_gp.set_params(generations=NUMBER_OF_GENERATION*2, warm_start=True)
    est_gp.fit(medium_to_evolve_x,medium_to_evolve_y)
    #print(est_gp._program)
    est_gp.set_params(generations=NUMBER_OF_GENERATION*3, warm_start=True)
    est_gp.fit(easy_to_evolve_x,easy_to_evolve_y)
    #print(est_gp._program)
    # list_of_est_program.append(str(est_gp._program))

    # Use the converted pca test set
    x0 = X_test_pca[:,0]
    predicted_formula_result_y = est_gp.predict(np.c_[x0.ravel()]).reshape(x0.shape)
    fitness = mean_squared_error(y_test, predicted_formula_result_y)
    print('Final Fitness: ',str(fitness))

def run_std_gp():
    for itr in range(0,NUMBER_OF_RUNS):
        train_std_gp(itr)

def run_pca_gp():
    for itr in range(0,NUMBER_OF_RUNS):
        train_pca_gp(itr)

def run_dv_gp():
    for itr in range(0,NUMBER_OF_RUNS):
        train_dv_gp(itr)

def draw_box_plots():
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # fake up some data
    spread = np.random.rand(50) * 100
    center = np.ones(25) * 50
    flier_high = np.random.rand(10) * 100 + 100
    flier_low = np.random.rand(10) * -100
    data = np.concatenate((spread, center, flier_high, flier_low))
    fig1, ax1 = plt.subplots()
    ax1.set_title('Basic Plot')
    ax1.boxplot(data)

def transform_result(result):
    generations = {}
    for run in result:
        for gen in range(0,len(result[run])):
            object_of_gen_in_run = result[run][gen]
            value_of_fit = [*object_of_gen_in_run.values()][0]
            if (gen in generations.keys()):
                generations[gen].append(value_of_fit)
            else:
                generations[gen] = []
                generations[gen].append(value_of_fit)
    return generations

def transform_to_array(data):
    final_result = []
    for item in data:
        final_result.append(data[item])
    print(final_result)

def get_fitness_each_gen():
    run_results = {}
    generation_results = []
    f = open("DV_GP_output_F2.txt","r")
    fl =f.readlines()
    read_fitness_in_next_line = False
    run_number = 0
    for x in fl:
        splits = x.split()
        splits_in_line = len(splits)
        if splits_in_line == 7 and read_fitness_in_next_line == True:
            single_generation = {}
            single_generation[str(splits[0])] = float(splits[4])
            generation_results.append(single_generation)
            run_results[str(run_number)] = generation_results
            continue
        if splits_in_line == 9:
            generation_results = []
            run_number += 1
            read_fitness_in_next_line = True
            continue
        else:
            read_fitness_in_next_line = False
    return run_results

def get_fitness_each_gen_dv():
    run_results = {}
    generation_results = []
    f = open("test.txt","r")
    fl =f.readlines()
    read_fitness_in_next_line = False
    run_number = 0
    for x in fl:
        splits = x.split()
        splits_in_line = len(splits)
        print(splits_in_line,splits)
        if splits_in_line == 7 and read_fitness_in_next_line == True:
            # single_generation = {}
            # single_generation[str(splits[0])] = float(splits[4])
            # generation_results.append(single_generation)
            # run_results[str(run_number)] = generation_results
            continue
        if splits_in_line == 9:
            generation_results = []
            run_number += 1
            read_fitness_in_next_line = True
            continue
        else:
            read_fitness_in_next_line = False
    return
    # return run_results
def main():
    # run_dv_gp()
    result = get_fitness_each_gen()
    transformed_result = transform_result(result)
    transform_to_array(transformed_result)
    # draw_box_plots(transformed_result)
main()
    
