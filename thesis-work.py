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
NUMBER_OF_GENERATION = 60
POPULATION_SIZE = 400
TEST_SIZE = .2
NUMBER_OF_REGIONS = 3
NUMBER_OF_RUNS = 30


def draw_box_plots(data,title):
    fig7, ax7 = plt.subplots(figsize=(18, 6))
    ax7.set_title(title)
    plt.ylabel('Fitness')
    plt.xlabel('Generations')
    ax7.boxplot(data)
    img_title = title+'.png'
    txt_title = title+'.txt'
    f = open(txt_title,'w+')
    f.write(str(data))
    plt.savefig(img_title)

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
    return final_result

def is_stagnation(gen_results,programs):
    # Check Stagnation
    # Take the average of last 5 generation fitness if its less the current stagnation is declared
    ttl = 0
    prev_avg = 0
    i = len(gen_results) - 2
    while i > len(gen_results) - 7:
        ttl += gen_results[i][i]
        i -= 1
    prev_avg = ttl/5
    curr_gen = gen_results[len(programs)-1][len(programs)-1]
    # returns true if stagnation else false
    return curr_gen > prev_avg

def rerun_model(est_gp,data_x,data_y,generation_results,first_time = False):
    # Warm start with the next generation
    est_gp.set_params(generations=len(est_gp._programs)+1, warm_start=True)
    est_gp.fit(data_x,data_y)

    # if first time than add all previous generation
    if first_time:
        start = 0
        end = len(est_gp._programs)
    else:
        start = len(est_gp._programs)-1
        end = len(est_gp._programs)
    for idGen in range(start,end):
        single_generation = {}
        single_generation[idGen] = math.inf
        for idPopulation in range(est_gp.population_size):
            if(est_gp._programs[idGen][idPopulation] != None):
                if est_gp._programs[idGen][idPopulation].raw_fitness_ < single_generation[idGen]:
                    single_generation[idGen] = est_gp._programs[idGen][idPopulation].raw_fitness_
        generation_results.append(single_generation)

def dv_gp(rows,features,function):
    run_results = {}
    for run_number in range(0,NUMBER_OF_RUNS):
        # Generating random data
        rng = check_random_state(run_number)
        X = rng.uniform(-1, 1, rows).reshape(rows//features, features)
        Y = function(X)

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
        hard_to_evolve_x = X_train[difficult_points[0]]
        hard_to_evolve_y = y_train[difficult_points[0]]

        # Medium to evolve points
        medium_to_evolve_x = X_train[difficult_points[1]]
        medium_to_evolve_y = y_train[difficult_points[1]]

        # Easy to evolve points
        easy_to_evolve_x = X_train[difficult_points[2]]
        easy_to_evolve_y = y_train[difficult_points[2]]

        # Training the system
        # Run the first 6 generations without checking stagation
        est_gp = SymbolicRegressor(population_size=POPULATION_SIZE,
                                generations=6, stopping_criteria=0.01,
                                p_crossover=0.7, p_subtree_mutation=0.1,
                                p_hoist_mutation=0.05, p_point_mutation=0.1,
                                max_samples=0.9, verbose=1,
                                parsimony_coefficient=0.01, 
                                random_state=0, 
                                init_depth=(2,6),
                                n_jobs=-1
                                )
        est_gp.fit(hard_to_evolve_x,hard_to_evolve_y)
        generation_results = []
        # The current data is hard to evolve points
        curr_data_x = hard_to_evolve_x
        curr_data_y = hard_to_evolve_y
        rerun_model(est_gp,curr_data_x,curr_data_y,generation_results,first_time=True)
        
        # Total regions and current region
        regions = 3
        cur_region = 0
        # Run the while loop till number of generations or till all the regions stagnate
        while len(generation_results) < NUMBER_OF_GENERATION:
            if cur_region < regions:
                # Check Stagnation
                # Take the average of last 5 generation fitness if its less the current stagnation is declared
                if is_stagnation(generation_results,est_gp._programs):
                    cur_region += 1
                    if cur_region == 1:

                        # Feed to model medium to evolve points
                        curr_data_x = medium_to_evolve_x
                        curr_data_y = medium_to_evolve_y
                        rerun_model(est_gp,curr_data_x,curr_data_y,generation_results)
                    elif cur_region == 2:

                        # Feed to model Easy to evolve points
                        curr_data_x = easy_to_evolve_x
                        curr_data_y = easy_to_evolve_y
                        rerun_model(est_gp,curr_data_x,curr_data_y,generation_results)
                else:
                    rerun_model(est_gp,curr_data_x,curr_data_y,generation_results)
            else:
                break
        run_results[run_number] = generation_results
    return run_results

def pca_gp(rows,features,function):
    run_results = {}
    for run_number in range(0,NUMBER_OF_RUNS):
        # Generating random data
        rng = check_random_state(run_number)
        X = rng.uniform(-1, 1, rows).reshape(rows//features, features)
        Y = function(X)

        # Dividing it into training and test set
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = TEST_SIZE, random_state = 0)

        # Convert it to PCA
        pca = PCA(n_components = 1)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Training the system
        est_gp = SymbolicRegressor(population_size=POPULATION_SIZE,
                                generations=NUMBER_OF_GENERATION, stopping_criteria=0.01,
                                p_crossover=0.7, p_subtree_mutation=0.1,
                                p_hoist_mutation=0.05, p_point_mutation=0.1,
                                max_samples=0.9, verbose=1,
                                parsimony_coefficient=0.01, random_state=0,n_jobs=-1)
        est_gp.fit(X_train_pca,y_train)
        generation_results = []
        for idGen in range(len(est_gp._programs)):
            single_generation = {}
            single_generation[idGen] = math.inf
            for idPopulation in range(est_gp.population_size):
                if(est_gp._programs[idGen][idPopulation] != None):
                    if est_gp._programs[idGen][idPopulation].raw_fitness_ < single_generation[idGen]:
                        single_generation[idGen] = est_gp._programs[idGen][idPopulation].raw_fitness_
            generation_results.append(single_generation)
        run_results[run_number] = generation_results
    return run_results

def std_gp(rows,features,function):
    run_results = {}
    for run_number in range(0,NUMBER_OF_RUNS):
        # Generating random data
        rng = check_random_state(run_number)
        X = rng.uniform(-1, 1, rows).reshape(rows//features, features)
        Y = function(X)

        # Dividing it into training and test set
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = TEST_SIZE, random_state = 0)

        # Training the system
        est_gp = SymbolicRegressor(population_size=POPULATION_SIZE,
                                generations=NUMBER_OF_GENERATION, stopping_criteria=0.01,
                                p_crossover=0.7, p_subtree_mutation=0.1,
                                p_hoist_mutation=0.05, p_point_mutation=0.1,
                                max_samples=0.9, verbose=1,
                                parsimony_coefficient=0.01, random_state=0,n_jobs=-1)
        est_gp.fit(X_train,y_train)
        generation_results = []
        for idGen in range(len(est_gp._programs)):
            single_generation = {}
            single_generation[idGen] = math.inf
            for idPopulation in range(est_gp.population_size):
                if(est_gp._programs[idGen][idPopulation] != None):
                    if est_gp._programs[idGen][idPopulation].raw_fitness_ < single_generation[idGen]:
                        single_generation[idGen] = est_gp._programs[idGen][idPopulation].raw_fitness_
            generation_results.append(single_generation)
        run_results[run_number] = generation_results
    return run_results

def main():
        methods = ['Standard GP','PCA GP','DV GP']
        runs_data = [
            [
                { 'FUNCTION': lambda X: X[:, 0]**2 - X[:, 1]**2 + X[:, 1] - 1 },
                { 'FEATURES': 2 },
                { 'ROWS': 400 },
                { 'TITLE': lambda title,eqn:  title + ' Equation ' + str(eqn)+ ' Total Runs ' + str(NUMBER_OF_RUNS) }
            ],
            [
                { 'FUNCTION': lambda X: 0.3 * X[:,0] * np.sin(2*math.pi*X[:,1])},
                { 'FEATURES': 2 },
                { 'ROWS': 400 },
                { 'TITLE': lambda title,eqn:  title + ' Equation ' + str(eqn)+ ' Total Runs ' + str(NUMBER_OF_RUNS) }
            ],
            [
                { 'FUNCTION': lambda X: X[:,0]**4 - X[:,0]**3 + 0.5*X[:,1]**2 - X[:,1] },
                { 'FEATURES': 2 },
                { 'ROWS': 400 },
                { 'TITLE': lambda title,eqn:  title + ' Equation ' + str(eqn)+ ' Total Runs ' + str(NUMBER_OF_RUNS) }
            ],
            [
                { 'FUNCTION': lambda X: 8/(2 + X[:,0]**2 + X[:,1]**2) },
                { 'FEATURES': 2 },
                { 'ROWS': 400 },
                { 'TITLE': lambda title,eqn:  title + ' Equation  ' + str(eqn)+ ' ' + str(NUMBER_OF_RUNS) }
            ],
            [
                { 'FUNCTION': lambda X: X[:,0]*X[:,1] + X[:,2]*X[:,3] + X[:,4]*X[:,5] + X[:,0]*X[:,6]*X[:,7] + X[:,2]*X[:,5]*X[:,8] },
                { 'FEATURES': 9 },
                { 'ROWS': 900 },
                { 'TITLE': lambda title,eqn:  title + ' Equation ' + str(eqn)+ ' Total Runs ' + str(NUMBER_OF_RUNS) }
            ],
            [
                { 'FUNCTION': lambda X: 10 * np.sin(math.pi*X[:,0]*X[:,1]) + 20*  (X[:,3]-.5)**2 + 10*X[:,4] + 5*X[:,5] },
                { 'FEATURES': 6 },
                { 'ROWS': 600 },
                { 'TITLE': lambda title,eqn:  title + ' Equation ' + str(eqn)+ ' Total Runs ' + str(NUMBER_OF_RUNS) }
            ]
        ]
        for method in methods:
            print('Running: '+ method)
            for details in range(0,len(runs_data)):
                title = runs_data[details][3]['TITLE'](method,details)
                print(title)
                if method == 'Standard GP':
                    result = std_gp(
                    rows = runs_data[details][2]['ROWS'], 
                    features = runs_data[details][1]['FEATURES'], 
                    function = runs_data[details][0]['FUNCTION']
                    )
                if method == 'PCA GP':
                    result = pca_gp(
                    rows = runs_data[details][2]['ROWS'], 
                    features = runs_data[details][1]['FEATURES'], 
                    function = runs_data[details][0]['FUNCTION']
                    )
                if method == 'DV GP':
                    result = dv_gp(
                    rows = runs_data[details][2]['ROWS'], 
                    features = runs_data[details][1]['FEATURES'], 
                    function = runs_data[details][0]['FUNCTION']
                    )
                transformed_result = transform_result(result)
                final_result = transform_to_array(transformed_result)
                draw_box_plots(
                data = final_result,
                title = title)
main()
    
