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
import sys


# sys.stdout = open('output.txt','wt')

def calculatePrevFit():
    f = open("output.txt","r")
    fl =f.readlines()
    print(fl,'asdfasdfasdf')
    fl.reverse()
    average_fit_list = []
    ind = 1
    total = 0
    prev_gen_lim = 5
    avg_gen_ttl = 0
    next_gen = 0
    while len(average_fit_list) < prev_gen_lim:
        splitted = fl[ind].split()
        len_of_splitted = len(splitted)
        if len_of_splitted == 7 and  splitted[0] != '|':
            total += float(splitted[4])
            average_fit_list.append(splitted[4])
        ind += 1
    avg_gen_ttl = total/prev_gen_lim
    next_gen = int(fl[0].split()[0])+1
    return (next_gen,avg_gen_ttl)
def train_model():
    print('train model called')
    # Global Variables
    FEATURES = 1
    NUMBER_OF_GENERATION = 20
    ROWS = 300
    POPULATION_SIZE = 400
    TEST_SIZE = .2
    NUMBER_OF_REGIONS = 3
    # Equation 1
    formula = lambda X: X**2 + X + 1
    # formula = lambda X: 10 * np.sin(math.pi*X[:,0]*X[:,1]) + 20*  (X[:,3]-.5)**2 + 10*X[:,4] + 5*X[:,5]
    NUMBER_OF_RUNS = 30
    # Generating random data
    rng = check_random_state(12)
    X = rng.uniform(-1, 1, ROWS)
    Y = formula(X)

    # Dividing it into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = TEST_SIZE, random_state = 0)

    # Training the system
    est_gp = SymbolicRegressor(population_size=POPULATION_SIZE,
                            generations=10, stopping_criteria=0.01,
                            p_crossover=0.7, p_subtree_mutation=0.1,
                            p_hoist_mutation=0.05, p_point_mutation=0.1,
                            max_samples=0.9, verbose=1,
                            parsimony_coefficient=0.01, random_state=0)
    est_gp.fit(X_train,y_train)
    # print(dir(est_gp))
    print(est_gp._programs[2][2])
    # print(type(self_est.run_details_))
    # est_gp.set_params(generations=6, warm_start=True)
    # est_gp.fit(X_train_pca,y_train)
    # calculatePrevFit()
    # est_gp.set_params(generations=7, warm_start=True)
    # est_gp.fit(X_train_pca,y_train)

    df = pd.DataFrame(columns=['Gen','OOB_fitness','Equation'])

    for idGen in range(len(est_gp._programs)):
        # print(est_gp._programs[idGen].oob_fitness_)
        for idPopulation in range(est_gp.population_size):
            if(est_gp._programs[idGen][idPopulation] != None):
                # print(est_gp._programs[idGen][idPopulation].oob_fitness_)
                df = df.append({'Gen': idGen, 'OOB_fitness': est_gp._programs[idGen][idPopulation].oob_fitness_, 'Equation': str(est_gp._programs[idGen][idPopulation])}, ignore_index=True)

    print('Best 3 models of last generation: ')
    print(df[df['Gen']==df['Gen'].max()].sort_values('OOB_fitness')[:3])

def main():
    train_model()

main()