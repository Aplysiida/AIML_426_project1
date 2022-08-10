import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

import genetic_algo

"""
read dataset into tuple (num items, bag capacity, items dataframe)
"""
def read_dataset(filepath):
    df = pd.read_table(filepath,sep=' ')
    columns = (int(df.columns.values[0]),int(df.columns.values[1]))
    df.columns.values[0], df.columns.values[1] = 'Value', 'Weight'  #rename column names to Value, Weight
    dataset = (columns[0], columns[1], df) #([0]num_items, [1]bag_capacity, [2]dataframe)
    return dataset

"""
evaluate fitness of individual in knapsack problem
"""
def individual_knapsack_fitness(individual, penalty_coeff, dataset, max_weight):
    if(not individual.__contains__(1)): return 0    #to avoid empty knapsack situation
    #decode knapsack chromosome to values and weights
    value,weight = zip(*[ dataset.iloc[i] for i, value in enumerate(individual) if (value == 1) ])
    #keeping fitness >= 0 since negative will break probability calculations
    return np.max((0, np.sum(value) - penalty_coeff*np.max((0, (np.sum(weight)-max_weight)))))

"""
maximum weight constraint for the knapsack problem
"""
def knapsack_constraint_check(individual, dataset, max_weight):
    _,weight =zip(*[ dataset.iloc[i] for i, value in enumerate(individual) if (value == 1) ])
    return np.sum(weight) < max_weight

"""
Draw convergence curves for each seed and iteration
"""
def draw_convergence_curves(x_values, y_values, optimal_value, dataset_name, seeds, mean, std):
    fig = plt.figure(constrained_layout=True)
    fig.suptitle(dataset_name+' Mean: '+str(mean)+' Standard Deviation: '+str(std))
    subfigs = fig.subfigures(nrows=len(seeds),ncols=1)

    #for i in range(iterations_num):
    for i,seed in enumerate(seeds):
        subfigs[i].suptitle('Seed: '+str(seed))
        axis = subfigs[i].subplots(nrows=1, ncols=1)
        axis.yaxis.grid(True)
        axis.axhline(optimal_value, color='green', linewidth=1.0)
        axis.plot(x_values[i], y_values[i],c='red')

    plt.show()


if __name__=="__main__":
    data_folder_path = sys.argv[1]

    datasets = [] #stores tuples as (dataset name, dataset, optimal value)
    datasets.append(('10_269: ', read_dataset(data_folder_path+'10_269'), 295))
    datasets.append(('23_10000: ',read_dataset(data_folder_path+'23_10000'),9767))
    datasets.append(('100_995: ',read_dataset(data_folder_path+'100_995'),1514))
    
    dataset_parameters = [] #stores tuples as (pop size, max iterations, penalty value, elitism rate, crossover rate, mutation rate)
    dataset_parameters.append((50,100,2,0.03,1.0,0.3))
    dataset_parameters.append((75,100,3,0.03,1.0,0.3))
    dataset_parameters.append((100,100,40,0.1,1.0,0.3))

    seed_rng = np.random.default_rng(123)
    seeds = seed_rng.integers(low=0,high=2000,size=5)

    for i in range(len(datasets)):
        penalty_coeff = dataset_parameters[i][2]
        dataset = datasets[i][1][2]
        max_weight = datasets[i][1][1]
        item_length = datasets[i][1][0]

        fitness_func = lambda x : individual_knapsack_fitness(x,penalty_coeff,dataset,max_weight)
        constraint = lambda x : knapsack_constraint_check(x,dataset,max_weight)

        GA_output = []  #GA best solution from each seed
        x_values = [] #iterations range for each GA
        y_values = [] #average of 5 best individuals each iteration
        for seed in seeds:  #iterate through each seed
            print('seed = ',seed)
            rng = np.random.default_rng(seed)   #set up rng so can get consistent results based on seed
            ga = genetic_algo.GA(rng, dataset, item_length, fitness_func, constraint)
            y,x,best = ga.GA_solution(pop_size = dataset_parameters[0][0],
                   max_iter = dataset_parameters[0][1],
                   elitism_rate=dataset_parameters[0][3],
                   crossover_rate=dataset_parameters[0][4],
                   mutation_rate=dataset_parameters[0][5])
            GA_output.append(best)
            x_values.append(range(x))
            y_values.append(y)

        mean = np.mean(GA_output)
        std = np.std(GA_output)
        print('mean = ',mean)
        print('standard deviation = ',std)
        draw_convergence_curves(x_values, y_values, datasets[i][2], datasets[i][0], seeds, mean, std)