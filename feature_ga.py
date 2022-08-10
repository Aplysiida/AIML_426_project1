import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import sys
import time

import genetic_algo

"""
read dataset as a dataframe with each row being an instance + [class label] and columns = [feature names]+[class]
"""
def read_dataset(dataset_path):
    names_file = open(dataset_path[1]+dataset_path[0]+'.names', 'r')
    names_file.readline()   #skip class values for now
    columns = [line.split()[0][:-1] for line in names_file] #get names
    names_file.close()

    feature_num = len(columns)

    #get data
    data = pd.read_table(dataset_path[1]+dataset_path[0]+'.data', sep = ',',header=None)
    data.columns = columns+['class']

    #get labels
    labels = data.iloc[:,feature_num]
    labels.columns = ['class']
    data = data.drop(labels='class',axis=1) #remove labels from data

    return (data,labels),feature_num

"""
Wrapper fitness function
"""
#to fix, need same rng as rest of genetic_algo
def individual_wrapper_fitness(individual, data, labels, seed):
    rng = np.random.default_rng(seed)   #split at same point always to keep consistent results
    to_drop = [i for i,value in enumerate(individual) if (value == 0)]
    subset_data = data.drop(data.columns[to_drop],axis=1) #transform dataset to have only selected features
    #split dataset into training, testing
    data_train, data_test, label_train, label_test = train_test_split(subset_data, labels, test_size = 0.2, random_state = rng.integers(low=0,high=200))
    #train classifier
    #predicted = MLPClassifier(max_iter = 1000, random_state=rng.integers(low=0,high=200)).fit(data_train, label_train).predict(data_test)
    #predicted = DecisionTreeClassifier(random_state=rng.integers(low=0,high=200)).fit(data_train, label_train).predict(data_test)
    predicted = KNeighborsClassifier(n_neighbors=5).fit(data_train, label_train).predict(data_test)
    #predicted = GaussianNB().fit(data_train, label_train).predict(data_test)
    accuracy = accuracy_score(label_test, predicted)    #calc accuracy
    return accuracy

"""
Filter Fitness function
"""
def individual_filter_function():
    print('hi')

"""
Draw convergence curves for each seed and iteration
"""
def draw_convergence_curves(x_values, y_values, dataset_name, seeds, mean, std,optimal_value=None):
    fig = plt.figure(constrained_layout=True)
    fig.suptitle(dataset_name+' Mean: '+str(mean)+' Standard Deviation: '+str(std))
    subfigs = fig.subfigures(nrows=len(seeds),ncols=1)

    for i,seed in enumerate(seeds):
        subfigs[i].suptitle('Seed: '+str(seed))
        axis = subfigs[i].subplots(nrows=1, ncols=1)
        axis.yaxis.grid(True)
        if(optimal_value != None):
            axis.axhline(optimal_value, color='green', linewidth=1.0)
        axis.plot(x_values[i], y_values[i],c='red')

    plt.show()

if __name__=="__main__":
    data_folder_paths = (('wbcd',sys.argv[1]), ('sonar',sys.argv[2])) #get tuple (dataset name, path to folder)

    dataset_names = ['wbcd','sonar']
    datasets = []
    datasets.append(read_dataset(data_folder_paths[0]))
    datasets.append(read_dataset(data_folder_paths[1]))
    
    dataset_parameters = [] #stores tuples as (pop size, max iterations, elitism rate, crossover rate, mutation rate)
    dataset_parameters.append((10,20,0.03,1.0,0.3))
    dataset_parameters.append((40,100,0.03,1.0,0.3))

    seed_rng = np.random.default_rng(123)
    seeds = seed_rng.integers(low=0,high=200,size=5)    

    dataset = datasets[0]
    dataset_parameter = dataset_parameters[0]

    for i,dataset in enumerate(datasets):
        dataset_parameter = dataset_parameters[i]
        print('at dataset ',dataset_names[i],':')

        GA_output = []  #GA best solution from each seed
        x_values = [] #iterations range for each GA 
        y_values = [] #average of 5 best individuals each iteration

        for seed in seeds:
            print('seed = ',seed)
            rng = np.random.default_rng(seed)
            fitness_func = lambda x : individual_wrapper_fitness(x,dataset[0][0],dataset[0][1],1)
    
            item_length = dataset[1]
            WrapperGA = genetic_algo.GA(rng, dataset, item_length, fitness_func)
            y,x,best = WrapperGA.GA_solution(
                pop_size = dataset_parameter[0],
                max_iter = dataset_parameter[1],
                elitism_rate= dataset_parameter[2],
                crossover_rate= dataset_parameter[3],
                mutation_rate= dataset_parameter[4]
            )
            GA_output.append(best)
            x_values.append(range(x))
            y_values.append(y)

        mean = np.mean(GA_output)
        std = np.std(GA_output)
        print('mean = ',mean)
        print('standard deviation = ',std)
        draw_convergence_curves(x_values, y_values, dataset_names[i], seeds, mean, std)