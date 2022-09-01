import pandas as pd
import numpy as np

from deap import algorithms
from deap.benchmarks.tools import hypervolume
from deap import base
from deap import creator
from deap import tools

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

import seaborn as sns

import sys

"""
Parses dataset into tuple in format (data, labels, number of features)
    dataset_path = (name of file, path to folder, file format)
"""
def read_dataset(dataset_path):
    filepath = dataset_path[1]+dataset_path[0]+'.'+dataset_path[2]
    data = pd.read_table(filepath,sep='[\s,]+',engine='python',header=None)  #separate based on space or comma
    #separate data and labels
    labels = data.iloc[:,-1]
    labels.columns = ['class']
    data = data.drop(data.columns[-1],axis=1)

    feature_num = len(data.columns)
    return [data, labels, feature_num]

def evaluate_accuracy(individual, classifier, data, labels):
    #calc accuracy
    to_drop = [i for i,value in enumerate(individual) if (value == 0)]
    subset_data = data.drop(data.columns[to_drop],axis=1) #transform dataset to have only selected features

    #train classifier
    if(np.count_nonzero(np.array(subset_data)) == 0):    #if no features selected then is useless
        return 0.0
    predicted = classifier.fit(subset_data, labels).predict(subset_data)
    return accuracy_score(labels, predicted)    #calc accuracy

def evaluate_fitness(individual, classifier, data, labels):
    accuracy = evaluate_accuracy(individual, classifier, data, labels)

    selected_feature_ratio = float(np.count_nonzero(np.array(individual) == 1))/float(len(individual))
    return (accuracy,selected_feature_ratio)

def setup_toolbox(rng, data, labels, feature_num):
    toolbox = base.Toolbox()
    #population
    toolbox.register("attr_binary", rng.integers, 0, 2)
    toolbox.register("individual", tools.initRepeat, creator.individual, toolbox.attr_binary,n=feature_num)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #operators
    toolbox.register("evaluate",lambda x : evaluate_fitness(x, KNeighborsClassifier(n_neighbors=5), data, labels))
    #toolbox.register("evaluate", benchmarks.zdt1)
    toolbox.register("mate",tools.cxOnePoint)
    toolbox.register("mutate",tools.mutFlipBit,indpb=1.0/feature_num)
    toolbox.register("select", tools.selNSGA2)

    return toolbox

def run_nsga2(pop_size, max_iter, crossover_rate, mutation_rate):
    #set up population
    pop = toolbox.population(n=pop_size)
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    pop = toolbox.select(pop, len(pop))

    #run generations
    for gen in range(1,max_iter):
        #generate children
        children_pop = algorithms.varAnd(pop, toolbox, cxpb=crossover_rate, mutpb=mutation_rate)
        #recalculate fitness
        invalid_ind = [ind for ind in children_pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        #select next gen pop
        pop = toolbox.select(pop + children_pop, k=pop_size)

    #return nondominated set of solutions/pareto front, hypervolume of front
    hv = hypervolume(pop,[0.0,1.0])
    return pop, hv

if __name__=="__main__":
    #store paths as (name of file, path to folder, file format)
    datasets_paths = [('vehicle',sys.argv[1], 'dat'),('clean1',sys.argv[2], 'data')]
    datasets_names = ['vehicle','clean1']
    datasets = [read_dataset(dataset_path) for dataset_path in datasets_paths]
    #for musk ignore non numerical features for classifier to work
    datasets[1][0] = datasets[1][0].select_dtypes(include=[np.number])
    datasets[1][2] = len(datasets[1][0].columns)

    #parameters used for each dataset
    dataset_parameters = [(50,100,1.0,0.3), (10,20,1.0,0.3)]

    seeds = np.random.default_rng(seed=1).integers(low=0,high=200,size=3)

    creator.create("fitnessfeature", base.Fitness, weights=(1.0, -1.0))
    creator.create("individual", list, fitness=creator.fitnessfeature)

    #for each dataset
    for i,(data, labels, feature_num) in enumerate(datasets):
        print('at dataset ',datasets_names[i])
        dataset_parameter = dataset_parameters[i]
        for seed in seeds:
            print('seed = ',seed)
            rng = np.random.default_rng(seed=seed)
            toolbox = setup_toolbox(rng, data, labels, feature_num)
            final_set, hv = run_nsga2(
                pop_size=dataset_parameter[0],
                max_iter=dataset_parameter[1],
                crossover_rate=dataset_parameter[2],
                mutation_rate=dataset_parameter[3]
            )
            #calc objectives fitness of solutions
            x_values = []
            y_values = []
            for solution in final_set:
                #calc x
                x = evaluate_accuracy(solution, KNeighborsClassifier(n_neighbors=5), data, labels)
                #calc y
                y = np.count_nonzero(np.array(solution) == 1)/len(solution)
                x_values.append(x)
                y_values.append(y)

            #draw figure of front
            print('hypervolume of solution = ',hv)
            sns.scatterplot(x=x_values,y=y_values)
            plt.show()