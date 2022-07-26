import pandas as pd
import numpy as np

"""
read dataset into tuple (num items, bag capacity, items dataframe)
"""
def read_dataset(filepath):
    df = pd.read_table(filepath,sep=' ')
    columns = (int(df.columns.values[0]),int(df.columns.values[1]))
    df.columns.values[0], df.columns.values[1] = 'Value', 'Weight'  #rename column names to Value, Weight
    dataset = (num_items, bag_capacity, dataframe) = (columns[0], columns[1], df) 
    return dataset


"""
evaluate fitness of individual in knapsack problem
"""
def individual_knapsack_fitness(individual, dataset):
    #decode knapsack chromosome to values and weights
    value,weight = zip(*[ dataset[2].iloc[i] for i, value in enumerate(individual) if (value == 1) ])
    return np.sum(value)

"""
----------GA implementation-----------

generate GA individual for population
"""
def generate_individual(num_items):
    rng = np.random.default_rng()
    return rng.integers(low=0,high=2,size=8)

"""
evaluate fitness of entire population
"""
def fitness_pop_eval(population, fit_func):
    return np.sum(list(map(fit_func, population)))

"""
calculate optimal solution through GA
"""
#todo: experiment with pop size
def GA_solution(dataset, pop_size = 5, ):
    pop=[generate_individual(dataset[0]) for i in range(pop_size)]
    fitness = fitness_pop_eval(pop, lambda x : individual_knapsack_fitness(x,dataset))  #calc fitness of pop
    #repeat until stopping criteria
        #generate new pop
            #for each individual
                #selection
                #crossover and mutation
                #children
        #evaluate fitness of new pop

    #get best individual from final pop

if __name__=="__main__":
    dataset1 = read_dataset('knapsack-data/10_269')
    dataset2 = read_dataset('knapsack-data/23_10000')
    dataset3 = read_dataset('knapsack-data/100_1000')
    
    GA_solution(dataset1)
    