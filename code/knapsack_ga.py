import pandas as pd
import numpy as np

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
----------GA for knapsack problem specifically----------

evaluate fitness of individual in knapsack problem
"""
def individual_knapsack_fitness(individual, dataset):
    #decode knapsack chromosome to values and weights
    value,weight = zip(*[ dataset[2].iloc[i] for i, value in enumerate(individual) if (value == 1) ])
    penalty_coeff = 10.0 #10 is enough to impact fitness if violate constraint
    #keeping fitness >= 0 since negative will break probability calculations
    return np.max((0, np.sum(value) - penalty_coeff*np.max((0, (np.sum(weight)-dataset[1])))))

"""
----------GA implementation----------

generate GA individual for population
"""
def generate_individual(num_items):
    rng = np.random.default_rng()
    return rng.integers(low=0,high=2,size=8)

"""
one-point crossover operator
"""
def one_point_crossover(parent1, parent2):
    split_pos = np.random.default_rng().integers(low=1,high=(len(parent1)-1))
    child1 = np.concatenate((parent1[0:split_pos],parent2[split_pos:len(parent2)]))
    child2 = np.concatenate((parent2[0:split_pos],parent1[split_pos:len(parent2)]))
    return (child1, child2)

"""
flip mutation operator
"""
def mutation(instance):
    mutation_pos = np.random.default_rng().integers(low=0,high=(len(instance)))
    instance[mutation_pos] = 0 if(instance[mutation_pos] == 1) else 1 #flip mutation
    return instance

"""
calculate probabilities for roulette wheel selection
"""
def prob_calc(population, fit_func, pop_fitness):
    return list(map(lambda x : fit_func(x)/pop_fitness, population))
    
    #prob = fit_func(instance)/pop_fitness
    #r = np.random.default_rng().random()
    #return (r <= prob)
"""
evaluate fitness of entire population and order population by fitness
"""
def fitness_pop_eval(population, fit_func):
    population.sort(key=fit_func,reverse=True)
    return np.sum(list(map(fit_func, population)))

"""
calculate optimal solution through GA
"""
#todo: experiment with pop size
def GA_solution(dataset, pop_size = 10, max_iter = 2000, elitism_rate = 0.03, crossover_rate = 1.0, mutation_rate = 0.3):
    pop=[generate_individual(dataset[0]) for i in range(pop_size)]
    fitness_func = lambda x : individual_knapsack_fitness(x,dataset)
    fitness = fitness_pop_eval(pop, fitness_func)  #calc total fitness of pop
    prob = prob_calc(pop, fitness_func,fitness)

    #generate new pop
    new_pop=[]
    new_pop += pop[0:int(elitism_rate * pop_size)] #elitism
    while(len(new_pop) < pop_size):
        parents = np.random.choice(pop_size,size=2,p=prob)  #selection
        if (np.random.default_rng().random() <= crossover_rate):
            children = one_point_crossover(pop[parents[0]], pop[parents[1]])#crossover
            #mutation
            children = [mutation(child) if(np.random.default_rng().random() <= mutation_rate) else child for child in children]
            new_pop += children
    #repeat until stopping criteria
        
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
    