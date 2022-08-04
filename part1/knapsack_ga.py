import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

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
def individual_knapsack_fitness(individual, penalty_coeff, dataset):
    if(not individual.__contains__(1)): return 0    #to avoid empty knapsack situation
    #decode knapsack chromosome to values and weights
    value,weight = zip(*[ dataset[2].iloc[i] for i, value in enumerate(individual) if (value == 1) ])
    #keeping fitness >= 0 since negative will break probability calculations
    return np.max((0, np.sum(value) - penalty_coeff*np.max((0, (np.sum(weight)-dataset[1])))))

"""
----------GA implementation----------

generate GA individual for population
"""
def generate_individual(num_items, weights, max_weight, rng):
    individual = [0] * num_items
    total_weight = 0.0
    unvisited = range(num_items)
    for i in range(num_items):
        index = rng.choice(unvisited, replace=False)
        if(total_weight+weights[index] >= max_weight):   
            return individual
        else:   
            individual[index] = 1
            total_weight += weights[index]
    return individual

"""
one-point crossover operator
"""
def one_point_crossover(parent1, parent2, rng):
    split_pos = rng.integers(low=1,high=(len(parent1)-1))
    child1 = np.concatenate((parent1[0:split_pos],parent2[split_pos:len(parent2)]))
    child2 = np.concatenate((parent2[0:split_pos],parent1[split_pos:len(parent2)]))
    return (child1, child2)

"""
All points here are equal chance of being flipped
"""
def uniform_flip(instance, rng):
    pos = rng.integers(low=0,high=(len(instance)))
    instance[pos] = 0 if(instance[pos] == 1) else 1 #flip mutation
    return instance

"""
Flip the point into the instance with the largest value
"""
def truncate_flip(instance, fitness_func):
    best_instance, best_value = instance.copy(),0
    for i,bit in enumerate(instance):
        flip_bit = 0 if(bit == 1) else 1
        new_instance = instance.copy()
        new_instance[i] = flip_bit
        fitness = fitness_func(new_instance)
        if (fitness > best_value):
            best_instance = new_instance
            best_value = fitness
    return best_instance

"""
flip mutation operator
"""
def mutation(instance, rng, fitness_func):
    #return uniform_flip(instance, rng)
    return truncate_flip(instance,fitness_func)

"""
calculate probabilities for roulette wheel selection
"""
def prob_calc(population, fit_func, pop_fitness):
    if(pop_fitness == 0): return [1.0/len(population) for _ in population] #if pop fitness 0, then all individuals are equally nonfit
    return list(map(lambda x : fit_func(x)/pop_fitness, population))

"""
evaluate fitness of entire population and order population by fitness
"""
def fitness_pop_eval(population, fit_func):
    population.sort(key=fit_func,reverse=True)
    return np.sum(list(map(fit_func, population)))

def gen_new_pop(pop, rng, fitness_func, fitness, pop_size, elitism_rate, crossover_rate, mutation_rate):
    prob = prob_calc(pop, fitness_func,fitness) #calc probabilities for roulette wheel

    new_pop=[]
    new_pop += pop[0:int(elitism_rate * pop_size)] #elitism
    while(len(new_pop) < pop_size):
        parents = rng.choice(pop_size,size=2,p=prob)  #selection
        if (rng.random() <= crossover_rate):
            children = one_point_crossover(pop[parents[0]], pop[parents[1]], rng)#crossover
            children = [mutation(child, rng, fitness_func) if(rng.random() <= mutation_rate) else child for child in children]  #mutation
            new_pop += children
    return new_pop[:pop_size]#avoid new population being bigger than correct pop size

"""
calculate optimal solution through GA
"""
def GA_solution(dataset, seed, pop_size = 50, max_iter = 100, max_convergence_iterations = 5, penalty_coeff = 1, elitism_rate = 0.1, crossover_rate = 1.0, mutation_rate = 0.9):
    rng = np.random.default_rng(seed)   #set up rng so can get consistent results based on seed

    pop=[generate_individual(dataset[0], dataset[2].iloc[:,0].tolist(), dataset[1], rng) for i in range(pop_size)]
    fitness_func = lambda x : individual_knapsack_fitness(x,penalty_coeff,dataset)
    num_iterations = 0   #keeps track of number of iterations done
    current_convergence_iterations = 0  #keep track of how many convergence iterations there have been
    avg_best = []   #store average of top 5 individual in each generation
    prev_avg = -10.0    #start at -10 so never converge at beginning

    for i in range(max_iter):   #repeat until stopping criteria is met
        fitness = fitness_pop_eval(pop, fitness_func)  #calc total fitness of pope
        
        current_avg = np.average([ fitness_func(individual) for individual in pop[:5]])
        if(abs(current_avg - prev_avg) < 0.001):   #check for convergence
            current_convergence_iterations += 1
            if(current_convergence_iterations > max_convergence_iterations): break
        else: current_convergence_iterations = 0    #no more convergence

        prev_avg = current_avg
        avg_best.append(current_avg)
        best_individual = pop[0]    #get best individual from pop

        new_pop = gen_new_pop(pop, rng, fitness_func, fitness, pop_size, elitism_rate, crossover_rate, mutation_rate)
        pop = new_pop
        num_iterations += 1

    best_individual_fitness = fitness_func(best_individual)
    print('best individual = ',best_individual,' fitness = ',best_individual_fitness)
    return avg_best,num_iterations, best_individual_fitness

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
    dataset_parameters.append((100,100,25,0.1,1.0,0.3))

    rng = np.random.default_rng(123)
    seeds = rng.integers(low=0,high=2000,size=5)

    for i in range(len(datasets)):
        print(datasets[i][0])
        GA_output = []  #GA best solution from each seed
        x_values = [] #iterations range for each GA
        y_values = [] #average of 5 best individuals each iteration
        for seed in seeds:
            print('seed = ',seed)
            y,x,best= GA_solution(datasets[i][1],seed, 
                            pop_size = dataset_parameters[i][0],
                            max_iter = dataset_parameters[i][1],
                            penalty_coeff=dataset_parameters[i][2], 
                            elitism_rate=dataset_parameters[i][3], 
                            crossover_rate=dataset_parameters[i][4], 
                            mutation_rate=dataset_parameters[i][5])
            GA_output.append(best)
            x_values.append(range(x))
            y_values.append(y)
        mean = np.mean(GA_output)
        std = np.std(GA_output)
        print('mean = ',mean)
        print('standard deviation = ',std)
        draw_convergence_curves(x_values, y_values, datasets[i][2], datasets[i][0], seeds, mean, std)