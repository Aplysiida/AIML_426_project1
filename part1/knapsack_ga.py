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
    #penalty_coeff = 3 #3 is enough to impact fitness if violate constraint
    #keeping fitness >= 0 since negative will break probability calculations
    #print('value = ',np.sum(value), ' weight = ',np.sum(weight),' max weight = ',dataset[1])
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
flip mutation operator
"""
def mutation(instance, rng):
    mutation_pos = rng.integers(low=0,high=(len(instance)))
    instance[mutation_pos] = 0 if(instance[mutation_pos] == 1) else 1 #flip mutation
    return instance

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

def gen_new_pop(pop, rng, prob, pop_size, elitism_rate, crossover_rate, mutation_rate):
    new_pop=[]
    new_pop += pop[0:int(elitism_rate * pop_size)] #elitism
    while(len(new_pop) < pop_size):
        parents = rng.choice(pop_size,size=2,p=prob)  #selection
        if (rng.random() <= crossover_rate):
            children = one_point_crossover(pop[parents[0]], pop[parents[1]], rng)#crossover
            #mutation
            children = [mutation(child, rng) if(rng.random() <= mutation_rate) else child for child in children]
            new_pop += children
    return new_pop[:pop_size]#avoid new population being bigger than correct pop size

"""
calculate optimal solution through GA
"""
def GA_solution(dataset, seed, pop_size = 100, max_iter = 200, max_convergence_iterations = 5, penalty_coeff = 1, elitism_rate = 0.1, crossover_rate = 1.0, mutation_rate = 0.9):
    rng = np.random.default_rng(seed)   #set up rng so can get consistent results based on seed

    pop=[generate_individual(dataset[0], dataset[2].iloc[:,0].tolist(), dataset[1], rng) for i in range(pop_size)]
    fitness_func = lambda x : individual_knapsack_fitness(x,penalty_coeff,dataset)
    num_iterations = 0   #keeps track of number of iterations done
    current_convergence_iterations = 0  #keep track of how many convergence iterations there have been
    avg_best = []   #store average of top 5 individual in each generation
    prev_avg = -10.0    #start at -10 so never converge at beginning

    for i in range(max_iter):   #repeat until stopping criteria is met
        fitness = fitness_pop_eval(pop, fitness_func)  #calc total fitness of pope
        #if(abs(fitness-prev_fitness) < 0.1): break  #check for convergence
        
        current_avg = np.average([ fitness_func(individual) for individual in pop[:5]])
        if(abs(current_avg - prev_avg) < 0.01):   #check for convergence
            current_convergence_iterations += 1
            if(current_convergence_iterations > max_convergence_iterations): break
        else: current_convergence_iterations = 0

        prev_avg = current_avg
        avg_best.append(current_avg)
        best_individual = pop[0]    #get best individual from pop

        prob = prob_calc(pop, fitness_func,fitness) #calc probabilities for roulette wheel
        new_pop = gen_new_pop(pop, rng, prob, pop_size, elitism_rate, crossover_rate, mutation_rate)
        pop = new_pop
        num_iterations += 1

    best_individual_fitness = fitness_func(best_individual)
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
    datasets.append(('100_1000: ',read_dataset(data_folder_path+'100_995'),1514))
    
    dataset_parameters = [] #stores tuples as (penalty value, elitism rate, crossover rate, mutation rate)
    dataset_parameters.append((2,0.03,1.0,0.3))
    dataset_parameters.append((3,0.03,1.0,0.3))
    dataset_parameters.append((2.09,0.1,1.0,0.4))

    rng = np.random.default_rng(12)
    seeds = rng.integers(low=0,high=2000,size=5)
    iterations_num = 2

    for i in range(len(datasets)):
        print(datasets[i][0])
        GA_output = []  #GA best solution from each seed
        x_values = [] #iterations range for each GA
        y_values = [] #average of 5 best individuals each iteration
        for seed in seeds:
            print('seed = ',seed)
            print('parameters = ',dataset_parameters[i])
            y,x,best= GA_solution(datasets[i][1],seed, 
                            penalty_coeff=dataset_parameters[i][0], 
                            elitism_rate=dataset_parameters[i][1], 
                            crossover_rate=dataset_parameters[i][2], 
                            mutation_rate=dataset_parameters[i][3])
            print('y = ',y)
            GA_output.append(best)
            x_values.append(range(x))
            y_values.append(y)
        mean = np.mean(GA_output)
        std = np.std(GA_output)
        print('mean = ',mean)
        print('standard deviation = ',std)
        draw_convergence_curves(x_values, y_values, datasets[i][2], datasets[i][0], seeds, mean, std)