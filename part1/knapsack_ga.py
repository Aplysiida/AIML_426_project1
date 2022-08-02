import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    return np.max((0, np.sum(value) - penalty_coeff*np.max((0, (np.sum(weight)-dataset[1])))))

"""
----------GA implementation----------

generate GA individual for population
"""
def generate_individual(num_items, rng):
    return rng.integers(low=0,high=2,size=num_items)

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
#todo: experiment with pop size
def GA_solution(dataset, seed, pop_size = 100, max_iter = 200, max_convergence_iterations = 5, penalty_coeff = 1, elitism_rate = 0.1, crossover_rate = 1.0, mutation_rate = 0.9):
    rng = np.random.default_rng(seed)   #set up rng so can get consistent results based on seed

    pop=[generate_individual(dataset[0], rng) for i in range(pop_size)]
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
        #best_individual = pop[0]    #get best individual from pop

        prob = prob_calc(pop, fitness_func,fitness) #calc probabilities for roulette wheel
        new_pop = gen_new_pop(pop, rng, prob, pop_size, elitism_rate, crossover_rate, mutation_rate)
        pop = new_pop
        num_iterations += 1

    """value_sum = lambda inst : np.sum([dataset[2].iloc[i]['Value'] for i, value in enumerate(pop[0]) if (value == 1)])
    print('--------')
    for p in pop:
        print(p,' fitness = ',value_sum(p))
    print('--------')

    print('best_individual = ',best_individual,' fitness = ', value_sum(best_individual))   """
    return avg_best,num_iterations

"""
Draw convergence curves for each seed and iteration
"""
def draw_convergence_curves(x_values, y_values, optimal_value, dataset_name, seeds, iterations_num):
    fig = plt.figure(constrained_layout=True)
    fig.suptitle(dataset_name)
    subfigs = fig.subfigures(nrows=iterations_num,ncols=1)

    #for i in range(iterations_num):
    for i,seed in enumerate(seeds):
        subfig = subfigs[i]
        subfig.suptitle('Iteration:'+str(i))
        axis = subfig.subplots(nrows=1, ncols=len(seeds))
        for j in range(iterations_num):
            axis[j].yaxis.grid(True)
            axis[j].axhline(optimal_value, color='green', linewidth=1.0)
            axis[j].plot(x_values[j], y_values[j],c='red')
            axis[j].set_title('Seed: '+str(seed))

    plt.show()


if __name__=="__main__":
    dataset1 = ('10_269', read_dataset('knapsack-data/10_269'), 295)
    dataset2 = ('23_10000',read_dataset('knapsack-data/23_10000'),9767)
    dataset3 = ('100_1000',read_dataset('knapsack-data/100_1000'),1514)
    
    #parameters = (penalty value, elitism rate, crossover rate, mutation rate)
    dataset1_parameters = (2,0.03,1.0,0.3)
    dataset2_parameters = (3,0.03,1.0,0.3)
    dataset3_parameters = (1,0.03,1.0,0.3)

    rng = np.random.default_rng(123)
    seeds = rng.integers(low=0,high=2000,size=2)
    iterations_num = 2

    x_values = [] #iterations range for each GA
    y_values = [] #average of 5 best individuals each iteration

    dataset = dataset3
    dataset_parameters = dataset3_parameters

    for seed in seeds:
        print('seed = ',seed)
        for i in range(iterations_num):
            print('i = ',i)
            y,x= GA_solution(dataset[1],seed, 
                            penalty_coeff=dataset_parameters[0], 
                            elitism_rate=dataset_parameters[1], 
                            crossover_rate=dataset_parameters[2], 
                            mutation_rate=dataset_parameters[3])
            x_values.append(range(x))
            y_values.append(y)
    
    draw_convergence_curves(x_values, y_values, dataset[2], dataset[0], seeds, iterations_num)
    