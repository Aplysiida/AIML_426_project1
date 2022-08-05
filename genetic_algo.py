import numpy as np

class GA:
    #Public functions
    """
    Constructors GA Parameters: rng, fitness_func, constraint(should return true if satisfied)
    """
    def __init__(self, rng, dataset, individual_length, fitness_func, constraint_func=-1):
        self.rng = rng
        self.dataset = dataset
        self.individual_length = individual_length
        self.fitness_func = fitness_func
        self.constraint = constraint_func

    def GA_solution(self, pop_size = 50, max_iter = 100, max_convergence_iterations = 5, elitism_rate = 0.1, crossover_rate = 1.0, mutation_rate = 0.9):
        pop = [self._generate_individual(self.individual_length, self.rng, self.constraint) for i in range(pop_size)]
        current_convergence_iterations = 0  #keep track of how many convergence iterations there have been
        num_iterations = 0  #keeps track of number of iterations done
        avg_best = []   #store average of top 5 individual in each generation
        prev_avg = -10.0    #start at -10 so never converge at beginning

        for i in range(max_iter):   #repeat until stopping criteria is met
            fitness = self._fitness_pop_eval(pop, self.fitness_func)  #calc total fitness of pop
            
            current_avg = np.average([self.fitness_func(individual) for individual in pop[:5]])
            if(abs(current_avg - prev_avg) < 0.001):   #check for convergence
                current_convergence_iterations += 1
                if(current_convergence_iterations > max_convergence_iterations): break
            else: current_convergence_iterations = 0    #no longer at convergence

            #update best averages
            prev_avg = current_avg
            avg_best.append(current_avg)
            best_individual = pop[0]    #get best individual from pop

            #new_pop = self._gen_pop(pop, self.rng, self.fitness_func, fitness, pop_size, elitism_rate, crossover_rate, mutation_rate)   
            #generate new population for next generation
            new_pop = self._gen_pop(pop, fitness, pop_size, self.rng, self.fitness_func, elitism_rate, crossover_rate, mutation_rate)
            pop = new_pop

            num_iterations += 1

        best_individual = pop[0]
        best_individual_fitness = self.fitness_func(best_individual)
        return avg_best,num_iterations, best_individual_fitness

    #Private functions
    #Genetic Operators
    """
    one-point crossover operator
    """
    def _one_point_crossover(self, parent1, parent2, rng):
        split_pos = rng.integers(low=1,high=(len(parent1)-1))
        child1 = np.concatenate((parent1[0:split_pos],parent2[split_pos:len(parent2)]))
        child2 = np.concatenate((parent2[0:split_pos],parent1[split_pos:len(parent2)]))
        return (child1, child2)

    """
    All points here are equal chance of being flipped
    """
    def _uniform_flip(self, instance, rng):
        pos = rng.integers(low=0,high=(len(instance)))
        instance[pos] = 0 if(instance[pos] == 1) else 1 #flip mutation
        return instance

    """
    Flip the point into the instance with the largest value
    """
    def _truncate_flip(self, instance, fitness_func):
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
    def _mutation(self, instance, rng, fitness_func):
        #return self._uniform_flip(instance, rng)
        return self._truncate_flip(instance,fitness_func)

    #Selection Scheme
    def _roulette_wheel(self, rng, fitness_func, pop, pop_fitness, pop_size):
        #calc probabilities of instances in population for roulette wheel
        #if pop fitness 0, then all individuals are equally nonfit
        prob = [1.0/pop_size for _ in pop] if (pop_fitness == 0) else list(map(lambda x : fitness_func(x)/pop_fitness, pop)) 
        parents = rng.choice(pop_size,size=2,p=prob)  #selection
        return parents

    """
    selection operator
    """
    def _selection(self, rng, fitness_func, pop, pop_fitness, pop_size):
        return self._roulette_wheel(rng, fitness_func, pop, pop_fitness, pop_size)

    #Fitness Evaluation
    """
    evaluate fitness of entire population and order population by fitness
    """
    def _fitness_pop_eval(self, population, fit_func):
        population.sort(key=fit_func,reverse=True)
        return np.sum(list(map(fit_func, population)))

    #Generating individuals and population
    """
    generate a single individual not using reproduction, used for initial population
    """
    def _generate_individual(self, individual_length, rng, constraint=-1):
        if(constraint == -1): #no constraint lambda defined for this GA, juts randomly generate individuals
            return rng.integers(low=0,high=2,size=individual_length)
        else:   #if constraint lambda defined, make sure individual generated doesn't violate constraint
            individual = [0] * individual_length

            unvisited = range(individual_length)
            for i in range(individual_length):
                index = rng.choice(unvisited, replace=False)
                new_individual = individual.copy()
                new_individual[index] = 1
                if(not constraint(new_individual)): #if latest change violate constraint then return individual
                    return individual
                else:
                    individual[index] = 1
                return individual

    """
    Generate initial population for GA
    """
    def _gen_initial_pop(self,pop_size, individual_length, rng, constraint):
        return [self._generate_individual(individual_length,rng,constraint) for individual in range(pop_size)]

    """
    Generate population for new generation
    """
    def _gen_pop(self, original_pop, original_pop_fitness, pop_size, rng, fitness_func, elitism_rate, crossover_rate, mutation_rate):
        new_pop = original_pop[0:int(elitism_rate * pop_size)].copy()    #elitism
        while(len(new_pop) < pop_size):
            parents = self._selection(rng, fitness_func, original_pop, original_pop_fitness, pop_size) #selection
            if(rng.random() <= crossover_rate):
                children = self._one_point_crossover(original_pop[parents[0]], original_pop[parents[1]], rng) #crossover
                children = [self._mutation(child, rng, fitness_func) for child in children]    #mutation
                new_pop += children
        return new_pop[:pop_size]#avoid new population being bigger than correct pop size