from deap.tools import selNSGA2

import genetic_algo

class Nsga_Algo(genetic_algo.Genetic_Algo):
    def __init__(self, rng, dataset, individual_length):
        self.rng = rng
        self.dataset = dataset
        self.individual_length = individual_length

        genetic_algo.Genetic_Algo.__init__(self, rng, dataset, individual_length, fitness_func=-1, constraint_func=-1)

    """
    Generate population for new generation
    """
    def _gen_pop(self, original_pop, original_pop_fitness, pop_size, rng, fitness, elitism_rate, crossover_rate, mutation_rate):
        """children_pop = []
        while(len(children_pop) < pop_size):
            parents = self._selection(rng, fitness, original_pop, original_pop_fitness, pop_size) #selection
            if(rng.random() <= crossover_rate):
                children = self._one_point_crossover(original_pop[parents[0]], original_pop[parents[1]], rng) #crossover
                children = [self._mutation(child, rng) for child in children]    #mutation
                #convert nd array back to list to keep population type consistent
                children = [child.tolist() for child in children]
                children_pop += children

        #NSGA2 elitism
        combined_pop = original_pop + children_pop
        new_pop = selNSGA2(combined_pop, k=pop_size*elitism_rate,nd='standard')
        return new_pop"""
        return super()._gen_pop(original_pop, original_pop_fitness, pop_size, rng, fitness, elitism_rate, crossover_rate, mutation_rate)

    #initial pop same
    #crossover and mutation same
    #selection same

    #elitism different
    #pop gen different
    """
    children = []
    until children pop full
        select parents
        apply crossover
        apply mutation to children
        add children to children pop
    
    elitism
    combined_pop = combine original pop and children pop
    new_pop = selNSGA2(combined_pop, k=pop_size*elitism_rate,nd='standard')

    return new_pop
    """
