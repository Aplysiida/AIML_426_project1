from deap.tools import selNSGA2

import genetic_algo

class Nsga_Algo(genetic_algo.Genetic_Algo):
    def __init__(self, rng, dataset, individual_length):
        self.rng = rng
        self.dataset = dataset
        self.individual_length = individual_length

        genetic_algo.Genetic_Algo.__init__(self, rng, dataset, individual_length, fitness_func=-1, constraint_func=-1)

    def _selection(self, rng, fitness, pop, pop_fitness, pop_size):
        #return selNSGA2 stuff

        return super()._selection(rng, fitness, pop, pop_fitness, pop_size)
