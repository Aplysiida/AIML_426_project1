import enum
from deap import base
from deap import creator
from deap import tools

import numpy as np

import random

"""
Functions defined for problem for Part 5
"""
def rosenbrock(points):
    sum = 0.0
    for i in range(len(points)-1):
        a = points[i]
        b = points[i+1]
        sum += 100.0 * pow(pow(a,2) - b, 2) + pow(a - 1.0 ,2)
    return sum

def griewanks(points):
    sum = 0.0
    product = 1.0
    for i,x in enumerate(points):
        sum += (pow(x,2))/4000.0
        product *= np.cos(x/(np.sqrt(i+1)))
    return sum - product + 1.0

def fitness(particle, problem_func):
    return [problem_func(particle)]

"""
Generate particle with random position and velocity
"""
def generate_particle(dimensions, pos_min, pos_max, vel_min, vel_max):
    particle = creator.Particle(random.uniform(pos_min, pos_max) for _ in range(dimensions))
    particle.velocity = [random.uniform(vel_min, vel_max) for _ in range(dimensions)]
    particle.vmin = vel_min
    particle.vmax = vel_max
    return particle

"""
Update particle's velocity and position
"""
def update_particle(particle, rng, global_best, w, phi1, phi2):
    #update velocity
    r1 = rng.random() * phi1
    r2 = rng.random() * phi2
    cog_comp = np.multiply(r1, np.subtract(particle.best, particle)).tolist()
    soc_comp = np.multiply(r2, np.subtract(global_best, particle)).tolist()

    particle.velocity = np.add(particle.velocity, np.add(cog_comp, soc_comp)).tolist()
    #add inertia weight
    particle.velocity = np.multiply(particle.velocity, w)
    #clamp velocity
    particle.velocity = np.clip(particle.velocity, a_min=particle.vmin, a_max=particle.vmax).tolist()
    
    #update pos
    particle[:] = np.add(particle, particle.velocity).tolist()

"""
Setup the DEAP toolbox for PSO
"""
def setup_toolbox(problem_func, rng, dimensions_num, pmin, pmax, vmin, vmax, w, phi1, phi2):
    toolbox = base.Toolbox()

    toolbox.register(
        'particle', 
        generate_particle, 
        dimensions=dimensions_num, 
        pos_min=pmin, pos_max=pmax, 
        vel_min=vmin, vel_max=vmax
    )
    toolbox.register('population', tools.initRepeat, list, toolbox.particle)
    toolbox.register('update', update_particle, rng=rng, w=w, phi1=phi1, phi2=phi2)
    toolbox.register('evaluate',fitness,problem_func=problem_func)

    return toolbox

def run_pso(toolbox, pop_size, max_iter):
    pop = toolbox.population(n=pop_size)

    global_best = None
    for gen in range(max_iter):
        for particle in pop:
            particle.fitness.values = toolbox.evaluate(particle)
            #update particle best
            if not particle.best or particle.best.fitness < particle.fitness:
                particle.best = creator.Particle(particle)
                particle.best.fitness.values = particle.fitness.values
            #update global best
            if not global_best or global_best.fitness < particle.fitness:
                global_best = creator.Particle(particle)
                global_best.fitness.values = particle.fitness.values
        #update particle
        for particle in pop:
            toolbox.update(particle, global_best=global_best)
    return global_best #return best solution from population

if __name__=="__main__":
    D = 20
    #create pso
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMin, velocity=list, vmin=None, vmax=None, best=None)    

    #run pso
    problem_funcs = (rosenbrock, griewanks)
    func_names = ('rosenbrock','griewanks')
    #parameters for pso stored in tuple as (pop size, max iterations, inertia w, chi1, chi2)
    pso_parameters = [(100, 100, 0.7, 1.5, 1.5), (100, 100, 0.7, 1.5, 1.5)]
    seeds = np.random.default_rng(seed = 15).integers(low=0, high=200, size=30)

    #run 30 times for each function
    for i, func in enumerate(problem_funcs):
        print('at function: ',func_names[i])
        pso_parameter = pso_parameters[i]
        best_solutions_fitness = []
        for seed in seeds:
            random.seed(int(seed))
            rng = np.random.default_rng(seed=seed)
            toolbox = setup_toolbox(
                rosenbrock, 
                rng, 
                D, 
                pmin=-30, pmax=30, 
                vmin=-5, vmax=5,
                w = pso_parameter[2], 
                phi1=pso_parameter[3], phi2=pso_parameter[4]
            )
            best = run_pso(toolbox, pop_size=pso_parameter[0], max_iter=pso_parameter[1])
            best_solutions_fitness.append(fitness(best, rosenbrock))
        print('mean value obtained  = ',np.mean(best_solutions_fitness))
        print('std deviation obtained = ',np.std(best_solutions_fitness))

    #run 30 times for Griewanks with D=50
    D=50
    seeds = np.random.default_rng(seed = 15).integers(low=0, high=200, size=30)
    print('Running ',func_names[1],' for D=',D)
    pso_parameter = pso_parameters[1]
    best_solutions_fitness = []
    for seed in seeds:
        random.seed(int(seed))
        rng = np.random.default_rng(seed=seed)
        toolbox = setup_toolbox(
            rosenbrock, 
            rng, 
            D, 
            pmin=-30, pmax=30, 
            vmin=-5, vmax=5,
            w = pso_parameter[2], 
            phi1=pso_parameter[3], phi2=pso_parameter[4]
        )
        best = run_pso(toolbox, pop_size=pso_parameter[0], max_iter=pso_parameter[1])
        best_solutions_fitness.append(fitness(best, rosenbrock))
    print('mean value obtained  = ',np.mean(best_solutions_fitness))
    print('std deviation obtained = ',np.std(best_solutions_fitness))