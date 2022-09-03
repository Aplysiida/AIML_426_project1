from deap import algorithms
from deap import base
from deap import creator
from deap import gp
from deap import tools

import numpy as np

import matplotlib.pyplot as plt

import operator
import random

"""
Protected division to avoid program crashing when dividing by 0
"""
def protected_div(x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return 1.0

"""
Create function and terminal sets for GP
"""
def create_primitive_set():
    pset = gp.PrimitiveSet("main",1)
    #create function set
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(np.sin, 1)
    pset.addPrimitive(np.cos, 1)
    pset.addPrimitive(np.log, 1)
    #create terminal set
    pset.addTerminal(1.0)
    #rename argument ARG0 to x
    pset.renameArguments(ARG0="x")
    return pset

"""
f(x) function from Part 4 problem
"""
def problem_func(x):
    if(x > 0):
        return (1/x) + np.sin(x)
    else:
        return (2.0*x) + pow(x,2) + 3.0

"""
Using mean square error for the fitness with error being the difference between the individual solution function 
and the problem function
    x_values = range of x values to calculate error from
"""
def fitness(individual, x_values):
    ind_func = toolbox.compile(expr=individual)
    sq_errors = [pow(ind_func(x) - problem_func(x) ,2) for x in x_values]
    return np.sum(sq_errors)/len(sq_errors) #return mean square error

"""
"""
def setup_toolbox(x_values, pset):
    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate",fitness,x_values)
    toolbox.register("mut_expr", gp.genHalfAndHalf, min_=0, max_=1)
    toolbox.register("mutate", gp.mutUniform, pset=pset)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("select", tools.selTournament, tournsize = 5)

    #defined to avoid too complicated trees, max height recommended by DEAP documentation
    toolbox.decorate("mate", gp.staticLimit(operator.attrgetter('height')), max_value=17)
    toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter('height')), max_value=17)

    return toolbox

def run_gp(toolbox, pop_size, max_iter, crossover_rate, mutation_rate):
    pop = toolbox.population(n=pop_size)
    best_gp = tools.HallOfFame(1)   #get best genetic program

    #since is like normal GA can just use eaSimple
    pop = algorithms.eaSimple(
        population=pop, 
        toolbox=toolbox, 
        cxpb=crossover_rate,
        mutpb=mutation_rate,
        ngen=max_iter,
        halloffame=best_gp
    )

    return best_gp

if __name__=="__main__":
    #generate dataset from problem function
    rng = np.random.default_rng(seed=1)
    x_num= 30 #number of instances
    x_values = np.linspace(start=-6.0, stop=15.0, num=x_num)
    y_values = [problem_func(x) for x in x_values] #np.vectorize(problem_function)(x_values)

    #set up GP
    pset = create_primitive_set()

    creator.create("fitnessmin", base.Fitness, weights=(-1.0))
    creator.create("individual", gp.PrimitiveTree, fitness=creator.fitnessmin, pset=pset)

    toolbox = setup_toolbox(x_values=x_values, pset=pset)

    #run GP
    gp_parameters = (0,0,0,0)

    seeds = np.random.default_rng.integers(low=0,high=200,size=3)
    random.seed(1)  #define seed

    """
    for i,y in enumerate(y_values):
        print('x = ',x_values[i],' y= ',y)
    plt.scatter(x=x_values,y=y_values)
    plt.show()
    """