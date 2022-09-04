from deap import algorithms
from deap import base
from deap import creator
from deap import gp
from deap import tools

import numpy as np

import matplotlib.pyplot as plt
import pygraphviz as pgv

import operator
import random

"""
Protected division to avoid program crashing when dividing by 0
"""
def protected_div(x,y):
    #print('x = ',x,' y = ',y)
    try:
        return x/y
    except (ZeroDivisionError, FloatingPointError):
        return 1.0

"""
Create function and terminal sets for GP
"""
def create_primitive_set():
    pset = gp.PrimitiveSet("main",1)
    #create function set
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(np.sin, 1)
    pset.addPrimitive(np.cos, 1)
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
    return [np.sum(sq_errors)/len(sq_errors)] #return mean square error

"""
Setup the DEAP toolbox for GP
"""
def setup_toolbox(x_values, pset):
    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate",fitness,x_values=x_values)
    toolbox.register("mut_expr", gp.genHalfAndHalf, min_=0, max_=1)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.mut_expr, pset=pset)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("select", tools.selTournament, tournsize = 5)

    #defined to avoid too complicated trees, max height recommended by DEAP documentation
    toolbox.decorate("mate", gp.staticLimit(operator.attrgetter('height'), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter('height'), max_value=17))

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
        halloffame=best_gp,
        verbose=False
    )

    return best_gp[0]

if __name__=="__main__":
    #generate dataset from problem function
    rng = np.random.default_rng(seed=1)
    x_num= 30 #number of instances
    x_values = np.linspace(start=-6.0, stop=15.0, num=x_num)
    y_values = [problem_func(x) for x in x_values] 

    #set up GP
    pset = create_primitive_set()

    creator.create("fitnessmin", base.Fitness, weights=(-1.0,))
    creator.create("individual", gp.PrimitiveTree, fitness=creator.fitnessmin, pset=pset)

    toolbox = setup_toolbox(x_values=x_values, pset=pset)

    #run GP
    #parameters in tuple (pop size, max iter, crossover rate, mutation rate,)
    gp_parameters = (1000,100,0.9,0.1)
    seeds = np.random.default_rng(seed=1).integers(low=0,high=200,size=3)

    np.seterr('raise')
    for seed in seeds:  #iterate through seeds and get result from each seed
        print('seed = ',seed)
        random.seed(int(seed))  #define seed
        best = run_gp(toolbox, gp_parameters[0], gp_parameters[1], gp_parameters[2], gp_parameters[3])
        #evaluate fitness of best tree
        best_fitness = fitness(best, x_values)
        print('best program fitness = ',best_fitness)
        print('best program depth = ',best.height)
        #draw gp 
        nodes, edges, labels = gp.graph(best)
        graph = pgv.AGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        graph.layout(prog='dot')

        #label nodes in graph chart
        for i in nodes:
            n = graph.get_node(i)
            n.attr["label"] = labels[i]

        graph.draw('tree'+str(seed)+'.png')

        best_func = toolbox.compile(expr=best)
        y_values = [best_func(x) for x in x_values] 
        plt.scatter(x=x_values,y=y_values)
        plt.show()
    