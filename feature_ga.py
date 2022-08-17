import enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import mutual_info_classif

import sys
import time

import genetic_algo

"""
read dataset as a dataframe with each row being an instance + [class label] and columns = [feature names]+[class]
"""
def read_dataset(dataset_path):
    names_file = open(dataset_path[1]+dataset_path[0]+'.names', 'r')
    names_file.readline()   #skip class values for now
    columns = [line.split()[0][:-1] for line in names_file] #get names
    names_file.close()

    feature_num = len(columns)

    #get data
    data = pd.read_table(dataset_path[1]+dataset_path[0]+'.data', sep = ',',header=None)
    data.columns = columns+['class']

    #get labels
    labels = data.iloc[:,feature_num]
    labels.columns = ['class']
    #data = data.drop(labels='class',axis=1) #remove labels from data
    data.sort_values(by=data.columns[:-1].tolist(), inplace=True, ignore_index=True)   #sort data set by feature values

    #small data
    data = data.iloc[0:11,0:6]
    data['class'] = labels
    data = data.append(data.loc[1],ignore_index=True)
    data = data.append(data.loc[1],ignore_index=True)
    data = data.append(data.loc[3],ignore_index=True)

    #return (data,labels),feature_num
    return data, feature_num

"""
Wrapper fitness function
"""
def individual_wrapper_fitness(individual, classifier, data, labels, seed):
    rng = np.random.default_rng(seed)   #split at same point always to keep consistent results
    to_drop = [i for i,value in enumerate(individual) if (value == 0)]
    subset_data = data.drop(data.columns[to_drop],axis=1) #transform dataset to have only selected features
    #split dataset into training, testing
    data_train, data_test, label_train, label_test = train_test_split(subset_data, labels, test_size = 0.2, random_state = rng.integers(low=0,high=200))
    #train classifier
    #predicted = MLPClassifier(max_iter = 1000, random_state=rng.integers(low=0,high=200)).fit(data_train, label_train).predict(data_test)
    #predicted = DecisionTreeClassifier(random_state=rng.integers(low=0,high=200)).fit(data_train, label_train).predict(data_test)
    predicted = classifier.fit(data_train, label_train).predict(data_test)
    #predicted = KNeighborsClassifier(n_neighbors=5).fit(data_train, label_train).predict(data_test)
    #predicted = GaussianNB().fit(data_train, label_train).predict(data_test)
    accuracy = accuracy_score(label_test, predicted)    #calc accuracy
    return accuracy

"""
Filter Fitness function
    data = dataset with data and labels
    data_values = range of possible instances X
    label_values = range of possible classes Y
"""
def individual_filter_fitness(individual,data, x_count, y_count):
    #drop unselected features in individual
    #TODO

    total_num = len(data.index)
    #calc H(X)
    h_x = 0.0
    for x,x_num,_ in x_count:
        p_x = x_num/total_num
        h_x = h_x - (p_x * np.log2(p_x))
    print('hx = ',h_x)
    #calc H(Y)
    h_y = 0.0
    for y,y_num in y_count:
        p_y = y_num/total_num
        h_y = h_y - (p_y * np.log2(p_y))
    print('hy = ',h_y)
    #calc H(Y|X)
    h_yx = 0.0
    for x,x_num,class_num in x_count:
        p_x = x_num/total_num
        for y,y_num in y_count:
            yx_count = class_num.count(y)
            p_yx = (yx_count / total_num) / p_x
            if (p_yx > 0.0):    #if such a conditional probability exists then it will affect entropy
                h_yx = h_yx - (p_x * p_yx * np.log2(p_yx))
    print('h_yx = ',h_yx)
    
    ig = h_y - h_yx #calc info gain
    print('ig = ',ig)
    return (2.0 * ig) / (h_x + h_y)
    
    #use only selected features in measuring fitness
    #to_drop = [i for i,value in enumerate(individual) if (value == 0)]
    #subset_data = data.drop(data.columns[to_drop],axis=1).drop('class',axis=1) #transform dataset to have only selected features
    #labels = data.loc[:,'class']

    """for i in range(len(subset_data_values.index)-1):    #iterate through possible values for X
        start = subset_data_values.iloc[i].name
        end = subset_data_values.iloc[i+1].name
        instances = subset_data[start:end]  #get all instances of value x in possible values X
        x_count = len(instances)    #number of instances with value x in data
        p_x = x_count/len(subset_data)  #probability(X = x)
        h_x = h_x - (p_x * np.log2(p_x))    #calc entropy for X

        for label in label_values:  #iterate through possible values for Y
            xy_count = np.count_nonzero(instances.loc[:,'class'] == label)     #number of instances that intersect x and y sets
            p_yx = (xy_count/len(subset_data)) / (x_count/len(subset_data)) #calc conditional probability(Y|X)
            to_add = (p_x * p_yx * np.log2(p_yx)) if (p_yx > 0) else 0  #calculate conditional entropy for each value
            h_yx = h_yx - (to_add)  #calc conditional entropy H(Y|X)
    
    for label_count in y_count: #calc probabilities and entropies of Y
        p_y = label_count/len(subset_data)
        h_y = h_y - (p_y * np.log2(p_y))
        
    ig_yx = h_y - h_yx  #calc information gain
    """

"""
Draw convergence curves for each seed and iteration
"""
def draw_convergence_curves(x_values, y_values, dataset_name, seeds, mean, std,optimal_value=None):
    fig = plt.figure(constrained_layout=True)
    fig.suptitle(dataset_name+' Mean: '+str(mean)+' Standard Deviation: '+str(std))
    subfigs = fig.subfigures(nrows=len(seeds),ncols=1)

    for i,seed in enumerate(seeds):
        subfigs[i].suptitle('Seed: '+str(seed))
        axis = subfigs[i].subplots(nrows=1, ncols=1)
        axis.yaxis.grid(True)
        if(optimal_value != None):
            axis.axhline(optimal_value, color='green', linewidth=1.0)
        axis.plot(x_values[i], y_values[i],c='red')

    plt.show()

def run_seeds(dataset, dataset_parameter, seeds, fitness_func, performance_func):
    times_taken = []    #time taken by GA for each seed to reach solution
    performance = []    #accuracy of selected feature subset using MLP classifier
    for seed in seeds:
        print('seed = ',seed)
        rng = np.random.default_rng(seed)

        item_length = dataset[1]
        
        start_time = time.time()
        ga = genetic_algo.GA(rng, dataset, item_length, fitness_func)
        _,_,_,best = ga.GA_solution(
                pop_size = dataset_parameter[0],
                max_iter = dataset_parameter[1],
                elitism_rate= dataset_parameter[2],
                crossover_rate= dataset_parameter[3],
                mutation_rate= dataset_parameter[4]
        )
        end_time = time.time()
        time_record = (end_time-start_time)
        times_taken.append(time_record)

        #get accuracy of feature subset
        accuracy = performance_func(best)
        performance.append(accuracy)
        print('time = ',time_record,' performance = ',accuracy)

    return np.mean(times_taken), np.std(times_taken), np.mean(performance), np.std(performance)


if __name__=="__main__":
    data_folder_paths = (('wbcd',sys.argv[1]), ('sonar',sys.argv[2])) #get tuple (dataset name, path to folder)

    dataset_names = ['wbcd','sonar']
    datasets = []
    datasets.append(read_dataset(data_folder_paths[0]))
    datasets.append(read_dataset(data_folder_paths[1]))
    
    dataset_parameters = [] #stores tuples as (pop size, max iterations, elitism rate, crossover rate, mutation rate)
    dataset_parameters.append((10,20,0.03,1.0,0.3))
    dataset_parameters.append((40,100,0.03,1.0,0.3))

    seed_rng = np.random.default_rng(123)
    seeds = seed_rng.integers(low=0,high=200,size=5)

    #warm up JIT to get accurate times

    dataset = datasets[0]
    dataset_parameter = dataset_parameters[0]
    data= dataset[0]

    individual = np.random.default_rng().integers(low=0,high=2,size=dataset[1])
    #for y (label, num of label occurences in dataset)
    label_count = list(zip(data.loc[:,'class'].unique(), data.loc[:,'class'].value_counts(sort=False)))
    #for x (feature combination, num of occurences in dataset)
    instance_count = list(zip(list(data.drop_duplicates(subset=data.columns[:-1]).itertuples(index=False,name=None)), 
                   list(data.iloc[:,:-1].value_counts(sort=False))
                   ))
    class_list = data.groupby(by=data.columns[:-1].tolist()).agg(lambda x: x.values.tolist())   #group the classes of the same instances together

    instance_count = list(zip(list(data.drop_duplicates(subset=data.columns[:-1]).itertuples(index=False,name=None)), #get instance
                   list(data.iloc[:,:-1].value_counts(sort=False)), #get number of occurances of this instance
                   [class_count[1][0] for class_count in class_list.iterrows()]  #get labels for the instances
                   ))

    print(data)
    print('----')
    individual_filter_fitness(individual,data,instance_count, label_count)

    """
    #filterGA
    for i,dataset in enumerate(datasets):
        dataset_parameter = dataset_parameters[i]
        print('at dataset ',dataset_names[i],':')


    #wrapperGA
    for i,dataset in enumerate(datasets):
        dataset_parameter = dataset_parameters[i]
        print('at dataset ',dataset_names[i],':')

        data = dataset[0].drop('class',axis=1)
        labels = dataset[0].loc[:,'class']

        seed = 1
        fitness_func = lambda x : individual_wrapper_fitness(x,
                                                             KNeighborsClassifier(n_neighbors=5),
                                                             data,
                                                             labels,
                                                             seed
                                                            )
        performance_func = lambda x : individual_wrapper_fitness(x,
                                                                 MLPClassifier(max_iter = 1000, random_state=seed),
                                                                 data,
                                                                 labels,
                                                                 seed
                                                                )

        avg_time, std_time, avg_performance, std_performance = run_seeds(dataset, dataset_parameter, seeds, fitness_func, performance_func)
        print('average time = ',avg_time,
              ' standard deviation time = ',std_time)
        print('average performance = ',avg_performance,
              ' standard deviation performance = ',std_performance)
    """
    """
    for i,dataset in enumerate(datasets):
        dataset_parameter = dataset_parameters[i]
        print('at dataset ',dataset_names[i],':')

        GA_output = []  #GA best solution from each seed
        x_values = [] #iterations range for each GA 
        y_values = [] #average of 5 best individuals each iteration

        data = dataset[0]

        #discretise feature instances/ get images for X and Y
        data_values = data.drop_duplicates(subset = data.columns[:-1]).drop('class',axis=1)  #unique feature values
        label_values = data.loc[:,'class'].drop_duplicates() #unique class values
        fitness_func = lambda x : individual_filter_fitness(x,data,data_values,label_values)

        y,x,best = run_seeds(dataset, dataset_parameter, seeds, fitness_func)

        GA_output.append(best)
        x_values.append(range(x))
        y_values.append(y)

    mean = np.mean(GA_output)
    std = np.std(GA_output)
    print('mean = ',mean)
    print('standard deviation = ',std)
    draw_convergence_curves(x_values, y_values, dataset_names[i], seeds, mean, std)
    """

    """
    #filter
    for i,dataset in enumerate(datasets):
        dataset_parameter = dataset_parameters[i]
        print('at dataset ',dataset_names[i],':')

        GA_output = []  #GA best solution from each seed
        x_values = [] #iterations range for each GA 
        y_values = [] #average of 5 best individuals each iteration

        data = dataset[0]

        #discretise feature instances/ get images for X and Y
        data_values = data.drop_duplicates(subset = data.columns[:-1]).drop('class',axis=1)  #unique feature values
        label_values = data.loc[:,'class'].drop_duplicates() #unique class values
        fitness_func = lambda x : individual_filter_fitness(x,data,data_values,label_values)

        for seed in seeds:
            print('seed = ',seed)
            rng = np.random.default_rng(seed)

            item_length = dataset[1]
            FilterGA = genetic_algo.GA(rng, dataset, item_length, fitness_func)
            y,x,best = FilterGA.GA_solution(
                pop_size = dataset_parameter[0],
                max_iter = dataset_parameter[1],
                elitism_rate= dataset_parameter[2],
                crossover_rate= dataset_parameter[3],
                mutation_rate= dataset_parameter[4]
            )

            GA_output.append(best)
            x_values.append(range(x))
            y_values.append(y)

        mean = np.mean(GA_output)
        std = np.std(GA_output)
        print('mean = ',mean)
        print('standard deviation = ',std)
        draw_convergence_curves(x_values, y_values, dataset_names[i], seeds, mean, std)
        """