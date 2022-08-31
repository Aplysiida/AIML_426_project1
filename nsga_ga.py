import pandas as pd

import sys

"""
Parses dataset into tuple in format (data, labels, number of features)
    dataset_path = (name of file, path to folder, file format)
"""
def read_dataset(dataset_path):
    filepath = dataset_path[1]+dataset_path[0]+'.'+dataset_path[2]
    data = pd.read_table(filepath,sep='[\s,]+',header=None)  #separate based on space or comma
    #separate data and labels
    labels = data.iloc[:,-1]
    labels.columns = ['class']
    data = data.drop(data.columns[-1],axis=1)

    feature_num = len(data.columns)
    return data, labels, feature_num


if __name__=="__main__":
    #store paths as (name of file, path to folder, file format)
    datasets_paths = [('vehicle',sys.argv[1], 'dat'),('clean1',sys.argv[2], 'data')]
    datasets = [read_dataset(dataset_path) for dataset_path in datasets_paths]

    #parameters used for each dataset
    dataset_parameters = [(10,20,0.03,1.0,0.3), (10,20,0.03,1.0,0.3)]

    #run NSGA
    print('running NSGA-II')
    