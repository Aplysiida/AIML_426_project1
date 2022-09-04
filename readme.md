AIML426 Project 1 Readme

Libraries used by this project:
    Numpy
    Pandas
    Deap
    MatPlotLib
    Seaborn
    PyGraphViz
        For PyGraphviz to work on Windows, install using the instuctions stated here:
            https://pygraphviz.github.io/documentation/stable/install.html
        Only difference from using above instructions is to install Graphviz 5.0.1 instead of Graphviz 2.46.0. 
        Graphviz can be downloaded from:
            https://graphviz.org/download/

Each py file is associated with a part in the assignment:
    knapsack_ga.py  - Part 1
    feature_ga.py   - Part 2
    nsga_ga.py      - Part 3
    symbol_gp.py    - Part 4
    min_pso.py      - Part 5
knapsack_ga.py, feature_ga.py and nsga_ga.py all need arguments which are the folder paths to the data files to read.
The arguments used are:
    knapsack_ga.py "path/to/knapsack-data/folder"
    feature_ga.py "path/to/wbcd/folder" "path/to/sonar/folder"
    nsga_ga.py "path/to/vehicle/folder" "path/to/musk/folder"
symbol_gp.py and min_pso.py takes in no arguments.

The programs will output their fitness in the terminal and output their charts in the matplotlib window. For symbol_gp.py it outputs its GP tree charts into png files in the same folder the py file is located in.