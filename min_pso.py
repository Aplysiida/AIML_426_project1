import numpy as np

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

if __name__=="__main__":
    x_values = np.linspace(start=-30, stop=30, num=20) #D = 20