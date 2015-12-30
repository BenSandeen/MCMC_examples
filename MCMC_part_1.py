import math
import numpy as numpy
pi = math.pi
sqrt = math.sqrt
e = math.exp(1)

x = []
y = 0
# y = []
x_appender = x.append
# y_appender = y.append

# def func(x):
# 	return x

# def proposed_dist(prev_x,mean,std_dev):
	# return prev_x

def probability(val, mean, std_dev):
	''''''
	p = 1/(sqrt(2*pi)*std_dev)*pow(e,(-1*((val-mean)**2)/(2*(std_dev**2))))
	return p

for i in range(10):
	print(probability(i,5,1))

# def jump(mean,std_dev):
# 	''''''

i = 1
x[0]
while i < num_iterations:
	x_appender(np.random.normal(mu,sigma))
	y = x[num_iterations] + 

	num_iterations++


