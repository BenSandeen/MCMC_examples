#here is all the code for the first assignment in the MCMC bootcamp. Ben & Jessie
#finished it Tuesday afternoon. Super non-technical comments by Jessie

import time
import numpy as np
import math
try:
	import matplotlib.pyplot as plt
except:
	None
from multiprocessing import *

totalTime = time.clock()
# set these functions to variables so we can call them much more quickly (and with
# fewer characters) as such: normal(mean,std_dev), which is equivalent to (but
# faster than) doing np.random.normal(mean,std_dev) 
normal = np.random.normal
uniform = np.random.uniform
exp = math.exp
pi = math.pi
#defining the Normal Gaussian Distribution so we can use that function in our code in the future.
def prob(x, sigma, mu):
	p = (1/((2*pi)**0.5*sigma))*exp(-((x-mu)**2)/(2*sigma**2))
	return p

#in the gaussian, mu is the mean and sigma is the standard deviation. we give them set values here.
mu = 3
sigma = 1.4

#in part a of the first assignment we are told to try 4 different jump size scalings. This defines the first of them.
# ayyy = 1.0
# ayyy = 0.6
ayyy = 2.2

#choosing a random starting point for x[0]
#computing the gaussian at that x[0]
# prob(x[0], sigma, mu)
#running the code for 10000 iterations x_values_
N = 100
for i in xrange(1):
	x = []
	x_appender = x.append
	x_appender(normal(mu, sigma))

	accepted = 0
	i = 0
	while i < N:
		#defines y as the current x plus a draw from the gaussian. 
		#(technically ayyy should be squared. put it in yourself and see how insane the results get.)
		y = x[i] + normal(0, ayyy)#*ayyy)#sigma)
		#H is Hasting's ratio (checks if the new value we've generated is more probable than the prior value)
		H = prob(y, sigma, mu)/prob(x[i], sigma, mu)
		# H = prob(y, ayyy, mu)/prob(x[i], ayyy, mu)
		#alpha is drawn from a uniform random distribution
		alpha = uniform()
		#so if the hastings ratio is better than a totally random choice, we keep the new value. If not, we leave x the same.
		if H >= alpha:
			x_appender(y)
			accepted+=1
		else:
			x_appender(x[i])
		i = i + 1

	plt.plot(x)
	plt.scatter(range(len(x)),x,marker='+')
	plt.xlim([0,len(x)])
	plt.xlabel('iterations')
	plt.ylabel('value of x')
	plt.show()

#######################################################

# Start of part 2 stuff
# We have just created our data in x; now we must try to reverse-engineer the
# parameters of the distribution which generated it.  Since Bayesian probabilities
# tell us that the posterior distribution function is equal to the likelihood times
# the prior distribution function, we'll use that.  But since we know nothing of
# the prior distribution function function, we'll use a uniform probability
# distribution (ie: all values of mu and sigma are equally probable).  But since
# only the relative probabilities are important, we can simply ignore the constant
# prior, so the posterior distribution function is equal to the likelihood

logger = math.log
def likelihood(x_list,sigma,mean):
	likeliness = 0
	for j in x_list:
		# if (prob(j,sigma,mean) <= 0):
			# print('prob(j,sigma,mean): ',prob(j,sigma,mean))
			# print('j: ',j)
			# print('sigma: ',sigma)
			# print('mean: ', mean)
		try:
			likeliness += logger(prob(j,sigma,mean))
		except: # if the probability is <= 0, the log of which is an error
			# likeliness = -100000
			likeliness -= 100000
	return likeliness

i = 0
mus = []
sigmas = []

# here are a bunch of variables and shortcuts for appending items to the lists.  The
# current code doesn't use all of them, but I didn't want to waste my time deleting
# the ones we don't use
default_mu = 0
default_sigma = 3
accepted_mus = 0
accepted_sigmas = 0
accepted_mus_and_sigmas = 0
mu_appender = mus.append
sigma_appender = sigmas.append
most_likely = (0,0)
best_hastings = -100000
hastings_list = []
guess_likelihoods = []
accepted_likelihoods = [] # used to color scatterplot of mus vs. sigmas by likelihood
hastings_mu_list = []
hastings_sigma_list = []
likelihoods = []
likelihoods_appender = likelihoods.append
hastings_appender = hastings_list.append
guess_likelihoods_appender = guess_likelihoods.append
accepted_likelihoods_appender = accepted_likelihoods.append
hastings_mu_list_appender = hastings_mu_list.append
hastings_sigma_list_appender = hastings_sigma_list.append

# get random starting points for mu and sigma
mu_appender(normal(0,1))
sigma_appender(normal(1,1))#uniform(-.3,2))

while i < (N*100):#N:#(N*10):
	mu_guess = mus[i] + normal(0,.03)#ayyy**(.5))#.7)#ayyy) #uniform(-1,1)
	sigma_guess = sigmas[i] + uniform(-0.1,0.1)#normal(0,.09)#ayyy**(.5))#.7)#uniform(-1,1)#uniform(-1,1)#normal(0.1,ayyy)
	# sigma_guess = sigmas[i] + uniform(-.5,2)
	if sigma_guess < 0.1:
		sigma_guess = 0.1
	
	# stuff for simply looking at the probability of both the random mu AND the
	# random sigma simultaneously.

	guess_likelihood = likelihood(x,sigma_guess,mu_guess)
	accepted_likelihood = likelihood(x,sigmas[i],mus[i])
	# guess_likelihoods_appender(guess_likelihood)
	# accepted_likelihoods_appender(accepted_likelihood)

	# hastings = guess_likelihood / accepted_likelihood
	hastings = guess_likelihood - accepted_likelihood
	# print('hastings: ',hastings)
	# hastings = likelihood(x,sigma_guess,mu_guess)/likelihood(x,sigmas[i],mus[i])
	# if hastings < best_hastings:
	if  hastings > best_hastings:
		best_hastings = hastings
		most_likely = (mu_guess,sigma_guess)

	# print('hastings: ',hastings)
	# alpha = uniform()#(0,2)
	# alpha = uniform(.98,1.05)
	# alpha = uniform(.95,1.1)
	alpha = uniform(-1,0.5)#1)
	# if hastings <= alpha:
	if hastings >= alpha: # the hastings difference (no longer a ratio) will be
	# positive if the proposed value is more likely, and negative if not
		mu_appender(mu_guess)
		sigma_appender(sigma_guess)
		accepted_mus_and_sigmas += 1
		likelihoods_appender(guess_likelihood)
	else:
		mu_appender(mus[i])
		sigma_appender(sigmas[i])
		likelihoods_appender(accepted_likelihood)

	# print('hastings: ', hastings)
######
	hastings_appender(hastings)
	i += 1

# jump_lengths = np.array(mus[:-1]) - np.array(mus[1:])
# print('average jump_lengths: ', np.average(jump_lengths))

# printing out stuff to verify whether or not things seem to be working
# print('last mu: ',mus[-1])
# print('last sigma: ',sigmas[-1])
# print('np.average(x): ',np.average(x))
# print('np.std(x): ',np.std(x))
# print('np.average(mus): ',np.average(mus))
# print('np.average(sigmas): ',np.average(sigmas))
# print('most likely (mu,sigma) pair: ',most_likely)
# # print('accepted_mus: ',accepted_mus)
# # print('accepted_sigmas: ',accepted_sigmas)
# print('accepted_mus_and_sigmas: ',accepted_mus_and_sigmas)

# print(len(set(mus[-10:])))
# print(len(set(sigmas[-10:])))

# cut out burn-in by starting after 1st bunch of the values, ideally letting us
# focus only on sections where parameters have already approached relatively
# stable values

# thin out the lists, so that we hopefully have effectively independent samples
# also remove first chunk of samples to ignore burn-in time (time spent navigating
# from initial random value to the proper values) 
mus = [mus[i] for i in xrange(int(len(mus)/10),len(mus),5)]
sigmas = [sigmas[i] for i in xrange(int(len(sigmas)/10),len(sigmas),5)]
# accepted_likelihoods = [accepted_likelihoods[i] for i in xrange(int((len(accepted_likelihoods)/10)-1),len(accepted_likelihoods),5)]
likelihoods = [likelihoods[i] for i in xrange(int((len(likelihoods)/10)-1),len(likelihoods),5)]

# print('mus: ',mus)
# print('sigmas: ',sigmas)
# print('len(mus): ',len(mus))
# print('len(sigmas): ', len(sigmas))
# print('len(accepted_likelihoods): ',len(accepted_likelihoods))
print("run time: ",time.clock() - totalTime)

# plt.plot(mus)
plt.scatter(range(len(mus)),mus,marker='+')
plt.xlim([0,len(mus)])
plt.title('mu')
plt.xlabel('iterations')
plt.ylabel('value of mu')
plt.show()

# plt.plot(sigmas)
plt.scatter(range(len(sigmas)),sigmas,marker='+')
plt.xlim([0,len(sigmas)])
plt.title('sigma')
plt.xlabel('iterations')
plt.ylabel('value of sigma')
plt.show()

# plt.plot(hastings_list,color='r')
# plt.plot(guess_likelihoods,color='b')
# plt.plot(accepted_likelihoods,color='g')
# plt.show()

# plt.hist(list(set(hastings_list)),bins=200)
# # plt.axis([-10,10,0,100])
# plt.show()

# plt.hist(accepted_likelihoods,bins=20)
# plt.show()
# print('len(likelihoods): ', len(likelihoods))
colors = np.array(likelihoods)#accepted_likelihoods)
plt.scatter(mus,sigmas,c=(colors),marker='+')#accepted_likelihoods)
plt.title('mu vs. sigma')
plt.xlabel('mu')
plt.ylabel('sigma')
plt.show()


