#here is all the code for the first assignment in the MCMC bootcamp. Ben & Jessie
#finished it Tuesday afternoon. Super non-technical comments by Jessie

num_chains = int(raw_input("How many chains?\t")) # allow user to request # chains...
iterations = int(raw_input("How many iterations?\t")) # ...and # iterations

import numpy as np
import math
import matplotlib.pyplot as plt

# localizing the most commonly called methods to allow it to run faster
normal = np.random.normal
uniform = np.random.uniform
exp = math.exp
pi = math.pi
cos = math.cos

def prob(mu,nu,temp):
	# summ = cos(1*pi*mu*nu)
	# for i in xrange(2,10):
	# 	summ += cos(i*pi*mu*nu)
	# return summ
	return ((16.0/(3*pi)*(exp(-(mu**2)-(9+4*mu**2+8*nu)**2)+0.5*exp(-(8*mu**2)-8*(nu-2)**2)))**(1/temp))

# we need to set up a given number of chains, spaced by an exponential growth
# factor c such that ~1.2 < c < 2
def chain_maker(num_of_chains):
	chain_temps = [1] # first chain has temp = 1
	chain_temps.append(1.5) # 2nd chain (because 1**x = 1 for all x)
	for i in xrange(1,num_of_chains): # loop starting with second chain
		chain_temps.append(chain_temps[i]**1.5)
	return chain_temps

def PTMCMC(num_of_chains,N=1000): # function that actually does the PTMCMC
	chain_temps_list = chain_maker(num_of_chains)

	print(pid)

	mus = [] # holds all chains' entire histories of mu
	nus = [] # likewise for nu

	for j in xrange(num_of_chains): # make nested lists, one for each temp chain
		mus.append([])
		nus.append([])
	
	for j in xrange(num_of_chains): # gives mu and nu initial value for each temp
		mus[j].append(normal(0,1))
		nus[j].append(normal(0,1))

	i = 0
	while i < N:
		if num_of_chains > 1:
			if (i%3) == 0: # propose swap once every 3 iterations
				# pick random chain to propose swapping with the chain immediately
				# above it.  Only swap one pair at a time, otherwise mass confusion
				# could transpire, and we could have the coldest and hottest chains
				# swapping, which could cause the temp=0 chain to jump back and forth 
				cold_chain = np.random.randint(num_of_chains-1)
				# used (num_of_chains-1) to exclude the hottest chain, which has no
				# neighbor hotter than it.  Allows all chains but the last propose
				# swaps (since each chain proposes swap with chain directly above it)

				hot_chain = cold_chain + 1
				cold_temp = chain_temps_list[cold_chain] # get temps of chains
				hot_temp = chain_temps_list[hot_chain]
				alpha_swap = uniform()
				
				H = prob(mus[cold_chain][i],nus[cold_chain][i],hot_temp)*prob(mus[hot_chain][i],nus[hot_chain][i],cold_temp)
				H /= prob(mus[cold_chain][i],nus[cold_chain][i],cold_temp)*prob(mus[hot_chain][i],nus[hot_chain][i],hot_temp)
				
				if H >= alpha_swap:
					# python allows to swap without temporary variables as follows:
					# a,b = b,a
					mus[cold_chain][i],mus[hot_chain][i] = mus[hot_chain][i],mus[cold_chain][i]
					nus[cold_chain][i],nus[hot_chain][i] = nus[hot_chain][i],nus[cold_chain][i]

		alphas = uniform(0,1,num_of_chains) # an alpha value for each chain
		# grab most recent value of mu and nu and add random # drawn from
		# a Gaussian to it
		proposed_mus = [x[i] + y for x,y in zip(mus,normal(0,.1,num_of_chains))]
		# 0.1 is arbitrary.  Picked 0.1 as std dev only because it seems to prevent
		# the temp=1 chain from jumping from one mode to the other, but is still large
		# enough to allow it to actually finish exploring a mode in a reasonable time
		proposed_nus = [x[i] + y for x,y in zip(nus,normal(0,.1,num_of_chains))]
		
		# note: zip(mu,nu,temp) gives us [(mu1,nu1,temp1),(mu2,nu2,temp2),...]
		proposed_probs = [prob(mu,nu,t) for mu,nu,t in zip(proposed_mus,proposed_nus,chain_temps_list)]
		
		# note that mu and nu should take value of each chain's nested list in the
		# mus and nus list, which is why we only need to specify the one index
		curr_probs = [prob(mu[i],nu[i],t) for mu,nu,t in zip(mus,nus,chain_temps_list)]
		
		# now compare the proposed values to the current values
		# get Hastings's ratio for each chain's proposed values
		H = [x/y for x,y in zip(proposed_probs,curr_probs)]
		
		# list of Trues and Falses so we can tell whether or not to use the proposed
		# values in the following loop
		use_proposed_vals = [(h > a) for h,a in zip(H,alphas)]

		for j in xrange(len(use_proposed_vals)):
			if use_proposed_vals[j]: # if the H > alpha for the jth chain 
				mus[j].append(proposed_mus[j])
				nus[j].append(proposed_nus[j])
			else:
				mus[j].append(mus[j][i])
				nus[j].append(nus[j][i])
		i += 1
	return mus[0],nus[0] # returns only the chain with temp = 1

#############################

mu_list,nu_list = PTMCMC(num_chains,iterations)

# runs PTMCMC again with only 1 chain, the results of which we'll plot to illuminate
# whether or not the PTMCMC actually helps
solo_mus,solo_nus = PTMCMC(1,iterations)

# compare the multi-chain PTMCMC and the single-chain MCMC
plt.scatter(solo_mus,solo_nus,color='r') # single chain plotted in red
plt.scatter(mu_list,nu_list)
plt.xlabel('mu')
plt.ylabel('nu')
plt.show()

# look at mu's progress from first iteration to last
plt.plot(range(iterations+1),mu_list)
plt.title('Mu\'s evolution')
plt.xlabel('iterations')
plt.ylabel('mu')
plt.show()

# look at nu's progress from first iteration to last
plt.plot(range(iterations+1),nu_list)
plt.title('Nu\'s evolution')
plt.xlabel('iterations')
plt.ylabel('nu')
plt.show()


