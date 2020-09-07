from scipy.stats import chisquare
import scipy.special
import numpy as np
import math
from scipy.special import gamma


def test():
  
  N = 50
  a = np.random.binomial(N-1,.6,N)
  print(a)
  print(fit(a))
  print()
  print()
  
  N = 150
  a = np.random.randint(100,size=N)
  print(a)
  print(fit(a))  

# IN: data - 1D np.array or list of data 
# OUT: mu - float, expected average value of fit Poisson distribution
#      goodness of fit - float in [0,1], p-value of poisson distribution fit with observed data
#                        according to chi-square test.
def fit(data):
  
  deg,counts = np.unique(data,return_counts=True)
  #print(deg,counts)
  deg = list(deg)
  kmin = min(deg)
  kmax = max(deg)
  
  avg_deg = np.average(data)
  N = len(data)
  mu = avg_deg / N
  #print("Mu^:",mu)
  
  poisson, binom = [False,False]
  if N > 100:
    calc_expected = E_poisson
  else:
    calc_expected = E_binomial
  
  expected = np.zeros(kmax+1)
  observed = np.zeros(kmax+1)
  
  for x in range(kmax+1):
    expected[x] = calc_expected(mu,x,N)
    if x in deg:
      observed[x] = counts[deg.index(x)]
  # print("Uncollapsed",round(np.sum(expected),2),np.sum(observed))
  # print(np.around(expected,1))
  # print(observed)
  e_large = expected >= 3
  e_small = expected < 3
  
  mop_up_exp = np.sum(expected[e_small])
  mop_up_obs = np.sum(observed[e_small])
  
  expected = np.concatenate((expected[e_large],np.array([mop_up_exp])))
  observed = np.concatenate((observed[e_large],np.array([mop_up_obs])))
  
  # print("Collapsed")
  # print(np.around(expected,2))
  # print(observed)
  
  dof = len(expected) - 2
    
  gof = chisquare(observed,f_exp=expected,ddof=1)
  
  return mu, gof[1]
    

def E_poisson(mu,x,N):
  avg_deg = mu*N
  log_pr = x*np.log(avg_deg) - avg_deg - math.lgamma(x+1)
  return np.exp(log_pr)*N
    
    
def E_binomial(mu,x,N):
  return scipy.special.binom(N,x)*mu**x*(1-mu)**(N-x)*N

if __name__ == "__main__":  
  test()