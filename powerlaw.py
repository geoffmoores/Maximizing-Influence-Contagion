import numpy as np
import scipy
from scipy.special import zeta
import random as r

## Algorithm as in Clauset et al "POWER-LAW DISTRIBUTIONS IN EMPIRICAL DATA"

def test():
  
  a = powerlaw_degrees(2.4,1,1000,5000)
  print(np.sort(a))
  print(fit(a))
  #goodness_of_fit(1, 3, 3, 50, a, M=5)

# IN: data - list(int)
#     (K_min) - minimum value K to begin the powerlaw fit at
#     (get_goodness_of_fit) - Boolean, if true calculates and returns Goodness of Fit 
#                             with powerlaw parameters
#     (get_var) - Boolean, if true returns the variance of the fitted powerlaw distribution
def fit(data,K_min=1,get_goodness_of_fit=True,get_var=True):
  
  d_min = min(data)
  d_max = max(data)
  
  lb = max(d_min,K_min)
  
  e_cdf = empirical_cdf(data)
  #  [0:gamma, 1:var, 2:pval, 3:kmin, 4:D, 5:N]
  fit = [0, 0, 0, 0, np.Inf, 0]
  # print(d_min,d_max,len(e_cdf),e_cdf[0:15])
  # print(e_cdf[-15:])
  
  for km in range(lb,d_max):
  
    tail = data[data>=km]
    N = len(tail)
    gamma = 1 + N*sum(map(lambda x: np.log(x/(km-.5)),tail))**-1
    if zeta(gamma,km) < .000000001:
      break

    e_cdf = empirical_cdf(tail,km)
    cdf = np.array([1-zeta(gamma,k+1)/zeta(gamma,km) for k in range(km,d_max+1)])
    # print(km,":::::::::::::::::::::::")
    # print(np.around(e_cdf[0:10],2))
    # print(np.around(cdf[0:10],2))
    # print(np.around(abs(cdf-e_cdf)[0:10],2))
    D_km = max(abs(cdf-e_cdf))  #[km-d_min:]
    if D_km < fit[4]:
      fit[4] = D_km
      fit[0] = gamma
      fit[3] = km
      fit[5] = N
      
  if fit[4] != np.Inf:
    if get_var:
      fit[1] = powerlaw_gamma_MLE_variance(fit[0],fit[3],fit[5])
    if get_goodness_of_fit:
      fit[2] = goodness_of_fit(fit[4],fit[0],fit[3],d_max,data)
  return fit
  
  
def goodness_of_fit(D_real, gamma, kmin, kmax, data, M=500):

  D_syns = []
  top = np.array([k**-gamma for k in range(kmin,kmax+1)])
  bottom = np.array([zeta(gamma,kmin) for k in range(kmin,kmax+1)])
  if np.sum(np.isnan(top)) > 0:
    print("NAN in top:",k,gamma,kmin,kmax)
    print(top)
  if np.sum(np.isnan(bottom)) > 0:
    print("NAN in Bottom:",k,gamma,kmin,kmax)
    print(bottom)
  bottom = np.where(np.isnan(bottom), 1.0, bottom)
  bottom = np.where(bottom == 0, 1.0, bottom) # this is safe because top 
  pdf = top / bottom
  pdf = pdf / np.sum(pdf)
  cutoff_data = list(data[data < kmin])
  num_cutoff = len(cutoff_data)
  T = len(data)
  count = 0
  
  for i in range(M):
    cutoff_samples = np.random.binomial(T,num_cutoff/T)
    powerlaw_samples = T-cutoff_samples
    powerlaw_sample = list(np.random.choice(range(kmin,kmax+1),size=powerlaw_samples,p=pdf))
    cutoff_sample = r.choices(cutoff_data,k=cutoff_samples)
    deg_sample = powerlaw_sample + cutoff_sample
    deg_fit = fit(np.array(deg_sample),get_goodness_of_fit=False,get_var=False)
    R_syn = deg_fit[4]
    # print(i,"+++++++++++++++++")
    # deg_sample.sort()
    # print(deg_sample)
    # print(deg_fit)
    if R_syn > D_real and R_syn != np.Inf:
      count += 1
    D_syns.append(R_syn)
    #print()
    
  p_val = count / M
  return p_val
    
  
  # eq 3.6 Clauset '09: Powerlaw
def powerlaw_gamma_MLE_variance(gamma,kmin,N):
  zeta_g_kmin = zeta(gamma,kmin)
  if zeta(gamma,kmin) == 0:
    return np.nan
    
  zeta_ratio_1 = hurwitz_2nd_der_appx(gamma,kmin)/zeta_g_kmin
  zeta_ratio_2 = (hurwitz_1st_der_appx(gamma,kmin)/zeta_g_kmin)**2
  var = N*(zeta_ratio_1-zeta_ratio_2)
  if var > 0:
    var = 1/(np.sqrt(var))
    return var
  else:
    return np.nan
    
def hurwitz_1st_der_appx(gamma,q,max_its=500,rel_error=.0001):
  sum = 0
  for k in range(q,max_its):
    if k**gamma == 0:
      break
    inc = -np.log(k)/k**gamma
    sum += inc
    if sum != 0 and inc/sum < rel_error:
      break
  return sum  
  
def hurwitz_2nd_der_appx(gamma,q,max_its=500,rel_error=.0001):
  sum = 0
  for k in range(q,max_its):
    if k**gamma == 0:
      break
    inc = np.log(k)**2/k**gamma
    sum += inc
    if sum != 0 and inc/sum < rel_error:
      break
  return sum
  
def empirical_cdf(data,start=None):
  N = len(data)
  uniques,counts = np.unique(data, return_counts=True)
  counts = counts / N
  
  dmax = max(uniques)
  dmin = min(uniques)
  if start != None: # allows for starting the array before the minimum in the data
    dmin = min(dmin,start)
    
  pdf = np.zeros(dmax-dmin+1)

  for k in range(len(uniques)):
    pdf[uniques[k]-dmin] = counts[k]
  return np.cumsum(pdf)
  
def empirical_pdf(data,start=None):
  N = len(data)
  uniques,counts = np.unique(data, return_counts=True)
  counts = counts / N
  
  dmax = max(uniques)
  dmin = min(uniques)
  if start != None: # allows for starting the array before the minimum in the data
    dmin = min(dmin,start)
    
  pdf = {}

  for k in range(len(uniques)):
    pdf[uniques[k]-dmin] = counts[k]
  return pdf
  
def discrete_var(data):
  mu_data = np.average(data)
  pdf = empirical_pdf(data)
  var = 0
  for x_val in pdf:
    var += (pdf[x_val]*x_val)**2
  var -= mu_data**2
  
  return var
  
  
# IN: gamma - float in [2,3] preferably, degree exponent of distribution
#     kmin - minimum degree generated in sequence
#     kmax - maximum degree generated
#     N - number of samples
# OUT: 1D np.array length N, powerlaw of degrees according to parameters given
def powerlaw_degrees(gamma,kmin,kmax,N):
  normalizer = sum([x**-gamma for x in range(kmin,kmax+1)])
  normalize = 1/normalizer
  pdf = np.array([x**(-gamma)/normalizer for  x in range(kmin,kmax+1)])
  #print(pdf)
  return np.random.choice(range(kmin,kmax+1),size=N,p=pdf)
  
if __name__ == "__main__":  
  test()