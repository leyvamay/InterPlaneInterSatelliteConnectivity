import numpy as np
import scipy.special
import math
import matplotlib.pyplot as plt
from config import *
from mpl_toolkits.mplot3d import Axes3D





No=k*Ts*B		# Noise power
Ndb=10*np.log10(No)	# Noise power [dB]
minSNR=2**(Rmin/B)-1	# Minimum SNR to achieve Rmin
minSNRdb=10*np.log10(minSNR)	# Minimum SNR to achieve Rmin [dB]
print('Min SNR [dB]=' +repr(minSNRdb))
SNRmargin=0		# SNR above the minimum

#plt.figure()
slant_range= np.linspace(10000,6000000,1000)
txt= np.zeros((len(slant_range),4))
txt[:,0]=slant_range/1e3
for n in range(len(f)):
	#print(repr(10*np.log10(k*Ts*B)))
	loss = 20*np.log10((4 * math.pi * slant_range * f[n]/c)) #+ 	10*np.log10(k*Ts*B)
	txt[:,n+1] = loss
	#FSPL = 20*np.log10(slant_range) + 20*np.log10(f[n]) - 147.55
	#plt.plot(slant_range, loss, '-x')
	#plt.plot(slant_range, FSPL)



# Maximum FSPL as a function of P, h, and N
Pmin=5 # Minimum number of orbital planes
Pmax=12 # Maximum number of orbital planes
hmin=600e3
delta_h=10e3
theta=[math.pi/2, math.pi/2+math.pi/N_p]
lmax=np.zeros(Pmax-Pmin+1, dtype=np.float128)
lmax_mix=np.zeros(Pmax-Pmin+1, dtype=np.float128)
maxL=np.zeros((Pmax-Pmin+1,len(f)))
maxL_mix=np.zeros((Pmax-Pmin+1,len(f)))
minEIRPdb=np.zeros((Pmax-Pmin+1,len(f)))
minEIRPdb_mix=np.zeros((Pmax-Pmin+1,len(f)))
indx=0
for P in range(Pmin,Pmax+1):
	eps=math.pi/P;
	h = np.array([(P-2)*delta_h, (P-1)*delta_h])+hmin
	lmax[indx] = math.sqrt(((h[0]+Re)**2)+((h[1]+Re)**2) - 2*(h[0]+Re)*(h[1]+Re)*(math.cos(eps)*math.sin(theta[0])*math.sin(theta[1])))

#	Mixed geometry
	n_thet=1001
	thet=np.linspace(0,math.pi/2,n_thet);
	l_mix=np.zeros((n_thet,2));
	eps=math.pi/(P-1);
	h = np.array([(P-3)*delta_h,(P-2)*delta_h, (P-1)*delta_h])+hmin
	for th in range(n_thet):
		l_mix[th,0]=math.sqrt(((h[0]+Re)**2)+((h[1]+Re)**2) - 2*(h[0]+Re)*(h[1]+Re)*(math.cos(thet[th])*math.cos(thet[th]+math.pi/N_p)+math.cos(eps)*math.sin(thet[th])*math.sin(thet[th]+math.pi/N_p)));

		l_mix[th,1]=math.sqrt(((h[1]+Re)**2)+((h[2]+Re)**2) - 2*(h[1]+Re)*(h[2]+Re)*(math.cos(thet[th])*math.cos(math.pi/2)+math.cos(math.pi/N_p)*math.sin(thet[th])*math.sin(math.pi/2)));
	lmax_ep = math.sqrt(((h[1]+Re)**2)+((h[1]+Re)**2) - 2*(h[1]+Re)*(h[1]+Re)*(math.cos(eps/2)*math.sin(theta[0])*math.sin(theta[1])))
	lmax_mix[indx]=max(np.ndarray.max(np.ndarray.min(l_mix,axis=1)),lmax_ep)
	
	for n in range(len(f)):
		maxL[indx,n] = 20*np.log10((4 * math.pi * lmax[indx] * f[n]/c))
		maxL_mix[indx,n] = 20*np.log10((4 * math.pi * lmax_mix[indx] * f[n]/c))
		minEIRPdb[indx,n] = minSNRdb+maxL[indx,n]+Ndb+SNRmargin
		minEIRPdb_mix[indx,n] = minSNRdb+maxL_mix[indx,n]+Ndb+SNRmargin
	indx+=1
minEIRP=10**(minEIRPdb/10)
print('Max slant range =' + repr(lmax))
print('Max FSPL= ' +repr(maxL))
fname='InputParams/minMCL_Np_'+repr(N_p)+'.txt'
np.savetxt(fname,maxL)
print('Minimum EIRP [dB]= ' +repr(minEIRPdb))
print('Minimum EIRP [W]= ' +repr(minEIRP))
fname='InputParams/minEIRP_Ts_'+repr(round(Ts))+'_Np_'+repr(N_p)+'.txt'
np.savetxt(fname,minEIRP)



