#################################################
#						#
#	Coded by Israel Leyva at CNT AAU	#
#						#
#################################################

import numpy as np
import scipy.special
import sys
import math
import matplotlib.pyplot as plt
import time
import random
from config import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linear_sum_assignment

random.seed(2)
np.random.seed(2)
np.set_printoptions(precision=3)


gains = [1, 219.58, 494.08] # Gains for parabolic antennas 


carrier_index = 0	# Select the carrier 
f=f[carrier_index]

Pref=7

starttime=time.time()

N = N_p*P	 	# Total number of satellites
n_experiments = int(1e3)	# Number of observations within the simulation period
n_rotations = 5	# Number of rotations of the lowest orbital plane
n_disp=0	# Displacement of rotation
fname_minMCL ='InputParams/minMCL_Np_'+repr(N_p)+'.txt'
fname_EIRP ='InputParams/minEIRP_Ts_'+repr(round(Ts))+'_Np_'+repr(N_p)+'.txt'
try: 
    minMPL = np.loadtxt(fname_minMCL)
except OSError:
    print("Could not find file" + fname_minMCL + ", which gives the maximum communication range. Make sure that the script Communication_Range.py has been run with the same communication parameters and that the results are saved in InputParams folder")
    sys.exit()

try: 
    minEIRP = np.loadtxt(fname_EIRP)
except OSError:
    print("Could not find file" + fname_EIRP + ", which gives the maximum communication range. Make sure that the script Communication_Range.py has been run with the same communication parameters and that the results are saved in InputParams folder")
    sys.exit()
MPL=minMPL[Pref-5,carrier_index]
EIRP=minEIRP[Pref-5,carrier_index]

rnp=(k*Ts*B)/EIRP	# Noise to signal power ratio

num_transceivers = 2
cross_seam_implementation=False	# Allow for cross-seam ISLs
FDMA=True			# Select between none, FDMA  or CDMA

Minimum_altitude = 600e3 # Altitude of the lowest orbital plane in meters
Delta_altitude = 10e3 # Altitude of the highest orbital plane in meters
lamda = 1 ### Level of self-interference. Set 0 for no self-interference and 1 for full self-interference
resources=[1]	# List with the number of resources; choose, for example: list(range(1,16+1))#[1,2,4,8,16]#[1]#list(range(1,30+1))
r_allocation=['matching', 'rr', 'random']
ra_index =0	# Choose 0 for matching, 1 for rr and 2 for random
pre_processing=1 # Choose 0 to exclude pre-processing and 1 to include
matching_period = 30	# Matching period in seconds

cdfs=1			#Set to 1 to perform GEO matching




class orbital_plane:
	def  __init__(self,altitude, inclination, period, n_sat):
		self.altitude = altitude
		self.inclination = inclination
		self.period = period
		self.n_sat = n_sat
		self.v = 2*math.pi * (altitude + Re) / T

	def  __repr__(self):
		return '\n altitude= {}, inclination= {}, period= {}, number of satellites= {}, satellite speed= {}'.format(
	self.altitude,
	self.inclination,
	self.period,
	self.n_sat,
	self.v)

	
class satellite(object):
	def  __init__(self,in_plane,i_in_plane,x,y,z,theta):
		self.in_plane= in_plane
		self.i_in_plane = i_in_plane
		self.x=x
		self.y=y
		self.z=z
		self.theta=theta

	def  __repr__(self):
		return '\n orbital plane= {}, index in plane= {}, pos x= {}, pos y= {}, pos z= {}'.format(
	self.in_plane,
	self.i_in_plane,
	self.x,
	self.y,
	self.z,
	self.theta)

	def rotate(self, deltat):
		self.theta += 2 *math.pi * deltat / Orbital_planes[self.in_plane].period
		self.theta = self.theta % (2 *math.pi)
		self.x = (Orbital_planes[self.in_plane].altitude+Re) * math.cos(Orbital_planes[self.in_plane].inclination) * math.sin(self.theta)
		self.y = (Orbital_planes[self.in_plane].altitude+Re) * math.sin(Orbital_planes[self.in_plane].inclination) * math.sin(self.theta)
		self.z = (Orbital_planes[self.in_plane].altitude+Re) * math.cos(self.theta)

	def rotate_axes(self, ang):
		R_z = np.array([[math.cos(ang),-math.sin(ang),0],[math.sin(ang), math.cos(ang),0],[0,0,1]])
		v = R_z.dot(np.array((self.x,self.y,self.z)))
		self.x = v[0]
		self.y = v[1]
		self.z = v[2]

class edge(object):
	def  __init__(self,sati,satj,weight, dji, dij):
		self.i = sati
		self.j = satj
		self.weight = weight
		self.dji = dji
		self.dij = dij


	def  __repr__(self):
		return '\n node i: {}, node j: {}, weight: {}'.format(
	self.i,
	self.j,
	self.weight)

	def __cmp__(self, other):
		if hasattr(other, 'weight'):
			return self.weight.__cmp__(other.weight)

def get_weight(edge):
		return(edge.weight)

def get_slant_range():
	slant_range = np.zeros((N,N), dtype=np.float128)
	for i in range(N):
		for j in range(i+1,N):
			slant_range[i,j] = math.sqrt((Satellites[i].x-Satellites[j].x)**2 +(Satellites[i].y-Satellites[j].y)**2+(Satellites[i].z-Satellites[j].z)**2)
	slant_range += np.transpose(slant_range)
	return slant_range

def get_weights_db():
	slant_range = get_slant_range()
	np.fill_diagonal(slant_range,1)
	L = 20*np.log10(4 * math.pi * slant_range * f / c)
	return L
	
def get_weights():
	slant_range = get_slant_range()
	L =(4 * math.pi * slant_range * f / c)**2
	return L
	
def get_direction():			# This is the simple function to get the direction. We should extend and consider a more realistic model
	direction = np.zeros((N,N), dtype=np.int8)
	for i in range(N):
		pi = Satellites[i].in_plane
		epsilon = -Orbital_planes[pi].inclination
		for j in range(N):
			direction[i,j] = np.sign(round(-Satellites[j].x*math.sin(epsilon)- Satellites[j].y*math.cos(epsilon)))
	return direction

def create_list(covered):
	List=[]
	if cross_seam_implementation:
		cross_seam = P
	else:
		cross_seam=P-1
	for i in range(N-1):
		for j in range(i+1,N):
			if Satellites[i].in_plane != Satellites[j].in_plane and abs(Satellites[i].in_plane-Satellites[j].in_plane)<cross_seam and ((i,direction[i,j]) not in covered) and ((j,direction[j,i]) not in covered) and direction[i,j]!=0 and Path_loss[i,j]<=MPL and slant_range[i,j]<LoS[Satellites[i].in_plane,Satellites[i].in_plane]:
				List.append(edge(i,j,Path_loss[i,j],direction[i,j], direction[j,i]))
	return List

def plot_sats(meta):
	Positions = np.zeros((N,3))
	for n in range(N):
		Positions[n,:] = [Satellites[n].x/1e6, Satellites[n].y/1e6, Satellites[n].z/1e6]
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	area = math.pi * (5**2)
	ax.scatter(Positions[:,0],Positions[:,1], Positions[:,2], c=meta, s=area)
	ax.scatter(Positions[0,0],Positions[0,1], Positions[0,2], marker="*", s=2*area)
	ax.scatter(Positions[int(N/P),0],Positions[int(N/P),1], Positions[int(N/P),2], marker="v", s=2*area)
	ax.set_aspect('equal', 'box')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	
	return Positions

def plot_isls(matching, meta):
	Positions = np.zeros((N,3))
	for n in range(N):
		Positions[n,:] = [Satellites[n].x/1e6, Satellites[n].y/1e6, Satellites[n].z/1e6]
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	area = math.pi * (5**2)
	ax.scatter(Positions[:,0],Positions[:,1], Positions[:,2], c=meta, s=area)
	ax.scatter(Positions[0,0],Positions[0,1], Positions[0,2], marker="*", s=2*area)
	ax.scatter(Positions[int(N/P),0],Positions[int(N/P),1], Positions[int(N/P),2], marker="v", s=2*area)
	ax.set_aspect('equal', 'box')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	cm=len(matching)
	for n in range(cm):
		pair=[matching[n][0],matching[n][1]]
		ax.plot([Positions[pair[0],0],Positions[pair[1],0]],[Positions[pair[0],1],Positions[pair[1],1]],[Positions[pair[0],2],Positions[pair[1],2]])
	
	return 0

def lcm(x, y):
	return x * y // gcd(x, y)

def prep_KM():
	w_matrix=np.zeros((N,N))
	w_matrix.fill(1e6)
	if cross_seam_implementation:
		cross_seam = P
	else:
		cross_seam=P-1
	for i in range(N-1):
		for j in range(i+1,N):
			if Satellites[i].in_plane != Satellites[j].in_plane and abs(Satellites[i].in_plane-Satellites[j].in_plane)<cross_seam and (direction[i,j]*direction[j,i]) ==-1  and Path_loss[i,j]<=MPL and slant_range[i,j]<LoS[Satellites[i].in_plane,Satellites[i].in_plane]:
				if direction[i,j]==-1:
					w_matrix[i,j]=Path_loss[i,j]
				else:
					w_matrix[j,i]=Path_loss[i,j]
	return w_matrix
	


cum_interference = np.zeros(resources[-1])
cum_SINR = np.zeros(resources[-1])
cum_rates = np.zeros((n_experiments,resources[-1]))
rates_ra = np.zeros(resources[-1])
rates_ra_vec = []
rates_vec = []
rates_vec_M = []
rates_vec_geo = []
p_delay_IE = []
p_delay_M = []
p_delay_geo = []
forest=np.zeros((n_experiments,2))

Orbital_planes = []
Satellites = []
index = 0
for m in range(P):
	n_sat_in_op = min(N-index,math.floor(N/P))
	index += n_sat_in_op
	h = Minimum_altitude + Delta_altitude*m
	T = 2 * math.pi * math.sqrt((h+Re)**3/(G*Me))
	epsilon_p = m*math.pi/P
	Orbital_planes.append(orbital_plane(h, epsilon_p, T, n_sat_in_op))

#print(Orbital_planes)


LoS=np.zeros((P,P))
for m in range(P):
	LoS[m,m] = 2*np.sqrt(Orbital_planes[m].altitude*(Orbital_planes[m].altitude+2*Re))
	for n in range(m):
		LoS[m,n] = np.sqrt(Orbital_planes[m].altitude*(Orbital_planes[m].altitude+2*Re))+np.sqrt(Orbital_planes[n].altitude*(Orbital_planes[n].altitude+2*Re))
		LoS[n,m] = LoS[m,n]

#print('LoS=' + repr(LoS))




m = 0
indexinplane = 0
indexinconst = Orbital_planes[0].n_sat
t = 0
random_disp = np.random.random(P)

# Initialize the positiond of the satellites
for n in range(N):
	if n==indexinconst:
		m +=1
		indexinplane = 0
		indexinconst += Orbital_planes[m].n_sat
	thet=2 *math.pi *(random_disp[m]+indexinplane/N_p)
	Satellites.append(satellite(m, indexinplane,0,0,0,thet))
	Satellites[n].rotate(n_disp*matching_period)
	indexinplane += 1
#print(repr(Satellites))
	
Periods=[]

for m in range(P):
	Periods.append(Orbital_planes[m].period)

	
#print('Orbital periods [s]: ' + repr(Periods))

if matching_period==0:
	Simulation_period = np.max(Periods) * n_rotations
	matching_period = Simulation_period/n_experiments
else:
	Simulation_period = matching_period * n_experiments


print("Running simulation with {} orbital planes and {} satellites per orbital plane\nTotal number of satellites is {}".format(P,N_p, N))

print('Carrier frequency [GHz]:' + repr(f/1e9))

print('Duration of the simulation [s]: ' + repr(Simulation_period))
print('Matching period [s] = ' + repr(matching_period))
print("Simulating {} matching instants".format(Simulation_period/matching_period))


slant_range=get_slant_range()

print('Intra-plane slant range [km] = {:0.3f}'.format(slant_range[0,1]/1e3))

slnt_rng = slant_range.copy()

################################################################################################
########################		Intra_plane rates		########################
################################################################################################
intra_plane_rates=[]

loss = get_weights()
for i in range(N-1):
	m=Satellites[i].in_plane
	if (Satellites[i+1].in_plane==m):
		intra_plane_rates.append(B*math.log2(1+1/(rnp*loss[i][i+1])))
	else:
		intra_plane_rates.append(B*math.log2(1+1/(rnp*loss[i][i-N_p+1])))
intra_plane_rates.append(B*math.log2(1+1/(rnp*loss[N-1][N-N_p])))


#plt.figure()
#plt.imshow(slnt_rng)

Path_loss = get_weights_db()

#plt.figure()
#plt.imshow(Path_loss)


#plt.figure()
#plt.imshow(C)

#print(repr(C))



d_orbital_neigh = []
aux=0
for m in range(P):
	d_orbital_neigh.append(slant_range[0+aux,1+aux])
	aux += Orbital_planes[m].n_sat

slant_range_intra = np.min(d_orbital_neigh)
prop_delay_intra = slant_range_intra/c

print("Intra-plane distance [km] = {}".format(np.array(d_orbital_neigh)/1e3))

#print(repr(slant_range_intra))
#print(repr(Ptl))
#print(repr(Pth))


######################################### Matching #########################################
t =0 #- matching_period

Exec_time = np.zeros((n_experiments,4))
Num_pairs = np.zeros((n_experiments,4))
rates_snr = np.zeros((n_experiments,4))
SINR = np.zeros(n_experiments)
Expected_cost = np.zeros((n_experiments,4))
n_experiment = 0
A_Markovian =[]
n_pos_pairs=0
length_of_list=0
direction = np.zeros((N,N))
for n_experiment in range(n_experiments):
	print('Experiment no. ' + repr(n_experiment))
	t+= matching_period
	for n in range(N):
		Satellites[n].rotate(matching_period)
	slant_range = get_slant_range()
	Path_loss = get_weights_db()
	direction = get_direction()
	loss =np.array(get_weights(),dtype=np.float128)
	#np.savetxt('iloss_direct.txt',loss)
	#print("Direction " + repr(direction))
	#input('')
	sat_dir = direction.copy()
	if num_transceivers == 1:
		direction = np.abs(direction)
	start = time.time()

	W=create_list(set())
	end = time.time()
	n_pos_pairs += len(W)
	init_time= end-start
	#np.savetxt('wmatrix.txt',W_matrix)
	#print(W)
	#print(len(W))

###################################### Independent experiments #######################################
	X = np.zeros(N)
	covered = set()
	cost = 0
	A =[]
	start = time.time()
	W_sorted=sorted(W,key=get_weight)
	#print('First element'+repr([W_sorted[0].i,W_sorted[0].j])+' '+repr(loss[W_sorted[0].i][W_sorted[0].j])+' '+repr(1/loss[W_sorted[0].i][W_sorted[0].j]))
	while W_sorted and W_sorted[0].weight<=MPL:
		if  ((W_sorted[0].i,W_sorted[0].dji) not in covered) and ((W_sorted[0].j,W_sorted[0].dij) not in covered):
			A.append([W_sorted[0].i,W_sorted[0].j, W_sorted[0].weight])
			p_delay_IE.append(slant_range[W_sorted[0].i,W_sorted[0].j]/c)
			covered.add((W_sorted[0].i,W_sorted[0].dji))
			covered.add((W_sorted[0].j,W_sorted[0].dij))
			X[W_sorted[0].i]+=1
			X[W_sorted[0].j]+=1
			#print(repr(covered))
			#input('')
			cost += W_sorted[0].weight
			rates_snr[n_experiment,0]+=2*B*math.log2(1+1/(rnp*loss[W_sorted[0].i][W_sorted[0].j]))
			rates_vec.append(B*math.log2(1+1/(rnp*loss[W_sorted[0].i][W_sorted[0].j])))
			Num_pairs[n_experiment,0]+=1
		W_sorted.pop(0)
	end = time.time()
	#print(repr({A[0:20][0]}))
	#input('')
	init_time=init_time*pre_processing
	Exec_time[n_experiment,0] = end-start+init_time #Remove this to exclude pre-processing
	Num_pairs[n_experiment,0] = len(A)
	Expected_cost[n_experiment,0] = cost/Num_pairs[n_experiment,0]
####################################### Signal to interference ratio (SIR) #######################################
	aux=np.copy(loss)
	np.fill_diagonal(aux,1) 
	ILoss = np.array(1/aux,dtype=np.float128)
	np.fill_diagonal(ILoss,lamda) 
	match = set()
	
	for a in range(len(A)):
		match = match | {A[a][0]} | {A[a][1]}
	matched=list(match)
	num_v=len(matched)
	num_e=len(A)
	forest[n_experiment,:]=[num_v,num_e]
	#print('Is=' + repr(A[0][0]))
	for n in range(len(matched)):
		for m in range(n+1,len(matched)):
			i = matched[n]
			j = matched[m]
			if slant_range[i,j]> LoS[Satellites[i].in_plane,Satellites[j].in_plane]:
				#print('[i,j] = ' + repr([i,j]))
				#input('')
				ILoss[i,j]=0
				ILoss[j,i]=0
	start = time.time()
	for num_res in resources:#range(r_min,r_max+1):
		B_res=B
		if FDMA:
			B_res= B/num_res
		rnp_res=(k*Ts*B_res)/EIRP
		#print('Channel bandwidth = ' + repr(B_res) + ' 1/SNR = ' + repr(rnp_res))
		Interference = np.zeros((len(A),num_res),dtype=np.float128)
		res_alloc = np.zeros((len(A), num_res))
		res_alloc_indx=np.zeros(len(A),dtype=int)
		SINR = np.zeros(len(A))
		max_interference_u = np.zeros(len(A),dtype=np.float128)
		max_interference_v = np.zeros(len(A),dtype=np.float128)
		#print('Loss =' + repr(Path_loss[0:10,0:10]))
		#print('Loss inverse =' + repr(ILoss[0:10,0:10]))
		#print('Matching = ' +repr(A))
		for a in range(len(A)):
			i=A[a][0]
			j=A[a][1]
			a1, inst_int_u = np.meshgrid(list(range(num_res)),max_interference_u)
			a1, inst_int_v = np.meshgrid(list(range(num_res)),max_interference_v)
			Inst_rates = np.zeros(num_res)
			for b in range(a):
				u=A[b][0]
				v=A[b][1]
				inst_int_u[b,res_alloc_indx[b]] +=np.maximum(ILoss[i,u], ILoss[j,u])
				inst_int_v[b,res_alloc_indx[b]] +=np.maximum(ILoss[i,v], ILoss[j,v])
				inst_int_u[a,:] +=np.maximum(ILoss[u,i]*res_alloc[b,:], ILoss[v,i]*res_alloc[b,:])
				inst_int_v[a,:] +=np.maximum(ILoss[u,j]*res_alloc[b,:], ILoss[v,j]*res_alloc[b,:])
				Inst_rates += np.log2(1+ILoss[u,v]/(rnp_res+ inst_int_u[b,:]))+np.log2(1+ILoss[u,v]/(rnp_res+inst_int_v[b,:]))
			Inst_rates += np.log2(1+ILoss[i,j]/(rnp_res+ inst_int_u[a,:]))+np.log2(1+ILoss[i,j]/(rnp_res+inst_int_v[a,:]))		
			#Delta_SINR = sum(ILoss[i,j]/(rnp+ Aux_u))+sum(ILoss[i,j]/(rnp+Aux_v))
			#print('ISL = '+repr(A[a]))
			#print('Aux u = ' +repr(inst_int_u))
			#print('Rates = ' +repr(Inst_rates))
			#input('')
			#if n_experiment==32:
			#	print('Aux, int=' + repr([Aux,Interference]))
			#	print('Delta_int=' + repr(Delta_int))
			if ra_index==0:
				band = list(Inst_rates).index(max(Inst_rates)) 			### Matching
			elif ra_index==1:
				band = a % num_res					### Round-robin
			else:
				band = random.randrange(num_res)  			### Random allocation 
			res_alloc[a,band]=1
			res_alloc_indx[a]=band
			max_interference_u = inst_int_u[:,band]
			max_interference_v = inst_int_v[:,band]
		min_SINR_u = np.zeros(len(A),dtype=np.float128)
		min_SINR_v = np.zeros(len(A),dtype=np.float128)
		for a in range(len(A)):
			u=A[a][0]
			v=A[a][1]
			#tst=np.array(1/loss[u,v],dtype=np.float128)
			#print('Here'+repr([u,v])+' '+repr(tst)+' '+repr(ILoss[u,v]))
			#input('')
			#if sum(Interference[a,:]):
			min_SINR_u[a] = ILoss[u,v]/(rnp_res+ max_interference_u[a])
			min_SINR_v[a] = ILoss[u,v]/(rnp_res+ max_interference_v[a])
			rates_ra[num_res-1] += B_res*math.log2(1+min_SINR_u[a])+B_res*math.log2(1+min_SINR_v[a])
			cum_rates[n_experiment,num_res-1] += B_res*math.log2(1+min_SINR_u[a])+B_res*math.log2(1+min_SINR_v[a])
			rates_ra_vec.append(B_res*math.log2(1+min_SINR_u[a]))
			rates_ra_vec.append(B_res*math.log2(1+min_SINR_v[a]))		
	end = time.time()
	Exec_time[n_experiment,3] = end-start#+init_time*pre_processing
	######################################### Markovian matching #########################################
	covered = set()
	XM = np.zeros(N)
	cost = 0
	A_Markovian_new = []
	start = time.time() #Here to include pre-processing
	for a in A_Markovian:
		i = a[0]
		j = a[1]
		if Path_loss[i,j]<=MPL and ((i,direction[i,j]) not in covered) and ((j,direction[j,i]) not in covered):
			A_Markovian_new.append([i,j, Path_loss])
			p_delay_M.append(slant_range[i,j]/c)
			covered.add((i,direction[i,j]))
			covered.add((j, direction[j,i]))
			#XM[i]+=1
			#XM[j]+=1
			cost += Path_loss[i,j]
			rates_snr[n_experiment,1]+=2*B*math.log2(1+1/(rnp*loss[i,j]))
			rates_vec_M.append(B*math.log2(1+1/(rnp*loss[i,j])))
			#Num_pairs[n_experiment,1]+=1
	W_M=create_list(covered)
	init_time=time.time()-start
	length_of_list+=len(W_M)
	#print(len(W_M))
	start = time.time() # Here to exclude pre-processing
	W_sorted=sorted(W_M,key=get_weight)
	
	A_Markovian = A_Markovian_new.copy()
	while W_sorted and W_sorted[0].weight<=MPL:
		if  ((W_sorted[0].i,W_sorted[0].dji) not in covered) and ((W_sorted[0].j,W_sorted[0].dij) not in covered):
			A_Markovian.append([W_sorted[0].i,W_sorted[0].j, W_sorted[0].weight])
			p_delay_M.append(slant_range[W_sorted[0].i,W_sorted[0].j]/c)
			covered.add((W_sorted[0].i,W_sorted[0].dji))
			covered.add((W_sorted[0].j,W_sorted[0].dij))
			XM[W_sorted[0].i]+=1
			XM[W_sorted[0].j]+=1
			cost += W_sorted[0].weight
			rates_snr[n_experiment,1]+=2*B*math.log2(1+1/(rnp*loss[W_sorted[0].i][W_sorted[0].j]))
			rates_vec_M.append(B*math.log2(1+1/(rnp*loss[W_sorted[0].i][W_sorted[0].j])))
			#Num_pairs[n_experiment,1]+=1
		W_sorted.pop(0)
	end = time.time()
	Exec_time[n_experiment,1] = end-start+init_time*pre_processing
	Num_pairs[n_experiment,1] = len(A_Markovian)
	Expected_cost[n_experiment,1] = cost/Num_pairs[n_experiment,1]
	

###################################### GEO matching #######################################
	#input('') 
	X_geo = np.zeros(N)
	covered = set()
	cost = 0
	A_geo =[]
	if(cdfs==1):
		rwidth= 2*math.pi/N_p
		ring=np.zeros((N_p,P), dtype=int)
		start = time.time()
		for i in range(N):
			indx=int(Satellites[i].theta//rwidth)
			m=Satellites[i].in_plane
			ring[indx,m]=i
		for indx in range(N_p):	
			for m in range(1,P):
				i=ring[indx,m-1]
				j=ring[indx,m]
				#if (Path_loss[i,j]>MPL):
				#	print('i j= '+repr([i,j])+' FSPL= ' + repr(Path_loss[i,j]) + ' latitudes = ' +repr([Satellites[i].theta,Satellites[j].theta]))
				#	meta=np.zeros(N)
				#	meta[i]=1
				#	meta[j]=1
				#	plot_sats(meta)
				#	plt.show()
				#	input('')
				if (((i,direction[i,j]) not in covered) and ((j,direction[j,i]) not in covered) and Path_loss[i,j]<=MPL):
					A_geo.append([i,j, Path_loss[i,j]])
					p_delay_geo.append(slant_range[i,j]/c)
					covered.add((i,direction[i,j]))
					covered.add((j,direction[j,i]))
					X_geo[i]+=1
					X_geo[j]+=1
					cost += Path_loss[i,j]
					#print('Ring = ' + repr(ring)+' loss= '+repr(loss[ring[m-1],[ring[m]]]))	
					rates_snr[n_experiment,2]+=2*B*math.log2(1+1/(rnp*loss[i,j]))
					rates_vec_geo.append(B*math.log2(1+1/(rnp*loss[i,j])))
		end = time.time()
		Exec_time[n_experiment,2] = end-start
		Num_pairs[n_experiment,2] = len(A_geo)
		Expected_cost[n_experiment,2] = cost/Num_pairs[n_experiment,2]


######################################### Hungarian algorithm #########################################
	if num_transceivers!=2:
		#pass
		continue
	else:
		W_matrix = prep_KM()
	start = time.time()	
	rows, cols = linear_sum_assignment(W_matrix)
	end = time.time()
	#Exec_time[n_experiment,3]=end-start+init_time

	A_Hungarian=[]
	cost = 0
	for i in range(N):
		if W_matrix[i,cols[i]]<MPL:
			A_Hungarian.append([i,cols[i],W_matrix[i,cols[i]]])
			cost += W_matrix[i,cols[i]]
			rates_snr[n_experiment,3]+=2*B*math.log2(1+1/(rnp*loss[i,cols[i]]))
	Num_pairs[n_experiment,3] = len(A_Hungarian)
	#print(repr(A_Hungarian))
	Expected_cost[n_experiment,3] = cost/Num_pairs[n_experiment,3]
#print('XM= ' + repr(XM))
sats_in_plane = np.zeros(N)
for n in range(N):
#	Satellites[n].rotate_axes(-math.pi/P)
	sats_in_plane[n] = Satellites[n].in_plane
	

Positions = plot_sats(sats_in_plane)
plt.xlabel('x')
plt.ylabel('y')
txt = np.zeros((N,5))
txt[:,0:3]=Positions
for n in range (N):
	txt[n,3]= Satellites[n].in_plane
txt[:,4] = X
fname = 'satellite_matching_' + repr(MPL) + 'dB_M' + repr(P)+ '.txt'
#np.savetxt(fname,txt)
plot_sats(X)
plot_isls(A,sats_in_plane)



Mean_num_pairs = np.mean(Num_pairs, axis=0)
Mean_exec_time = np.mean(Exec_time, axis=0)
Mean_expected_cost = np.mean(Expected_cost, axis=0)
Mean_rates_snr = np.mean(rates_snr, axis=0)/1e6
Mean_rates_ra = rates_ra/(n_experiments*1e6)
Mean_n_pos_pairs = np.array([n_pos_pairs/(n_experiments*N), length_of_list/(n_experiments*N)])
Mean_interference = np.array([cum_interference/(n_experiments*Mean_num_pairs[0]), cum_interference/(num_transceivers*N*n_experiments)])
Mean_intra_plane_rates = [np.mean(intra_plane_rates, axis=0)/1e6, 2*np.sum(intra_plane_rates, axis=0)/1e6]
Mean_prop_delay_inter = [np.mean(np.array(p_delay_IE)), np.mean(np.array(p_delay_M)), np.mean(np.array(p_delay_geo))]
Mean_prop_delay_intra = np.mean(prop_delay_intra)

Mean_rates = [np.mean(rates_vec)/1e6, np.mean(rates_ra_vec)/1e6]
SNR_inter_plane_rates=[np.mean(rates_vec)/1e6, np.median(rates_vec)/1e6, np.percentile(rates_vec,95, interpolation='lower')/1e6]
Inter_plane_prop_delay=[np.mean(p_delay_IE), np.median(p_delay_IE, axis=0), np.percentile(p_delay_IE,95, interpolation='lower')]

SNR_intra_plane_rates=[np.mean(intra_plane_rates)/1e6, np.median(intra_plane_rates)/1e6, np.percentile(intra_plane_rates,95, interpolation='lower')/1e6]
Intra_plane_prop_delay=[np.mean(prop_delay_intra), np.median(prop_delay_intra), np.percentile(prop_delay_intra,95, interpolation='lower')]

print('Mean number of possible pairs= ' + repr(Mean_n_pos_pairs))
print('Mean number of pairs= ' + repr(Mean_num_pairs))
print('Mean execution time = ' + repr(Mean_exec_time))
print('Mean expected cost= ' + repr(Mean_expected_cost))
print('Mean sum rates_SNR [Mbps]= ' + repr(Mean_rates_snr))
print('Mean sum rates SINR [Mbps]= ' + repr(Mean_rates_ra))


print('Rates at the inter-plane ISLs (SNR)= ' + repr(SNR_inter_plane_rates))
print('Rates at the intra-plane ISLs = ' + repr(SNR_intra_plane_rates))

print('Inter-plane propagation delay = ' + repr(Inter_plane_prop_delay))
print('Intra-plane propagation delay= ' + repr(Intra_plane_prop_delay))
endtime=time.time()

totaltime=endtime-starttime
print('Execution time= ' + repr(totaltime)+ ' s')

fname=f'forests_N{N}_P{P}_Q{num_transceivers}.txt'
#np.savetxt(fname, forest, delimiter=',', fmt='%d')



plt.figure()
numbeans = 500
#beans = np.append([0],np.linspace(min(rates_ra_vec),max(rates_vec),numbeans))
#beans = np.append([0],np.linspace(1e3,1e7,numbeans))	# For B=20MHz
#beans = np.append([0],np.logspace(3,7,numbeans))	# For B=20MHz
#beans = np.linspace(1e-5,1e-2,numbeans)				# For rho=1e12
#beans = np.append([0],np.logspace(-2,1,numbeans))				# For rho=10^(17.1)
beans = np.append([0],np.logspace(-2,1,numbeans))				# For rho=10^(17.1)
hst, bns = np.histogram(np.array(rates_vec)/1e6, bins=beans)
pmf= np.array(hst)/np.sum(np.array(hst))
CDF = np.cumsum(pmf)
#print('F_(SIR)= ' + repr(CDF))
#plt.figure()
plt.plot(beans[1:],CDF, label='ISL matching')
txt = np.zeros((len(beans)-1,6))
txt[:,0] = beans[1:]
txt[:,1] = CDF

hst, bns = np.histogram(np.array(rates_vec_M)/1e6, bins=beans)
pmf= np.array(hst)/np.sum(np.array(hst))
CDF = np.cumsum(pmf)
#print('F_(SIR)= ' + repr(CDF))
#plt.figure()
plt.plot(beans[1:],CDF, '--',label='Markovian')
txt[:,2] = CDF

hst, bns = np.histogram(np.array(rates_vec_geo)/1e6, bins=beans)
pmf= np.array(hst)/np.sum(np.array(hst))
CDF = np.cumsum(pmf)
#print('F_(SIR)= ' + repr(CDF))
#plt.figure()
plt.plot(beans[1:],CDF, '--',label='GEO')
txt[:,3] = CDF

hst, bns = np.histogram(rates_ra_vec, bins=beans)
pmf= np.array(hst)/np.sum(np.array(hst))
CDF = np.cumsum(pmf)
#print('F_(SIR)= ' + repr(CDF))
#plt.figure()
plt.plot(beans[1:],CDF, '--',label='RA')
txt[:,4] = CDF

hst, bns = np.histogram(np.array(intra_plane_rates)/1e6, bins=beans)
pmf= np.array(hst)/np.sum(np.array(hst))
CDF = np.cumsum(pmf)
plt.plot(beans[1:],CDF, '--',label='Intra-plane')
txt[:,5] = CDF

plt.xscale('log')
plt.legend()
plt.title('CDF of the allocated rates')
fname = 'CDF_rates_Ts_'+repr(round(Ts))+'_P' + repr(P) + '_N' + repr(N) + '_'+ r_allocation[ra_index] + '_lambda' + repr(lamda) + '_R' + repr(resources[-1])+ '.txt'
#np.savetxt(fname,txt)
#np.savetxt('CDF_SIR_MPL170_rr_no_self_int_R10.txt',txt)

plt.figure()
numbeans = 200
beans = np.append([0],np.linspace(1e-4,20e-3,numbeans))
hst, bns = np.histogram(p_delay_IE, bins=beans)
pmf= np.array(hst)/np.sum(np.array(hst))
CDF = np.cumsum(pmf)
#print('F_(SIR)= ' + repr(CDF))
#plt.figure()
plt.plot(beans[1:],CDF, label='GIEM')
#plt.xscale('log')
txt = np.zeros((len(beans)-1,4))
txt[:,0] = beans[1:]*1e3
txt[:,1] = CDF
hst, bns = np.histogram(p_delay_M, bins=beans)
pmf= np.array(hst)/np.sum(np.array(hst))
CDF = np.cumsum(pmf)
plt.plot(beans[1:],CDF, label='GMM')
txt[:,2] = CDF
hst, bns = np.histogram(p_delay_geo, bins=beans)
pmf= np.array(hst)/np.sum(np.array(hst))
CDF = np.cumsum(pmf)
txt[:,3] = CDF
plt.plot(beans[1:],CDF, label='GEO')
fname = 'CDF_p_delay_Ts_'+repr(round(Ts))+'_MPL' + repr(MPL) + '_Q'+ repr(num_transceivers)+ '_P'+ repr(P)+'.txt'
#np.savetxt(fname,txt)



#print('IE: ' + repr(A))
#print('MM: ' + repr(A_Markovian))
#print('HA: ' + repr(A_Hungarian))
numbeans = 1000
t_max = np.amax(Exec_time)
beans = np.append([0],np.logspace(-5,1,numbeans))
H_Exec_time_IE, be  = np.histogram(Exec_time[:,0],bins=beans)
H_Exec_time_MM, be = np.histogram(Exec_time[:,1],bins=beans)
H_Exec_time_GEO, be = np.histogram(Exec_time[:,2],bins=beans)
H_Exec_time_RA, be = np.histogram(Exec_time[:,3],bins=beans)
Exec_time_IE_CDF = np.cumsum(np.array(H_Exec_time_IE) / n_experiments)
Exec_time_MM_CDF = np.cumsum(np.array(H_Exec_time_MM)/ n_experiments)
Exec_time_GEO_CDF = np.cumsum(np.array(H_Exec_time_GEO)/ n_experiments)
Exec_time_RA_CDF = np.cumsum(np.array(H_Exec_time_RA)/ n_experiments)

txt = np.zeros((len(beans)-1,5))
txt[:,0] = beans[1:]
txt[:,1] = Exec_time_IE_CDF
txt[:,2] = Exec_time_MM_CDF
txt[:,3] = Exec_time_GEO_CDF
txt[:,4] = Exec_time_RA_CDF
fname = 'CDFs_' + repr(matching_period) + 's_MPL' + repr(MPL)+'_Ts_'+repr(round(Ts)) + '_Q'+repr(num_transceivers)+ '_pre_processing' +repr(pre_processing)+'.txt'
np.savetxt(fname,txt)
#np.savetxt('CDFs_300s_MPL170_600km.txt',txt)

plt.figure()
plt.plot(beans[0:numbeans],Exec_time_IE_CDF, label='GIEM')
plt.plot(beans[0:numbeans],Exec_time_MM_CDF, label='GMM')
plt.plot(beans[0:numbeans],Exec_time_GEO_CDF, label='GEO')
plt.plot(beans[0:numbeans],Exec_time_RA_CDF, label='RA')
plt.xscale('log')
plt.legend()
plt.show()
