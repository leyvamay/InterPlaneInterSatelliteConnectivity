import matplotlib.pyplot as plt
import math
import numpy as np
import networkx as nx

Re = 6378e3	# Radius of the earth [m]
G = 6.67259e-11	# Universal gravitational constant [m^3/kg s^2]
Me = 5.9736e24	# Mass of the earth
k = 1.38e-23	# Boltzmann's constant
c = 299792458	# Speed of light [m/s]

#################### Spectral efficiency for DVB-2 system
speff_thresholds = np.array([0,0.434841,0.490243,0.567805,0.656448,0.789412,0.889135,0.988858,1.088581,1.188304,1.322253,1.487473,1.587196,1.647211,1.713601,1.779991,1.972253,2.10485,2.193247,2.370043,2.458441,2.524739,2.635236,2.637201,2.745734,2.856231,2.966728,3.077225,3.165623,3.289502,3.300184,3.510192,3.620536,3.703295,3.841226,3.951571,4.206428,4.338659,4.603122,4.735354,4.933701,5.06569,5.241514,5.417338,5.593162,5.768987,5.900855])
lin_thresholds = np.array([1e-10,0.5188000389,0.5821032178,0.6266138647,0.751622894,0.9332543008,1.051961874,1.258925412,1.396368361,1.671090614,2.041737945,2.529297996,2.937649652,2.971666032,3.25836701,3.548133892,3.953666201,4.518559444,4.83058802,5.508076964,6.45654229,6.886522963,6.966265141,7.888601176,8.452788452,9.354056741,10.49542429,11.61448614,12.67651866,12.88249552,14.48771854,14.96235656,16.48162392,18.74994508,20.18366364,23.1206479,25.00345362,30.26913428,35.2370871,38.63669771,45.18559444,49.88844875,52.96634439,64.5654229,72.27698036,76.55966069,90.57326009])
db_thresholds = np.array([-100.00000, -2.85000, -2.35000 ,-2.03000 ,-1.24000, -0.30000, 0.22000, 1.00000, 1.45000, 2.23000, 3.10000, 4.03000, 4.68000, 4.73000, 5.13000, 5.50000, 5.97000, 6.55000, 6.84000, 7.41000, 8.10000, 8.38000, 8.43000, 8.97000, 9.27000, 9.71000,10.21000,10.65000,11.03000, 11.10000, 11.61000, 11.75000, 12.17000, 12.73000, 13.05000, 13.64000, 13.98000, 14.81000, 15.47000, 15.87000, 16.55000, 16.98000, 17.24000, 18.10000, 18.59000, 18.84000, 19.57000])

####################

def commercial_constellation_parameters(name):
    if name =="Kepler":
        print("Using Kepler constellation design")
        return 140, np.ones(7)*600e3, 98.6, True          
    elif name =="Iridium_NEXT":
        print("Using Iridium NEXT constellation design")
        return 66, np.ones(6)*780e3, 86.4, True       
    elif name =="OneWeb":
        print("Using OneWeb constellation design")
        return 648, np.ones(18)*1200e3, 86.4,True
    elif name =="Starlink":           # 
        print("Using Starlink Phase 1 550 km altitude orbital shell design")
        return 1584, np.ones(72)*550e3, 53, False 
    else:
        print("Using homemade constellation design")
        return 140, np.ones(7)*600e3, 53, True

class Constellation:
    def __init__(self, N, h, inclination, walker_star):
        self.N = N 						# Number of satellites
        self.P = len(h)					# Number of orbital planes
        self.h = h						# Altitude of deployment for each orbital plane (set to the same altitude here)
        self.inclination = inclination				# Inclination angle for the orbital planes,
        self.N_p = int(self.N/self.P)				# Number of satellites per orbital plane
        self.walker_star = walker_star				# True for Walker star and False for Walker Delta
        self.right_ascension = np.array((2*math.pi-self.walker_star*math.pi)*np.arange(0,self.P))/self.P
        self.satellites=[]
        for n in range(N):
            p=n//self.N_p
            self.satellites.append(Satellite(n,p,n-p*self.N_p,self.h[p], self.N_p,self.right_ascension[p], self.inclination, 0))
    def rotate(self, delta_t):
        for satellite in self.satellites:
            satellite.rotate(delta_t)        

class Satellite:
    def __init__(self, n,orbital_plane,index_in_plane, h, N_p, right_ascension, inclination, displacement):
        self.id = n
        self.in_plane = orbital_plane
        self.index = index_in_plane
        self.h = h
        self.period = 2 * math.pi * math.sqrt((self.h+Re)**3/(G*Me))	# Orbital period 
        self.polar_angle = 2 *math.pi *(self.index+displacement)/N_p
        self.right_ascension = right_ascension
        self.delta = math.pi/2-math.radians(inclination)
        self.v = 2*math.pi * (self.h + Re) / self.period
        self.x = 0
        self.y = 0
        self.z = 0
        self.rotate(0)
        
    def rotate(self, delta_t):			# To rotate the satellites after a period delta_t using the polar angle
        self.polar_angle = (self.polar_angle+2 *math.pi * delta_t / self.period)%(2*math.pi)
        self.x = (self.h+Re) * (math.sin(self.polar_angle)*math.cos(self.right_ascension)+math.cos(self.polar_angle)*math.sin(self.delta)*math.sin(self.right_ascension))
        self.y = (self.h+Re) * (math.sin(self.polar_angle)*math.sin(self.right_ascension)-math.cos(self.polar_angle)*math.sin(self.delta)*math.cos(self.right_ascension))
        self.z = (self.h+Re) * math.cos(self.polar_angle)*math.cos(self.delta)

    def  __repr__(self):
        return '\n %%%%%%%%\nSatellite = {} in plane= {} \npos x= {} \npos y= {}  \npos z= {}\n polar angle = {} right ascension angle = {}\n Orbital period = {} hours \nOrbital velocity = {} km/s'.format(
    self.index,
    self.in_plane,
    '%.2f'%self.x,
    '%.2f'%self.y,
    '%.2f'%self.z,
    '%.2f'%self.polar_angle,
    '%.2f'%self.right_ascension,
    '%.2f'%(self.period/3600),
    '%.2f'%(self.v/1e3))
    			
class GroundSegment:
    def __init__(self, ground_stations):
        self.ground_stations = []
        for i in range(len(ground_stations)):       
            self.ground_stations.append(GroundStation(i,ground_stations[i][0],ground_stations[i][1],ground_stations[i][2]))
    def rotate(self, delta_t):
        for gs in self.ground_stations:
            gs.rotate(delta_t) 

class GroundStation:
    def  __init__(self,ID,location,latitude_deg,longitude_deg):
        self.id = ID						# ID 
        self.h = 0
        self.location = location
        self.latitude = math.radians(latitude_deg)		# Latitude in radians
        self.longitude = math.radians(longitude_deg)		# Longitude in radians

        self.polar_angle = (math.pi/2-self.latitude+2*math.pi)%(2*math.pi)	# Polar angle in radians						# Distance to the center of the Earth
        self.x=Re*math.cos(self.longitude)*math.sin(self.polar_angle)	# Cartesian coordinates  (x,y,z)
        self.y=Re*math.sin(self.longitude)*math.sin(self.polar_angle)
        self.z=Re*math.cos(self.polar_angle)

        self.rotation_period = 2 * math.pi * math.sqrt((35786e3+Re)**3/(G*Me))	# Rotation of the Earth
        
    def  __repr__(self):
        return '\n ID= {}, latitude= {}, longitude = {}'.format(
    self.id,
    '%.2f'%math.degrees(self.latitude),
    '%.2f'%math.degrees(self.longitude),
    )

    def rotate(self, delta_t):			# To rotate the satellites after a period delta_t using the polar angle
        self.longitude = (self.longitude +2 *math.pi * delta_t / self.rotation_period) % (2*math.pi)
        self.x=Re*math.cos(self.longitude)*math.sin(self.polar_angle)
        self.y=Re*math.sin(self.longitude)*math.sin(self.polar_angle)
             
class FSOLink():
    def  __init__(self, sat_i, sat_j, power, comm_range, data_rate): 
        self.power = power
        self.comm_range = comm_range
        self.module_rate = data_rate
        self.endvertices = (sat_i, sat_j)
        self.slant_range=0
        self.in_range= False
        self.data_rate = 0
        self.time_to_complete = 0
        self.get_slant_range()
        self.active = False
    def  __repr__(self):
        return '\n%%%%%\nLink between Satellites {} and {}\nData rate = {} Mbps\n Power = {} W\n Transmission range = {} km'.format(
        self.endvertices[0].index,
        self.endvertices[1].index,
        self.data_rate/1e6,
        self.power,
        self.slant_range/1e3,)
    def get_slant_range(self):
        self.slant_range=math.sqrt((self.endvertices[0].x-self.endvertices[1].x)**2 +(self.endvertices[0].y-self.endvertices[1].y)**2+(self.endvertices[0].z-self.endvertices[1].z)**2)		
        self.in_range = self.comm_range>self.slant_range
    def get_data_rate(self):
        self.data_rate = self.in_range*self.module_rate
        return self.data_rate
    def establish(self):
        if self.in_range:
            self.active = True
            self.data_rate = self.module_rate * self.active
    def disable(self):
        self.active = False
        self.data_rate = 0
    def transmission(self, data_size, t):
        #print(self.data_rate)
        self.time_to_complete = max(t,self.time_to_complete) + data_size/self.data_rate + self.slant_range/c   
        return self.time_to_complete  
    def transmission_time(self, data_size):
        transmission_time = data_size/self.data_rate + self.slant_range/c   
        return transmission_time  
        
  
class InterSatelliteLinks:
    def __init__(self, satellites, link_type):
        self.links = []
        self.rates = np.zeros((len(satellites),len(satellites)))
        for u in range(len(satellites)):
            for v in range(u):
                if link_type=="RF" or (link_type=="Hybrid" and satellites[u].in_plane!=satellites[v].in_plane):
                    ####### NGEO inter-plane
                    #print('NGEO RF inter-satellite link\n')
                    f = 26e9    # Carrier frequency GEO to ground (Hz)
                    B = 500e6   # Maximum bandwidth
                    maxPtx = 10  # Maximum tansmission power in W
                    Adtx = 0.26 # Transmitter antenna diameter in m
                    Adrx = 0.26 # Receiver antenna diameter in m
                    pL = 0.3    # Pointing loss in dB
                    Nf = 2      # Noise figure in dB
                    Tn = 290    # Noise temperature in K
                    eff = 0.55
                    self.links.append(RFLink(satellites[u], satellites[v], f, B, maxPtx,Adtx, Adrx, pL, Nf, Tn, eff, "ISL", 0))
                    self.rates[u,v] = self.links[-1].get_data_rate()
                    self.rates[v,u] = self.rates[u,v]
                else:
                    #print('FSO link\n')
                    ####### NGEO intra-plane high rate
                    R = 10e9        # Fixed data rate (bps)
                    Pt = 60         # Transmission power (W)
                    maxRange = 6000e3   # Communication range
                    self.links.append(FSOLink(satellites[u], satellites[v], Pt, maxRange, R))
                    self.rates[u,v] = self.links[-1].get_data_rate()
                    self.rates[v,u] = self.rates[u,v]

    def plot_rates(self):
        plt.figure()
        plt.imshow(self.rates/1e9)
        plt.xlabel("Satellites")
        plt.ylabel("Satellites")
        plt.title("Rates inter-satellite links [Gbps]")
class GroundtoSatelliteLinks:
    def __init__(self, satellites, gs):  
        self.links = []
        self.rates_downlink = np.zeros((len(satellites),len(gs)))
        self.rates_uplink = np.zeros((len(satellites),len(gs)))
        for u in range(len(satellites)):
            for v in range(len(gs)):
                #print('NGEO to GS RF link\n')
                ####### GSL parameters
                eff = 0.55  # Efficiency of the parabolic antenna
                ####### NGEO to GS (feeder)
                f = 20e9    # Carrier frequency downlink (Hz)
                B = 500e6   # Maximum bandwidth
                maxPtx = 10     # Maximum tansmission power in W
                Adtx = 0.26 # Transmitter antenna diameter in m
                Adrx = 0.33     # Receiver antenna diameter in m
                pL = 0.3    # Pointing loss in dB
                Nf = 1.5    # Noise figure in dB
                Tn = 50     # Noise temperature in K
                min_elevation = 25  # minimum elevation angle
                self.links.append(RFLink(satellites[u], gs[v], f, B, maxPtx,Adtx, Adrx, pL, Nf, Tn, eff, "GSL", 0))
                self.rates_downlink[u,v] = self.links[-1].get_data_rate()
                

                f = 30e9    # Carrier frequency uplink (Hz)
                maxPtx = 20     # Maximum tansmission power in W
                Adtx = 0.33 # Transmitter antenna diameter in m
                Adrx = 0.26     # Receiver antenna diameter in m
                Nf = 2    # Noise figure in dB
                Tn = 290     # Noise temperature in K
                self.links.append(RFLink(gs[v],satellites[u], f, B, maxPtx,Adtx, Adrx, pL, Nf, Tn, eff,"GSL", 0))
                self.rates_uplink[u,v] = self.links[-1].get_data_rate()
    def plot_rates(self):
        plt.figure()
        plt.imshow(self.rates_downlink/1e9)
        plt.xlabel("Ground stations")
        plt.ylabel("Satellites")
        plt.title("Rates NGEO to GS [Gbps]")
        plt.figure()
        plt.imshow(self.rates_uplink/1e9)
        plt.xlabel("Ground stations")
        plt.ylabel("Satellites")
        plt.title("Rates GS to NGEO [Gbps]")

        
class RFLink():
    def  __init__(self, i, j,frequency, bandwidth, maxPtx, aDiameterTx, aDiameterRx, pointingLoss, noiseFigure, noiseTemperature,eff, link_type,min_elevation_angle=0): 
        self.power = maxPtx
        self.power_db = 10*math.log10(self.power)
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.Gtx = 10*math.log10(eff*((math.pi*aDiameterTx*self.frequency/c)**2))
        self.Grx = 10*math.log10(eff*((math.pi*aDiameterRx*self.frequency/c)**2))
        self.G =  self.Gtx+self.Grx - 2*pointingLoss
        self.No = 10*math.log10(self.bandwidth*k)+noiseFigure + 10* math.log10(290+(noiseTemperature-290) *(10**(-noiseFigure/10)))
        self.GoT = 10*math.log10(eff*((math.pi*aDiameterRx*self.frequency/c)**2))-noiseFigure - 10* math.log10(290+(noiseTemperature-290) *(10**(-noiseFigure/10)))
        self.endvertices = (i, j)
        self.slant_range= 0 
        self.in_range= False
        self.comm_range = 0
        self.active = False
        self.min_elevation = math.radians(min_elevation_angle)
        self.link_type = link_type
        self.get_slant_range()
        self.data_rate = 0
        self.path_loss = 0
        if self.in_range:
            self.get_data_rate()
        
    def  __repr__(self):
        return '\n%%%%%\nLink between Satellites {} and GS {}\nData rate = {} Mbps\n Power = {} W\n Max range = {} km \n Slant range = {} km'.format(
        self.endvertices[0].index,
        self.endvertices[1].id,
        self.data_rate/1e6,
        self.power,
        self.comm_range/1e3,
        self.slant_range/1e3,)
    def get_slant_range(self):
        self.slant_range=math.sqrt((self.endvertices[0].x-self.endvertices[1].x)**2 +(self.endvertices[0].y-self.endvertices[1].y)**2+(self.endvertices[0].z-self.endvertices[1].z)**2)	
        if self.link_type == "ISL":
            self.comm_range = math.sqrt(self.endvertices[0].h*(self.endvertices[0].h+2*Re))+math.sqrt(self.endvertices[1].h*(self.endvertices[1].h+2*Re))
        else:
            self.comm_range	=math.sqrt((Re+self.endvertices[0].h)**2-(Re*math.cos(self.min_elevation))**2)-Re*math.sin(self.min_elevation)
        self.in_range = self.comm_range>self.slant_range
    def get_data_rate(self):
        self.get_slant_range()
        self.path_loss_db = 10*math.log10((4*math.pi*self.slant_range*self.frequency/c)**2)
        self.snr = np.power(10,(self.power_db + self.G- self.path_loss_db - self.No )/10)  
        feasible_speffs = speff_thresholds[np.nonzero(lin_thresholds<=self.snr)]
        speff = feasible_speffs[-1]
        self.data_rate = self.bandwidth*speff
        return self.data_rate
    def establish(self):
        if self.in_range:
            self.active = True
            self.data_rate = self.module_rate * self.active
    def transmission(self, data_size, t):
        """
        Uses the data size, the current time, and the time when the last transmission was completed to calculate the time when the current transmission will be completed. Returns the time at which the transmission will be received
        
        """
        #print(self.data_rate)
        self.time_to_complete = max(t,self.time_to_complete) + data_size/self.data_rate    
        return self.time_to_complete + self.slant_range/c # Time when the transmission will be received 
    def transmission_time(self, data_size):
        transmission_time = data_size/self.data_rate + self.slant_range/c   
        return transmission_time          

    	       
def plot3D(sats,  gs=()):
    N = len(sats)
    N_gs = len(gs)
    Positions_sats = np.zeros((N,3)) 
    Positions_gs = np.zeros((N_gs,3))   
    meta_sats = np.zeros(N)
    for n in range(N):
        Positions_sats[n,:] = [sats[n].x/1e6, sats[n].y/1e6, sats[n].z/1e6]
        meta_sats[n] = sats[n].in_plane
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_box_aspect((np.ptp(Positions_sats[:,0]), np.ptp(Positions_sats[:,1]), np.ptp(Positions_sats[:,2])))
    area = math.pi * (5**2)
    ax.scatter(Positions_sats[:,0],Positions_sats[:,1], Positions_sats[:,2], c=meta_sats, s=area, label="Satellites")
#ax.set_aspect('equal', 'box')
    if N_gs:    
        Positions_gs = np.zeros((N_gs,3))
        for n in range(N_gs):
            Positions_gs[n,:]= [gs[n].x/1e6, gs[n].y/1e6, gs[n].z/1e6]
        ax.scatter(Positions_gs[:,0],Positions_gs[:,1], Positions_gs[:,2], s=area, marker="X", label="GSs")
        #print(Positions_gs)
    #ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
#ax.axis('off')
    return Positions_sats, Positions_gs      
    

        
