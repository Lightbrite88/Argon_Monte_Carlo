import matplotlib.pyplot as plt
import numpy as np
import random
import math
import sys
import time
import utils
from scipy import stats
from scipy.optimize import curve_fit
np.seterr(all='raise')
np.set_printoptions(threshold=sys.maxsize)
#comment to show on commit
"""
Create Your Own Direct Simulation Monte Carlo (With Python)
Author: Jeff Hatton (Science methodology by Sean Wagner)
Based on structure by Philip Mocz (2021) Princeton Univeristy, @PMocz
Simulate dilute gas with DSMC with 1:1 particle collisions
Setup: Open air cube of Argon
dimensionless units of m = sigma = k T0 = 1
"""

""" Direct Simulation Monte Carlo """

# Grab Currrent Time Before Running the Code
start = time.time()

#Shape description
#Pore
pore_coated_radius = 30 * 10 ** -9
gap_radius = pore_coated_radius + 4 * 10 ** -9
pore_height = 3000 * 10 ** -9
hot_coating_height = 30 * 10 ** -9
gap_height = hot_coating_height
cold_coating_height = pore_height - hot_coating_height - gap_height
hot_volume = utils.cylinder_volume(pore_coated_radius, hot_coating_height)
gap_volume = utils.cylinder_volume(gap_radius, gap_height)
cold_volume = utils.cylinder_volume(pore_coated_radius, cold_coating_height)
#Open air cold/hot
open_air_radius = 5 * pore_coated_radius
open_air_height = 100 * 10 ** -9              # metres
open_air_volume = utils.cylinder_volume(open_air_radius, open_air_height)
total_volume = hot_volume + gap_volume + cold_volume + open_air_volume*2 # metres^3
total_height = pore_height + open_air_height*2 # metres
#cell size
num_x_subdivions    = 27
num_y_subdivions    = 27
num_z_subdivions    = 550             
dx                  = open_air_radius/num_x_subdivions # cell x
dy                  = open_air_radius/num_y_subdivions # cell y
dz                  = total_height/num_z_subdivions # cell z - total_height/num_z_subdivions + overlap = ~ 2 particle diameters high
collision_x_overlap = dx/10
collision_y_overlap = dy/10
collision_z_overlap = dz/10

#Physics
argon_mass          = 6.63 * 10**-26        # kg
ar_molar_mass       = 0.039948              # Kg/mole
molecules_per_mole  = 6.02214179 * 10**23   # molecules per mole
ideal_gas_const     = 8.3145                # J/(mole*kelvin)
boltzman            = 1.38 * 10**(-23)      # m^2Kg/s^2K
temp_ambient        = 298                   # kelvin
sigma               = 3.6 * 10**(-19)       # 3.6*10^-19 m^2
argon_radius        = np.sqrt(sigma/(4*np.pi))  # 1.692568750643269 * 10^-10 m 
print(total_height/argon_radius)
collision_radius    = argon_radius*1        #consider increase by 15% for collision detection purposes ~1.946 * 10^-10 m
collision_range     = collision_radius*2    # ~3.89 * 10^-10 m
pressure            = 101325                # N/m^2
lambda_mfp          = boltzman*temp_ambient/(np.sqrt(2)*sigma*pressure) # ~79.7 nm mean free path
v_mean              = np.sqrt(3*ideal_gas_const*temp_ambient/ar_molar_mass) # mean speed
num_moles           = total_volume*pressure/(ideal_gas_const*temp_ambient)
a_shape             = np.sqrt(boltzman*temp_ambient/argon_mass) # argon boltzmann shaping factor
num_molecules       = np.round(num_moles * molecules_per_mole).astype(int)
print(num_molecules)
open_air_collision_radius = open_air_radius - argon_radius
gap_collision_radius = gap_radius - argon_radius
pore_collision_radius = pore_coated_radius - argon_radius

#Time
tau                 = lambda_mfp / v_mean   # mean-free time
Nmft                = 20                    # number of mean-free times to run simulation
#Change this value
NMFT_slice          = 1000                    #num timesteps per mean-free time (25 -> ~28 atom lengths per timestep) use 1000 max
num_timesteps       = Nmft*NMFT_slice       # number of time steps 
dt                  = Nmft*tau/num_timesteps# timestep

# Simulation
open_air_particles = np.floor(num_molecules * (open_air_volume/total_volume)).astype(int)
cold_pore_particles = np.floor(num_molecules * (cold_volume/total_volume)).astype(int)
hot_pore_particles = np.floor(num_molecules * (hot_volume/total_volume)).astype(int)
gap_particles = np.floor(num_molecules * (gap_volume/total_volume)).astype(int)
remaining_particles = num_molecules - gap_particles - hot_pore_particles - cold_pore_particles - open_air_particles*2
N                   = num_molecules     # number of sampling particles
Nsim                = 1         # number of simulations to run

# set the random number generator seed
np.random.seed(17)
random.seed(17) 

# prep figure
num_bins = 200
fig = plt.figure(figsize=(4*(Nsim+1),16))

# 3d speed -> velocity vector
def random_components(r):
    costheta = np.random.uniform(low=-1.0, high=1.0)
    phi = random.uniform(0,math.pi)
    theta = math.acos(costheta)
    Fx = r*math.cos(phi)*math.sin(theta)
    Fy = r*math.sin(phi)*math.sin(theta) * np.random.choice([-1,1])
    Fz = r*math.cos(theta)
    return Fx,Fy,Fz

#Intended curve for fitting (exponential decay)
def fit_exp_function(independant_variable, coeff_1, coeff_2):
    return coeff_1 * np.exp(coeff_2 * np.array(independant_variable))

#Intended curve for fitting (inverse)
def fit_inv_function(independant_variable, coeff_1, coeff_2, coeff_3):
    return coeff_1 * (independant_variable-coeff_2)**coeff_3

def init_positions():
    def x_func(radius, rand_rad, theta):
        return radius * np.sqrt(rand_rad) * math.cos(theta)

    def y_func(radius, rand_rad, theta):
        return radius * np.sqrt(rand_rad) * math.sin(theta)

    v_x_func = np.vectorize(x_func)
    v_y_func = np.vectorize(y_func)
    theta = np.random.uniform(0, 2*np.pi, num_molecules)
    rand_radius = np.random.uniform(0, 1, num_molecules)
    x_vals = np.zeros(num_molecules)
    y_vals = np.zeros(num_molecules)
    z_vals = np.zeros(num_molecules)
    #hot open air cylinder (bottom)
    x_vals[0:open_air_particles] = v_x_func(open_air_radius-argon_radius, rand_radius[0:open_air_particles], theta[0:open_air_particles])
    y_vals[0:open_air_particles] = v_y_func(open_air_radius-argon_radius, rand_radius[0:open_air_particles], theta[0:open_air_particles])
    z_vals[0:open_air_particles] = np.random.uniform(0+argon_radius, open_air_height-argon_radius, open_air_particles)
    #hot coating pore cylinder
    x_vals[open_air_particles:open_air_particles + hot_pore_particles] = v_x_func(pore_coated_radius-argon_radius, rand_radius[open_air_particles:open_air_particles + hot_pore_particles], theta[open_air_particles:open_air_particles + hot_pore_particles])
    y_vals[open_air_particles:open_air_particles + hot_pore_particles] = v_y_func(pore_coated_radius-argon_radius, rand_radius[open_air_particles:open_air_particles + hot_pore_particles], theta[open_air_particles:open_air_particles + hot_pore_particles])
    z_vals[open_air_particles:open_air_particles + hot_pore_particles] = np.random.uniform(open_air_height, open_air_height+hot_coating_height, hot_pore_particles)
    #gap cylinder
    x_vals[open_air_particles + hot_pore_particles:open_air_particles + hot_pore_particles + gap_particles] = v_x_func(gap_radius-argon_radius, rand_radius[open_air_particles + hot_pore_particles:open_air_particles + hot_pore_particles + gap_particles], theta[open_air_particles + hot_pore_particles:open_air_particles + hot_pore_particles + gap_particles])
    y_vals[open_air_particles + hot_pore_particles:open_air_particles + hot_pore_particles + gap_particles] = v_y_func(gap_radius-argon_radius, rand_radius[open_air_particles + hot_pore_particles:open_air_particles + hot_pore_particles + gap_particles], theta[open_air_particles + hot_pore_particles:open_air_particles + hot_pore_particles + gap_particles])
    z_vals[open_air_particles + hot_pore_particles:open_air_particles + hot_pore_particles + gap_particles] = np.random.uniform(open_air_height+hot_coating_height+argon_radius, open_air_height+hot_coating_height+gap_height-argon_radius, gap_particles)
    #cold coating cylinder
    x_vals[open_air_particles + hot_pore_particles + gap_particles:open_air_particles + hot_pore_particles + gap_particles + cold_pore_particles] = v_x_func(pore_coated_radius-argon_radius, rand_radius[open_air_particles + hot_pore_particles + gap_particles:open_air_particles + hot_pore_particles + gap_particles + cold_pore_particles], theta[open_air_particles + hot_pore_particles + gap_particles:open_air_particles + hot_pore_particles + gap_particles + cold_pore_particles])
    y_vals[open_air_particles + hot_pore_particles + gap_particles:open_air_particles + hot_pore_particles + gap_particles + cold_pore_particles] = v_y_func(pore_coated_radius-argon_radius, rand_radius[open_air_particles + hot_pore_particles + gap_particles:open_air_particles + hot_pore_particles + gap_particles + cold_pore_particles], theta[open_air_particles + hot_pore_particles + gap_particles:open_air_particles + hot_pore_particles + gap_particles + cold_pore_particles])
    z_vals[open_air_particles + hot_pore_particles + gap_particles:open_air_particles + hot_pore_particles + gap_particles + cold_pore_particles] = np.random.uniform(open_air_height+hot_coating_height+gap_height, open_air_height+hot_coating_height+gap_height+cold_coating_height, cold_pore_particles)
    #cold open air cylinder (top)
    x_vals[open_air_particles + hot_pore_particles + gap_particles + cold_pore_particles:] = v_x_func(open_air_radius-argon_radius, rand_radius[open_air_particles + hot_pore_particles + gap_particles + cold_pore_particles:], theta[open_air_particles + hot_pore_particles + gap_particles + cold_pore_particles:])
    y_vals[open_air_particles + hot_pore_particles + gap_particles + cold_pore_particles:] = v_y_func(open_air_radius-argon_radius, rand_radius[open_air_particles + hot_pore_particles + gap_particles + cold_pore_particles:], theta[open_air_particles + hot_pore_particles + gap_particles + cold_pore_particles:])
    z_vals[open_air_particles + hot_pore_particles + gap_particles + cold_pore_particles:] = np.random.uniform(open_air_height+hot_coating_height+gap_height+cold_coating_height+argon_radius, total_height-argon_radius, open_air_particles+remaining_particles)
    return x_vals,y_vals,z_vals

def init_velocities():
    # 3d directionless speeds for argon boltzmann distribution
    speeds = stats.maxwell.rvs(loc=0, scale=a_shape, size=num_molecules)  
    x_velocities = []
    y_velocities = []
    z_velocities = []
    # velocity vector components for 3d speeds 
    for v in speeds:
        #using spherical coordinates to cartesian
        x,y,z = random_components(v)
        x_velocities.append(x)
        y_velocities.append(y)
        z_velocities.append(z)
    x_velocities = np.array(x_velocities)
    y_velocities = np.array(y_velocities)
    z_velocities = np.array(z_velocities)
    return x_velocities, y_velocities, z_velocities

#free path tracking
dist_since_collision = np.zeros((Nsim, num_molecules))
dist_x_since_collision = np.zeros((Nsim, num_molecules))
dist_y_since_collision = np.zeros((Nsim, num_molecules))
dist_z_since_collision = np.zeros((Nsim, num_molecules))
completed_paths = [[] for _ in range(Nsim)]
completed_x_paths = [[] for _ in range(Nsim)]
completed_y_paths = [[] for _ in range(Nsim)]
completed_z_paths = [[] for _ in range(Nsim)]
full_path_traveled = np.zeros((Nsim, num_molecules), dtype=bool)

# Simulation Main Loop
for sim in range(Nsim):

    # Initialize positions
    x_vals,y_vals,z_vals = init_positions()
    prior_x_vals = np.zeros(num_molecules)
    prior_y_vals = np.zeros(num_molecules)
    prior_z_vals = np.zeros(num_molecules)

    x_velocities, y_velocities, z_velocities = init_velocities()            
    total_cols = 0
    total_errs = 0
    # Evolve
    for i in range(3):
        
        print('  timestep',i,'of',num_timesteps,'  (sim',sim+1,'/',Nsim,')')
        # collide particles using acceptance--rejection scheme
        N_collisions = 0
        
        # drift
        prior_x_vals = np.copy(x_vals)
        prior_y_vals = np.copy(y_vals)
        prior_z_vals = np.copy(z_vals)
        x_vals += dt*x_velocities
        y_vals += dt*y_velocities
        z_vals += dt*z_velocities
        #increase distance since last collision
        dist_since_collision[sim] += abs(np.sqrt(np.square(dt*x_velocities) + np.square(dt*y_velocities) + np.square(dt*z_velocities)))
        dist_x_since_collision[sim] += abs(dt*x_velocities)
        dist_y_since_collision[sim] += abs(dt*y_velocities)
        dist_z_since_collision[sim] += abs(dt*z_velocities)

        # CASE 1 
        # collide specular side of open air cylinder
        hit_side_open_air = np.sqrt(x_vals**2 + y_vals**2) > open_air_radius #true/false array
        x_positions_in_case = x_vals[hit_side_open_air]
        y_positions_in_case = y_vals[hit_side_open_air]
        x_velocities_in_case = x_velocities[hit_side_open_air]
        y_velocities_in_case = y_velocities[hit_side_open_air]
        z_velocities_in_case = z_velocities[hit_side_open_air]
        continue_path = dist_since_collision[sim][hit_side_open_air]
        continue_x_path = dist_x_since_collision[sim][hit_side_open_air]
        continue_y_path = dist_y_since_collision[sim][hit_side_open_air]
        continue_z_path = dist_z_since_collision[sim][hit_side_open_air]
        has_collided = full_path_traveled[sim][hit_side_open_air]
        num_particles_in_case = np.sum( hit_side_open_air )
        for p in range(num_particles_in_case):
            try:
                x, y, vx, vy, vz = x_positions_in_case[p], y_positions_in_case[p], x_velocities_in_case[p], y_velocities_in_case[p], z_velocities_in_case[p]
                a = (-vx)**2 + (-vy)**2
                b = 2 * (x * (-vx) + y * (-vy))
                c = x**2 + y**2 - open_air_collision_radius**2
                t = np.min([(-b + np.sqrt(b**2 - 4*a*c))/(2*a), (-b - np.sqrt(b**2 - 4*a*c))/(2*a)])
                col_x, col_y = x-vx*t, y-vy*t
                normal_vect = np.array([col_x, col_y])
                normalized_norm_vect = normal_vect/open_air_collision_radius
                vel_vect = np.array([vx, vy])
                scalar = np.dot(vel_vect, normalized_norm_vect)
                reflected_vel_vect = vel_vect - 2 * scalar * normalized_norm_vect
                new_vx, new_vy = reflected_vel_vect[0], reflected_vel_vect[1]
                new_x, new_y = col_x + new_vx*t, col_y + new_vy*t
                if has_collided[p]: # a full path has been observed
                    completed_paths[sim].append(abs(continue_path[p] - abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)))
                    completed_x_paths[sim].append(abs(continue_x_path[p] - abs(vx*t)))
                    completed_y_paths[sim].append(abs(continue_y_path[p] - abs(vy*t)))
                    completed_z_paths[sim].append(abs(continue_z_path[p] - abs(vz*t)))
                else: #this was the end of a partial path
                    has_collided[p] = True
                x_positions_in_case[p], y_positions_in_case[p], x_velocities_in_case[p], y_velocities_in_case[p] = new_x, new_y, new_vx, new_vy
                continue_path[p] = abs(np.sqrt(new_vx**2 + new_vy**2 + vz**2)*t)
                continue_x_path[p] = abs(new_vx*t)
                continue_y_path[p] = abs(new_vy*t)
                continue_z_path[p] = abs(vz*t)
            except:
                print([a, b, c])
                total_errs += 1
        x_vals[hit_side_open_air] = x_positions_in_case
        y_vals[hit_side_open_air] = y_positions_in_case
        x_velocities[hit_side_open_air] = x_velocities_in_case
        y_velocities[hit_side_open_air] = y_velocities_in_case
        dist_since_collision[sim][hit_side_open_air] = continue_path
        dist_x_since_collision[sim][hit_side_open_air] = continue_x_path
        dist_y_since_collision[sim][hit_side_open_air] = continue_y_path
        dist_z_since_collision[sim][hit_side_open_air] = continue_z_path
        full_path_traveled[sim][hit_side_open_air] = has_collided
        N_collisions += num_particles_in_case            

        # CASE 2
        # collide specular exterier vertical limits of described open air shape
        #top side
        hit_vertical_ext_open_air_min = z_vals < 0
        dt_ac = z_vals[hit_vertical_ext_open_air_min] / z_velocities[hit_vertical_ext_open_air_min] # time after collision
        x_velocities_in_case = x_velocities[hit_vertical_ext_open_air_min]
        y_velocities_in_case = y_velocities[hit_vertical_ext_open_air_min]
        z_velocities_in_case = z_velocities[hit_vertical_ext_open_air_min]
        continue_path = dist_since_collision[sim][hit_vertical_ext_open_air_min]
        continue_x_path = dist_x_since_collision[sim][hit_vertical_ext_open_air_min]
        continue_y_path = dist_y_since_collision[sim][hit_vertical_ext_open_air_min]
        continue_z_path = dist_z_since_collision[sim][hit_vertical_ext_open_air_min]
        has_collided = full_path_traveled[sim][hit_vertical_ext_open_air_min]
        num_particles_in_case = np.sum( hit_vertical_ext_open_air_min )
        for p in range(num_particles_in_case):
            t = dt_ac[p]
            vx, vy, vz = x_velocities_in_case[p], y_velocities_in_case[p], z_velocities_in_case[p]
            if has_collided[p]: # a full path has been observed
                completed_paths[sim].append(abs(continue_path[p] - abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)))
                completed_x_paths[sim].append(abs(continue_x_path[p] - abs(vx*t)))
                completed_y_paths[sim].append(abs(continue_y_path[p] - abs(vy*t)))
                completed_z_paths[sim].append(abs(continue_z_path[p] - abs(vz*t)))
            else: #this was the end of a partial path
                has_collided[p] = True
            continue_path[p] = abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)
            continue_x_path[p] = abs(vx*t)
            continue_y_path[p] = abs(vy*t)
            continue_z_path[p] = abs(vz*t)
        dist_since_collision[sim][hit_vertical_ext_open_air_min] = continue_path
        dist_x_since_collision[sim][hit_vertical_ext_open_air_min] = continue_x_path
        dist_y_since_collision[sim][hit_vertical_ext_open_air_min] = continue_y_path
        dist_z_since_collision[sim][hit_vertical_ext_open_air_min] = continue_z_path
        full_path_traveled[sim][hit_vertical_ext_open_air_min] = has_collided
        z_velocities[hit_vertical_ext_open_air_min] = -z_velocities[hit_vertical_ext_open_air_min]  # reverse normal component of velocity
        z_vals[hit_vertical_ext_open_air_min] = dt_ac * z_velocities[hit_vertical_ext_open_air_min]
        N_collisions += hit_vertical_ext_open_air_min.sum()

        #bottom side
        hit_vertical_ext_open_air_max = z_vals > total_height
        dt_ac = (z_vals[hit_vertical_ext_open_air_max]-total_height) / z_velocities[hit_vertical_ext_open_air_max] # time after collision
        x_velocities_in_case = x_velocities[hit_vertical_ext_open_air_max]
        y_velocities_in_case = y_velocities[hit_vertical_ext_open_air_max]
        z_velocities_in_case = z_velocities[hit_vertical_ext_open_air_max]
        continue_path = dist_since_collision[sim][hit_vertical_ext_open_air_max]
        continue_x_path = dist_x_since_collision[sim][hit_vertical_ext_open_air_max]
        continue_y_path = dist_y_since_collision[sim][hit_vertical_ext_open_air_max]
        continue_z_path = dist_z_since_collision[sim][hit_vertical_ext_open_air_max]
        has_collided = full_path_traveled[sim][hit_vertical_ext_open_air_max]
        num_particles_in_case = np.sum( hit_vertical_ext_open_air_max )
        for p in range(num_particles_in_case):
            t = dt_ac[p]
            vx, vy, vz = x_velocities_in_case[p], y_velocities_in_case[p], z_velocities_in_case[p]
            if has_collided[p]: # a full path has been observed
                completed_paths[sim].append(abs(continue_path[p] - abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)))
                completed_x_paths[sim].append(abs(continue_x_path[p] - abs(vx*t)))
                completed_y_paths[sim].append(abs(continue_y_path[p] - abs(vy*t)))
                completed_z_paths[sim].append(abs(continue_z_path[p] - abs(vz*t)))
            else: #this was the end of a partial path
                has_collided[p] = True
            continue_path[p] = abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)
            continue_x_path[p] = abs(vx*t)
            continue_y_path[p] = abs(vy*t)
            continue_z_path[p] = abs(vz*t)
        dist_since_collision[sim][hit_vertical_ext_open_air_max] = continue_path
        dist_x_since_collision[sim][hit_vertical_ext_open_air_max] = continue_x_path
        dist_y_since_collision[sim][hit_vertical_ext_open_air_max] = continue_y_path
        dist_z_since_collision[sim][hit_vertical_ext_open_air_max] = continue_z_path
        full_path_traveled[sim][hit_vertical_ext_open_air_max] = has_collided
        z_velocities[hit_vertical_ext_open_air_max] = -z_velocities[hit_vertical_ext_open_air_max]  # reverse normal component of velocity
        z_vals[hit_vertical_ext_open_air_max] = total_height + dt_ac * z_velocities[hit_vertical_ext_open_air_max]
        N_collisions += hit_vertical_ext_open_air_max.sum()      

        # CASE 3
        # collide specular interior vertical limits of described open air shape
        hit_vertical_int_open_air_cold = np.logical_and(prior_z_vals > total_height - open_air_height, np.logical_and(z_vals < total_height - open_air_height, np.sqrt(x_vals**2 + y_vals**2) > pore_coated_radius)) #cold side
        dt_ac = (z_vals[hit_vertical_int_open_air_cold]-(total_height - open_air_height)) / z_velocities[hit_vertical_int_open_air_cold] # time after collision
        x_velocities_in_case = x_velocities[hit_vertical_int_open_air_cold]
        y_velocities_in_case = y_velocities[hit_vertical_int_open_air_cold]
        z_velocities_in_case = z_velocities[hit_vertical_int_open_air_cold]
        continue_path = dist_since_collision[sim][hit_vertical_int_open_air_cold]
        continue_x_path = dist_x_since_collision[sim][hit_vertical_int_open_air_cold]
        continue_y_path = dist_y_since_collision[sim][hit_vertical_int_open_air_cold]
        continue_z_path = dist_z_since_collision[sim][hit_vertical_int_open_air_cold]
        has_collided = full_path_traveled[sim][hit_vertical_int_open_air_cold]
        num_particles_in_case = np.sum( hit_vertical_int_open_air_cold )
        for p in range(num_particles_in_case):
            t = dt_ac[p]
            vx, vy, vz = x_velocities_in_case[p], y_velocities_in_case[p], z_velocities_in_case[p]
            if has_collided[p]: # a full path has been observed
                completed_paths[sim].append(abs(continue_path[p] - abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)))
                completed_x_paths[sim].append(abs(continue_x_path[p] - abs(vx*t)))
                completed_y_paths[sim].append(abs(continue_y_path[p] - abs(vy*t)))
                completed_z_paths[sim].append(abs(continue_z_path[p] - abs(vz*t)))
            else: #this was the end of a partial path
                has_collided[p] = True
            continue_path[p] = abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)
            continue_x_path[p] = abs(vx*t)
            continue_y_path[p] = abs(vy*t)
            continue_z_path[p] = abs(vz*t)
        dist_since_collision[sim][hit_vertical_int_open_air_cold] = continue_path
        dist_x_since_collision[sim][hit_vertical_int_open_air_cold] = continue_x_path
        dist_y_since_collision[sim][hit_vertical_int_open_air_cold] = continue_y_path
        dist_z_since_collision[sim][hit_vertical_int_open_air_cold] = continue_z_path
        full_path_traveled[sim][hit_vertical_int_open_air_cold] = has_collided
        z_velocities[hit_vertical_int_open_air_cold] = -z_velocities[hit_vertical_int_open_air_cold]  # reverse normal component of velocity
        z_vals[hit_vertical_int_open_air_cold] = total_height - open_air_height + dt_ac * z_velocities[hit_vertical_int_open_air_cold]
        N_collisions += hit_vertical_int_open_air_cold.sum()
        
        hit_vertical_int_open_air_hot = np.logical_and(prior_z_vals < open_air_height, np.logical_and(z_vals > open_air_height, np.sqrt(x_vals**2 + y_vals**2) > pore_coated_radius)) #hot side
        dt_ac = (z_vals[hit_vertical_int_open_air_hot]-open_air_height) / z_velocities[hit_vertical_int_open_air_hot] # time after collision
        x_velocities_in_case = x_velocities[hit_vertical_int_open_air_hot]
        y_velocities_in_case = y_velocities[hit_vertical_int_open_air_hot]
        z_velocities_in_case = z_velocities[hit_vertical_int_open_air_hot]
        continue_path = dist_since_collision[sim][hit_vertical_int_open_air_hot]
        continue_x_path = dist_x_since_collision[sim][hit_vertical_int_open_air_hot]
        continue_y_path = dist_y_since_collision[sim][hit_vertical_int_open_air_hot]
        continue_z_path = dist_z_since_collision[sim][hit_vertical_int_open_air_hot]
        has_collided = full_path_traveled[sim][hit_vertical_int_open_air_hot]
        num_particles_in_case = np.sum( hit_vertical_int_open_air_hot )
        for p in range(num_particles_in_case):
            t = dt_ac[p]
            vx, vy, vz = x_velocities_in_case[p], y_velocities_in_case[p], z_velocities_in_case[p]
            if has_collided[p]: # a full path has been observed
                completed_paths[sim].append(abs(continue_path[p] - abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)))
                completed_x_paths[sim].append(abs(continue_x_path[p] - abs(vx*t)))
                completed_y_paths[sim].append(abs(continue_y_path[p] - abs(vy*t)))
                completed_z_paths[sim].append(abs(continue_z_path[p] - abs(vz*t)))
            else: #this was the end of a partial path
                has_collided[p] = True
            continue_path[p] = abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)
            continue_x_path[p] = abs(vx*t)
            continue_y_path[p] = abs(vy*t)
            continue_z_path[p] = abs(vz*t)
        dist_since_collision[sim][hit_vertical_int_open_air_hot] = continue_path
        dist_x_since_collision[sim][hit_vertical_int_open_air_hot] = continue_x_path
        dist_y_since_collision[sim][hit_vertical_int_open_air_hot] = continue_y_path
        dist_z_since_collision[sim][hit_vertical_int_open_air_hot] = continue_z_path
        full_path_traveled[sim][hit_vertical_int_open_air_hot] = has_collided
        z_velocities[hit_vertical_int_open_air_hot] = -z_velocities[hit_vertical_int_open_air_hot]  # reverse normal component of velocity
        z_vals[hit_vertical_int_open_air_hot] = open_air_height + dt_ac * z_velocities[hit_vertical_int_open_air_hot]
        N_collisions += hit_vertical_int_open_air_hot.sum()

        # #EDGE CASE: TODO: partical starts in open air, ends out of bounds (normally case 3), but went in to the pore and should have collided with the coating (normally case 4)
        # #currently not prioritized as corners may be slightly rounded realistically anyways

        # CASE 4
        # collide with gap interior wall
        hit_gap_cylinder_wall = np.logical_and( np.logical_and(prior_z_vals < total_height-open_air_height-cold_coating_height, prior_z_vals > open_air_height + hot_coating_height), 
                                                np.logical_and(np.sqrt(prior_x_vals**2 + prior_y_vals**2) < gap_radius, np.sqrt(x_vals**2 + y_vals**2) > gap_radius))
        x_positions_in_case = x_vals[hit_gap_cylinder_wall]
        y_positions_in_case = y_vals[hit_gap_cylinder_wall]
        x_velocities_in_case = x_velocities[hit_gap_cylinder_wall]
        y_velocities_in_case = y_velocities[hit_gap_cylinder_wall]
        z_velocities_in_case = z_velocities[hit_gap_cylinder_wall]
        continue_path = dist_since_collision[sim][hit_gap_cylinder_wall]
        continue_x_path = dist_x_since_collision[sim][hit_gap_cylinder_wall]
        continue_y_path = dist_y_since_collision[sim][hit_gap_cylinder_wall]
        continue_z_path = dist_z_since_collision[sim][hit_gap_cylinder_wall]
        has_collided = full_path_traveled[sim][hit_gap_cylinder_wall]
        num_particles_in_case = np.sum( hit_gap_cylinder_wall )
        for p in range(num_particles_in_case):
            try:
                x, y, vx, vy = x_positions_in_case[p], y_positions_in_case[p], x_velocities_in_case[p], y_velocities_in_case[p]
                a = (-vx)**2 + (-vy)**2
                b = 2 * (x * (-vx) + y * (-vy))
                c = x**2 + y**2 - gap_collision_radius**2
                t = np.min([(-b + np.sqrt(b**2 - 4*a*c))/(2*a), (-b - np.sqrt(b**2 - 4*a*c))/(2*a)])
                col_x, col_y = x-vx*t, y-vy*t
                normal_vect = np.array([col_x, col_y])
                normalized_norm_vect = normal_vect/gap_collision_radius
                vel_vect = np.array([vx, vy])
                scalar = np.dot(vel_vect, normalized_norm_vect)
                reflected_vel_vect = vel_vect - 2 * scalar * normalized_norm_vect
                new_vx, new_vy = reflected_vel_vect[0], reflected_vel_vect[1]
                new_x, new_y = col_x + new_vx*t, col_y + new_vy*t
                if has_collided[p]: # a full path has been observed
                    completed_paths[sim].append(abs(continue_path[p] - abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)))
                    completed_x_paths[sim].append(abs(continue_x_path[p] - abs(vx*t)))
                    completed_y_paths[sim].append(abs(continue_y_path[p] - abs(vy*t)))
                    completed_z_paths[sim].append(abs(continue_z_path[p] - abs(vz*t)))
                else: #this was the end of a partial path
                    has_collided[p] = True
                x_positions_in_case[p], y_positions_in_case[p], x_velocities_in_case[p], y_velocities_in_case[p] = new_x, new_y, new_vx, new_vy
                continue_path[p] = abs(np.sqrt(new_vx**2 + new_vy**2 + vz**2)*t)
                continue_x_path[p] = abs(new_vx*t)
                continue_y_path[p] = abs(new_vy*t)
                continue_z_path[p] = abs(vz*t)
            except:
                print([a, b, c])
                total_errs += 1
        x_vals[hit_gap_cylinder_wall] = x_positions_in_case
        y_vals[hit_gap_cylinder_wall] = y_positions_in_case
        x_velocities[hit_gap_cylinder_wall] = x_velocities_in_case
        y_velocities[hit_gap_cylinder_wall] = y_velocities_in_case
        dist_since_collision[sim][hit_gap_cylinder_wall] = continue_path
        dist_x_since_collision[sim][hit_gap_cylinder_wall] = continue_x_path
        dist_y_since_collision[sim][hit_gap_cylinder_wall] = continue_y_path
        dist_z_since_collision[sim][hit_gap_cylinder_wall] = continue_z_path
        full_path_traveled[sim][hit_gap_cylinder_wall] = has_collided
        N_collisions += num_particles_in_case

        # CASE 5
        # collide with top or bottom bases of gap cylinder
        hit_gap_cylinder_base_bottom = np.logical_and( np.logical_and( np.sqrt(prior_x_vals**2 + prior_y_vals**2) > pore_coated_radius, z_vals < open_air_height + hot_coating_height),
                                                np.logical_and( prior_z_vals < total_height-open_air_height-cold_coating_height, prior_z_vals > open_air_height + hot_coating_height))
        dt_ac = ((z_vals[hit_gap_cylinder_base_bottom])-(open_air_height + hot_coating_height)) / z_velocities[hit_gap_cylinder_base_bottom] # time after collision
        x_velocities_in_case = x_velocities[hit_gap_cylinder_base_bottom]
        y_velocities_in_case = y_velocities[hit_gap_cylinder_base_bottom]
        z_velocities_in_case = z_velocities[hit_gap_cylinder_base_bottom]
        continue_path = dist_since_collision[sim][hit_gap_cylinder_base_bottom]
        continue_x_path = dist_x_since_collision[sim][hit_gap_cylinder_base_bottom]
        continue_y_path = dist_y_since_collision[sim][hit_gap_cylinder_base_bottom]
        continue_z_path = dist_z_since_collision[sim][hit_gap_cylinder_base_bottom]
        has_collided = full_path_traveled[sim][hit_gap_cylinder_base_bottom]
        num_particles_in_case = np.sum( hit_gap_cylinder_base_bottom )
        for p in range(num_particles_in_case):
            t = dt_ac[p]
            vx, vy, vz = x_velocities_in_case[p], y_velocities_in_case[p], z_velocities_in_case[p]
            if has_collided[p]: # a full path has been observed
                completed_paths[sim].append(abs(continue_path[p] - abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)))
                completed_x_paths[sim].append(abs(continue_x_path[p] - abs(vx*t)))
                completed_y_paths[sim].append(abs(continue_y_path[p] - abs(vy*t)))
                completed_z_paths[sim].append(abs(continue_z_path[p] - abs(vz*t)))
            else: #this was the end of a partial path
                has_collided[p] = True
            continue_path[p] = abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)
            continue_x_path[p] = abs(vx*t)
            continue_y_path[p] = abs(vy*t)
            continue_z_path[p] = abs(vz*t)
        dist_since_collision[sim][hit_gap_cylinder_base_bottom] = continue_path
        dist_x_since_collision[sim][hit_gap_cylinder_base_bottom] = continue_x_path
        dist_y_since_collision[sim][hit_gap_cylinder_base_bottom] = continue_y_path
        dist_z_since_collision[sim][hit_gap_cylinder_base_bottom] = continue_z_path
        full_path_traveled[sim][hit_gap_cylinder_base_bottom] = has_collided
        z_velocities[hit_gap_cylinder_base_bottom] = -z_velocities[hit_gap_cylinder_base_bottom]  # reverse normal component of velocity
        z_vals[hit_gap_cylinder_base_bottom] = open_air_height + hot_coating_height + dt_ac * z_velocities[hit_gap_cylinder_base_bottom]
        N_collisions += hit_gap_cylinder_base_bottom.sum()

        hit_gap_cylinder_base_top = np.logical_and( np.logical_and( np.sqrt(prior_x_vals**2 + prior_y_vals**2) > pore_coated_radius, z_vals > total_height-open_air_height-cold_coating_height),
                                                np.logical_and( prior_z_vals < total_height-open_air_height-cold_coating_height, prior_z_vals > open_air_height + hot_coating_height))
        dt_ac = (z_vals[hit_gap_cylinder_base_top]-(total_height - open_air_height - cold_coating_height)) / z_velocities[hit_gap_cylinder_base_top] # time after collision
        x_velocities_in_case = x_velocities[hit_gap_cylinder_base_top]
        y_velocities_in_case = y_velocities[hit_gap_cylinder_base_top]
        z_velocities_in_case = z_velocities[hit_gap_cylinder_base_top]
        continue_path = dist_since_collision[sim][hit_gap_cylinder_base_top]
        continue_x_path = dist_x_since_collision[sim][hit_gap_cylinder_base_top]
        continue_y_path = dist_y_since_collision[sim][hit_gap_cylinder_base_top]
        continue_z_path = dist_z_since_collision[sim][hit_gap_cylinder_base_top]
        has_collided = full_path_traveled[sim][hit_gap_cylinder_base_top]
        num_particles_in_case = np.sum( hit_gap_cylinder_base_top )
        for p in range(num_particles_in_case):
            t = dt_ac[p]
            vx, vy, vz = x_velocities_in_case[p], y_velocities_in_case[p], z_velocities_in_case[p]
            if has_collided[p]: # a full path has been observed
                completed_paths[sim].append(abs(continue_path[p] - abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)))
                completed_x_paths[sim].append(abs(continue_x_path[p] - abs(vx*t)))
                completed_y_paths[sim].append(abs(continue_y_path[p] - abs(vy*t)))
                completed_z_paths[sim].append(abs(continue_z_path[p] - abs(vz*t)))
            else: #this was the end of a partial path
                has_collided[p] = True
            continue_path[p] = abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)
            continue_x_path[p] = abs(vx*t)
            continue_y_path[p] = abs(vy*t)
            continue_z_path[p] = abs(vz*t)
        dist_since_collision[sim][hit_gap_cylinder_base_top] = continue_path
        dist_x_since_collision[sim][hit_gap_cylinder_base_top] = continue_x_path
        dist_y_since_collision[sim][hit_gap_cylinder_base_top] = continue_y_path
        dist_z_since_collision[sim][hit_gap_cylinder_base_top] = continue_z_path
        full_path_traveled[sim][hit_gap_cylinder_base_top] = has_collided
        z_velocities[hit_gap_cylinder_base_top] = -z_velocities[hit_gap_cylinder_base_top]  # reverse normal component of velocity
        z_vals[hit_gap_cylinder_base_top] = total_height - open_air_height - cold_coating_height + dt_ac * z_velocities[hit_gap_cylinder_base_top]
        N_collisions += hit_gap_cylinder_base_top.sum()                                                

        # CASE 6
        # collide with coated pore wall
        hit_pore_coating = np.logical_and(  np.logical_and( np.sqrt(prior_x_vals**2 + prior_y_vals**2) < pore_coated_radius, np.sqrt(x_vals**2 + y_vals**2) > pore_coated_radius), #always necessary for a collision with the coating wall
                                            np.logical_or(  np.logical_and(z_vals < total_height-open_air_height, z_vals > total_height-open_air_height-cold_coating_height), #cold coating reflection - treated as specular in this simulation
                                                            np.logical_and(z_vals < open_air_height + hot_coating_height, z_vals > open_air_height))) #hot coating reflection - treated as specular in this simulation
        x_positions_in_case = x_vals[hit_pore_coating]
        y_positions_in_case = y_vals[hit_pore_coating]
        x_velocities_in_case = x_velocities[hit_pore_coating]
        y_velocities_in_case = y_velocities[hit_pore_coating]
        z_velocities_in_case = z_velocities[hit_pore_coating]
        continue_path = dist_since_collision[sim][hit_pore_coating]
        continue_x_path = dist_x_since_collision[sim][hit_pore_coating]
        continue_y_path = dist_y_since_collision[sim][hit_pore_coating]
        continue_z_path = dist_z_since_collision[sim][hit_pore_coating]
        has_collided = full_path_traveled[sim][hit_pore_coating]
        num_particles_in_case = np.sum( hit_pore_coating )
        for p in range(num_particles_in_case):
            try:
                x, y, vx, vy = x_positions_in_case[p], y_positions_in_case[p], x_velocities_in_case[p], y_velocities_in_case[p]
                a = (-vx)**2 + (-vy)**2
                b = 2 * (x * (-vx) + y * (-vy))
                c = x**2 + y**2 - pore_collision_radius**2
                t = np.min([(-b + np.sqrt(b**2 - 4*a*c))/(2*a), (-b - np.sqrt(b**2 - 4*a*c))/(2*a)])
                col_x, col_y = x-vx*t, y-vy*t
                normal_vect = np.array([col_x, col_y])
                normalized_norm_vect = normal_vect/pore_collision_radius
                vel_vect = np.array([vx, vy])
                scalar = np.dot(vel_vect, normalized_norm_vect)
                reflected_vel_vect = vel_vect - 2 * scalar * normalized_norm_vect
                new_vx, new_vy = reflected_vel_vect[0], reflected_vel_vect[1]
                new_x, new_y = col_x + new_vx*t, col_y + new_vy*t
                if has_collided[p]: # a full path has been observed
                    completed_paths[sim].append(abs(continue_path[p] - abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)))
                    completed_x_paths[sim].append(abs(continue_x_path[p] - abs(vx*t)))
                    completed_y_paths[sim].append(abs(continue_y_path[p] - abs(vy*t)))
                    completed_z_paths[sim].append(abs(continue_z_path[p] - abs(vz*t)))
                else: #this was the end of a partial path
                    has_collided[p] = True
                x_positions_in_case[p], y_positions_in_case[p], x_velocities_in_case[p], y_velocities_in_case[p] = new_x, new_y, new_vx, new_vy
                continue_path[p] = abs(np.sqrt(new_vx**2 + new_vy**2 + vz**2)*t)
                continue_x_path[p] = abs(new_vx*t)
                continue_y_path[p] = abs(new_vy*t)
                continue_z_path[p] = abs(vz*t)
            except:
                print([a, b, c])
                total_errs += 1
        x_vals[hit_pore_coating] = x_positions_in_case
        y_vals[hit_pore_coating] = y_positions_in_case
        x_velocities[hit_pore_coating] = x_velocities_in_case
        y_velocities[hit_pore_coating] = y_velocities_in_case
        dist_since_collision[sim][hit_pore_coating] = continue_path
        dist_x_since_collision[sim][hit_pore_coating] = continue_x_path
        dist_y_since_collision[sim][hit_pore_coating] = continue_y_path
        dist_z_since_collision[sim][hit_pore_coating] = continue_z_path
        full_path_traveled[sim][hit_pore_coating] = has_collided
        N_collisions += num_particles_in_case

        if (i%200 == 0):
            hit = np.sqrt(x_vals**2 + y_vals**2) > open_air_radius #true/false array
            print('             ',hit.sum(),' missed case 1\'s')
            hit = z_vals < 0
            print('             ',hit.sum(),' missed case 2a\'s')
            hit = z_vals > total_height
            print('             ',hit.sum(),' missed case 2b\'s')
            hit = np.logical_and(prior_z_vals > total_height - open_air_height, np.logical_and(z_vals < total_height - open_air_height, np.sqrt(x_vals**2 + y_vals**2) > pore_coated_radius)) #cold side
            print('             ',hit.sum(),' missed case 3a\'s')
            hit = np.logical_and(prior_z_vals < open_air_height, np.logical_and(z_vals > open_air_height, np.sqrt(x_vals**2 + y_vals**2) > pore_coated_radius)) #hot side
            print('             ',hit.sum(),' missed case 3b\'s')
            hit = np.logical_and( np.logical_and(prior_z_vals < total_height-open_air_height-cold_coating_height, prior_z_vals > open_air_height + hot_coating_height), 
                                                np.logical_and(np.sqrt(prior_x_vals**2 + prior_y_vals**2) < gap_radius, np.sqrt(x_vals**2 + y_vals**2) > gap_radius))
            print('             ',hit.sum(),' missed case 4\'s')
            hit = np.logical_and( np.logical_and( np.sqrt(prior_x_vals**2 + prior_y_vals**2) > pore_coated_radius, z_vals < open_air_height + hot_coating_height),
                                                np.logical_and( prior_z_vals < total_height-open_air_height-cold_coating_height, prior_z_vals > open_air_height + hot_coating_height))
            print('             ',hit.sum(),' missed case 5a\'s')
            hit = np.logical_and( np.logical_and( np.sqrt(prior_x_vals**2 + prior_y_vals**2) > pore_coated_radius, z_vals > total_height-open_air_height-cold_coating_height),
                                                np.logical_and( prior_z_vals < total_height-open_air_height-cold_coating_height, prior_z_vals > open_air_height + hot_coating_height))
            print('             ',hit.sum(),' missed case 5b\'s')
            hit = np.logical_and(  np.logical_and( np.sqrt(prior_x_vals**2 + prior_y_vals**2) < pore_coated_radius, np.sqrt(x_vals**2 + y_vals**2) > pore_coated_radius), #always necessary for a collision with the coating wall
                                            np.logical_or(  np.logical_and(z_vals < total_height-open_air_height, z_vals > total_height-open_air_height-cold_coating_height), #cold coating reflection - treated as specular in this simulation
                                                            np.logical_and(z_vals < open_air_height + hot_coating_height, z_vals > open_air_height))) #hot coating reflection - treated as specular in this simulation
            print('             ',hit.sum(),' missed case 6\'s')

        # loop over cells      TODO
        for x_layer in range(-num_x_subdivions, num_x_subdivions):
            in_x_layer = ((x_layer*dx - collision_x_overlap) < x_vals) & (x_vals < ((x_layer+1)*dx))
            for y_layer in range(-num_y_subdivions, num_y_subdivions):
                in_y_layer = ((y_layer*dy - collision_y_overlap) < y_vals) & (y_vals < ((y_layer+1)*dy))
                for z_layer in range(num_z_subdivions):         
                    in_z_layer = ((z_layer*dz  - collision_z_overlap) < z_vals) & (z_vals < ((z_layer+1)*dz))
                    in_cell = in_z_layer & in_x_layer & in_y_layer  #true false array of whether particle is in cell
                    num_particles_in_cell = np.sum( in_cell )

                    continue_path = dist_since_collision[sim][in_cell]
                    continue_x_path = dist_x_since_collision[sim][in_cell]
                    continue_y_path = dist_y_since_collision[sim][in_cell]
                    continue_z_path = dist_z_since_collision[sim][in_cell]
                    has_collided = full_path_traveled[sim][in_cell]
                    x_positions_in_cell = x_vals[in_cell]
                    y_positions_in_cell = y_vals[in_cell]
                    z_positions_in_cell = z_vals[in_cell]
                    x_velocities_in_cell = x_velocities[in_cell]
                    y_velocities_in_cell = y_velocities[in_cell]
                    z_velocities_in_cell = z_velocities[in_cell]

                    #detect collision - geometric distance less than 2*radius
                    for i in range(num_particles_in_cell): #for each particle
                        for j in range(i): #for each combination (not permutations) - don't compare pairs twice
                            if i != j: #ignore comparing particles to themselves
                                #calculate gemoetric distance between particles
                                x1, x2, y1, y2, z1, z2 = x_positions_in_cell[j], x_positions_in_cell[i], y_positions_in_cell[j], y_positions_in_cell[i], z_positions_in_cell[j], z_positions_in_cell[i]
                                particle_separation = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)                              
                                if particle_separation < collision_range: #collision detected
                                    vx1, vx2, vy1, vy2, vz1, vz2 = x_velocities_in_cell[j], x_velocities_in_cell[i], y_velocities_in_cell[j], y_velocities_in_cell[i], z_velocities_in_cell[j], z_velocities_in_cell[i]            

                                    # quadratic solve for t
                                    a = (-vx2+vx1)**2 + (-vy2+vy1)**2 + (-vz2+vz1)**2
                                    b = 2*((x2-x1)*(-vx2+vx1) + (y2-y1)*(-vy2+vy1) + (z2-z1)*(-vz2+vz1))
                                    c = (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 - collision_range**2
                                    t = np.max([(-b + np.sqrt(b**2 - 4*a*c))/(2*a), (-b - np.sqrt(b**2 - 4*a*c))/(2*a)])
                                    if has_collided[j]: # a full path has been observed
                                        completed_paths[sim].append(abs(continue_path[j] - abs(np.sqrt(vx1**2 + vy1**2 + vz1**2)*t)))
                                        completed_x_paths[sim].append(abs(continue_x_path[j] - abs(vx1*t)))
                                        completed_y_paths[sim].append(abs(continue_y_path[j] - abs(vy1*t)))
                                        completed_z_paths[sim].append(abs(continue_z_path[j] - abs(vz1*t)))
                                    else: #this was the end of a partial path
                                        has_collided[j] = True
                                    if has_collided[i]: # a full path has been observed
                                        completed_paths[sim].append(abs(continue_path[i] - abs(np.sqrt(vx2**2 + vy2**2 + vz2**2)*t)))
                                        completed_x_paths[sim].append(abs(continue_x_path[i] - abs(vx2*t)))
                                        completed_y_paths[sim].append(abs(continue_y_path[i] - abs(vy2*t)))
                                        completed_z_paths[sim].append(abs(continue_z_path[i] - abs(vz2*t)))
                                    else: #this was the end of a partial path
                                        has_collided[i] = True
                                
                                    #New positions pre collision (collision range apart)
                                    new_x1, new_y1, new_z1, new_x2, new_y2, new_z2 = x1-vx1*t, y1-vy1*t, z1-vz1*t, x2-vx2*t, y2-vy2*t, z2-vz2*t

                                    #FOR EACH DETECTED COLLISION
                                    
                                    #Find position vector 
                                    pos_vect = [new_x2-new_x1, new_y2-new_y1, new_z2-new_z1]
                                    #Create second var -> normalized position vector (unit vector)
                                    normalized_pos_vect = pos_vect/collision_range
                                    #P value for elastic collision
                                    p = (np.dot([vx1, vy1, vz1], normalized_pos_vect) - np.dot([vx2, vy2, vz2], normalized_pos_vect))/argon_mass
                                    #New velocities post collision
                                    new_vx1 = vx1 - p * argon_mass * normalized_pos_vect[0]; 
                                    new_vy1 = vy1 - p * argon_mass * normalized_pos_vect[1];
                                    new_vz1 = vz1 - p * argon_mass * normalized_pos_vect[2]; 
                                    new_vx2 = vx2 + p * argon_mass * normalized_pos_vect[0]; 
                                    new_vy2 = vy2 + p * argon_mass * normalized_pos_vect[1];
                                    new_vz2 = vz2 + p * argon_mass * normalized_pos_vect[2];
                                    #new positions post collision
                                    post_collision_position1 = [new_x1 + new_vx1*t, new_y1 + new_vy1*t, new_z1 + new_vz1*t]
                                    post_collision_position2 = [new_x2 + new_vx2*t, new_y2 + new_vy2*t, new_z2 + new_vz2*t]
                                    #update cell values
                                    x_positions_in_cell[j] = post_collision_position1[0]
                                    y_positions_in_cell[j] = post_collision_position1[1]
                                    z_positions_in_cell[j] = post_collision_position1[2]
                                    x_positions_in_cell[i] = post_collision_position2[0]
                                    y_positions_in_cell[i] = post_collision_position2[1]
                                    z_positions_in_cell[i] = post_collision_position2[2]
                                    x_velocities_in_cell[j] = new_vx1
                                    y_velocities_in_cell[j] = new_vy1
                                    z_velocities_in_cell[j] = new_vz1
                                    x_velocities_in_cell[i] = new_vx2
                                    y_velocities_in_cell[i] = new_vy2
                                    z_velocities_in_cell[i] = new_vz2
                                    continue_path[i] = abs(np.sqrt(new_vx2**2 + new_vy2**2 + new_vz2**2)*t)
                                    continue_path[j] = abs(np.sqrt(new_vx1**2 + new_vy1**2 + new_vz1**2)*t)
                                    continue_x_path[i] = abs(new_vx2*t)
                                    continue_z_path[i] = abs(new_vz2*t)
                                    continue_y_path[i] = abs(new_vy2*t)
                                    continue_x_path[j] = abs(new_vx1*t)
                                    continue_y_path[j] = abs(new_vy1*t)
                                    continue_z_path[j] = abs(new_vz1*t)
                                    N_collisions += 1
                            #return/replace new values from cell to total molecule list 
                            x_vals[in_cell] = x_positions_in_cell
                            y_vals[in_cell] = y_positions_in_cell
                            z_vals[in_cell] = z_positions_in_cell
                            x_velocities[in_cell] = x_velocities_in_cell
                            y_velocities[in_cell] = y_velocities_in_cell
                            z_velocities[in_cell] = z_velocities_in_cell
                            dist_since_collision[sim][in_cell] = continue_path
                            dist_x_since_collision[sim][in_cell] = continue_x_path
                            dist_y_since_collision[sim][in_cell] = continue_y_path
                            dist_z_since_collision[sim][in_cell] = continue_z_path
                            full_path_traveled[sim][in_cell] = has_collided
        total_cols += N_collisions
        print('    ',N_collisions,' collisions')
    print(' ',total_errs,' errors/warnings - potential lost particles')        
    print(' ',total_cols,' collisions')
    print(' ',len(completed_paths[sim]),' completed paths')
    #generate figure for graphing
    #Subplot for total distance
    ax1 = fig.add_subplot(4, Nsim, sim+1)
    data = completed_paths[sim]
    n_total, bins_total, patches_total = ax1.hist(data, range=(0,10**-6), bins=num_bins, density=True, color ='green', label='3d distance data') # color = numpy.random.rand(3,)
    #popt, pcov = curve_fit(fit_exp_function, bins_total[0:len(n_total)], n_total, p0=[14000000.0, -11000000.0], maxfev=25000) #po is first guess, maxfev is number of guesses
    #ax1.plot(bins_total[0:len(n_total)], fit_exp_function(bins_total[0:len(n_total)], *popt), 'r--', label='fit: a=%5.8f, b=%5.8f' % tuple(popt))
    ax1.set_xlabel('Path length before collision (m)')
    ax1.set_ylabel('Probability')
    ax1.legend()
    #Subplot for x distance
    ax2 = fig.add_subplot(4, Nsim, sim+2)
    x_data = completed_x_paths[sim]
    n_x, bins_x, patches_x = ax2.hist(x_data, range=(0,10**-6), bins=num_bins, density=True, color ='green', label='x data') # color = numpy.random.rand(3,)
    #popt_x, pcov_x = curve_fit(fit_inv_function, bins_x[0:len(n_x)], n_x, p0=[1.0, 0.0, -3.0], maxfev=25000) #po is first guess, maxfev is number of guesses
    # ax.plot(bins[0:len(n)], small_data_fit_function(bins[0:len(n)], *popt), 'r--', label='fit: a=%5.8f, b=%5.8f' % tuple(popt))
    #ax2.plot(bins_x[0:len(n_x)], fit_inv_function(bins_x[0:len(n_x)], *popt_x), 'r--', label='fit: a=%5.8f, b=%5.8f, c=%5.8f' % tuple(popt_x))
    ax2.set_xlabel('X Path length before collision (m)')
    ax2.set_ylabel('Probability')
    ax2.legend()
    #Subplot for y distance
    ax3 = fig.add_subplot(4, Nsim, sim+3)
    y_data = completed_y_paths[sim]
    n_y, bins_y, patches_y = ax3.hist(y_data, range=(0,10**-6), bins=num_bins, density=True, color ='green', label='y data') # color = numpy.random.rand(3,)
    #popt_y, pcov = curve_fit(fit_inv_function, bins_y[0:len(n_y)], n_y, p0=[1.0, 0.0, -3.0], maxfev=25000) #po is first guess, maxfev is number of guesses
    # ax.plot(bins[0:len(n)], small_data_fit_function(bins[0:len(n)], *popt), 'r--', label='fit: a=%5.8f, b=%5.8f' % tuple(popt))
    #ax3.plot(bins_y[0:len(n_y)], fit_inv_function(bins_y[0:len(n_y)], *popt_y), 'r--', label='fit: a=%5.8f, b=%5.8f, c=%5.8f' % tuple(popt_y))
    ax3.set_xlabel('Y Path length before collision (m)')
    ax3.set_ylabel('Probability')
    ax3.legend()
    #Subplot for z distance
    ax4 = fig.add_subplot(4, Nsim, sim+4)
    z_data = completed_z_paths[sim]
    n_z, bins_z, patches_z = ax4.hist(z_data, range=(0,10**-6), bins=num_bins, density=True, color ='green', label='z data') # color = numpy.random.rand(3,)  
    # popt_z, pcov = curve_fit(fit_inv_function, bins_z[0:len(n_z)], n_z, p0=[1.0, 0.0, -3.0], maxfev=25000) #po is first guess, maxfev is number of guesses
    # ax.plot(bins[0:len(n)], small_data_fit_function(bins[0:len(n)], *popt), 'r--', label='fit: a=%5.8f, b=%5.8f' % tuple(popt))
    #ax4.plot(bins_z[0:len(n_z)], fit_inv_function(bins_z[0:len(n_z)], *popt_z), 'r--', label='fit: a=%5.8f, b=%5.8f, c=%5.8f' % tuple(popt_z))
    ax4.set_xlabel('Z Path length before collision (m)')
    ax4.set_ylabel('Probability')
    ax4.legend()

# print mean free path
for sim in range(Nsim):
    print('Simulation '+ str(sim+1) + ' mean free path: ' + str(np.average(completed_paths[sim])))
    print('Simulation '+ str(sim+1) + ' mean x free path: ' + str(np.average(completed_x_paths[sim])))
    print('Simulation '+ str(sim+1) + ' mean y free path: ' + str(np.average(completed_y_paths[sim])))
    print('Simulation '+ str(sim+1) + ' mean z free path: ' + str(np.average(completed_z_paths[sim])))
    print('Num of collisions total: '+ str(len(completed_paths[sim])))

#save histogram data to text file
file_x = open("hist_x_axis_total_data.txt", "w")
file_x.write(str(bins_total[0:len(n_total)]))
file_x.close()
file_y = open("hist_y_axis_total_data.txt", "w")
file_y.write(str(n_total))
file_y.close()
file_x = open("hist_x_axis_x_data.txt", "w")
file_x.write(str(bins_x[0:len(n_x)]))
file_x.close()
file_y = open("hist_y_axis_x_data.txt", "w")
file_y.write(str(n_x))
file_y.close()
file_x = open("hist_x_axis_y_data.txt", "w")
file_x.write(str(bins_y[0:len(n_y)]))
file_x.close()
file_y = open("hist_y_axis_y_data.txt", "w")
file_y.write(str(n_y))
file_y.close()
file_x = open("hist_x_axis_z_data.txt", "w")
file_x.write(str(bins_z[0:len(n_z)]))
file_x.close()
file_y = open("hist_y_axis_z_data.txt", "w")
file_y.write(str(n_z))
file_y.close()

# Grab Currrent Time After Running the Code
end = time.time()
runtime = end - start
print( 'Runtime: '+ str(runtime/60.0) + ' minutes')

#show graph
plt.show()
