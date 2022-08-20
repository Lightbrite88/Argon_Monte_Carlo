import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils #local util functions

from functools import partial
from math import cos, sin, pi, acos
from mpmath import quad, exp
from multiprocessing import cpu_count, Manager, Value, Pool
from random import seed, uniform
from scipy.stats import maxwell
from sys import maxsize
from time import time

np.seterr(all='raise')
np.set_printoptions(threshold=maxsize)


"""
Create Your Own Direct Simulation Monte Carlo (With Python)
Author: Jeff Hatton (Science methodology by Sean Wagner)
Based on structure by Philip Mocz (2021) Princeton Univeristy, @PMocz
Simulate dilute gas with DSMC with 1:1 particle collisions
Setup: Open air pore with Argon
"""

""" Direct Simulation Monte Carlo """

# Grab Currrent Time Before Running the Code
start                       = time()

#Shape description
#Pore
pore_coated_radius          = 30 * 10 ** -9
gap_radius                  = pore_coated_radius + 4 * 10 ** -9
pore_height                 = 3000 * 10 ** -9
hot_coating_height          = 30 * 10 ** -9
gap_height                  = hot_coating_height
cold_coating_height         = pore_height - hot_coating_height - gap_height
hot_volume                  = utils.cylinder_volume(pore_coated_radius, hot_coating_height)
gap_volume                  = utils.cylinder_volume(gap_radius, gap_height)
cold_volume                 = utils.cylinder_volume(pore_coated_radius, cold_coating_height)
#Open air cold/hot
open_air_radius             = 5 * pore_coated_radius
open_air_height             = 100 * 10 ** -9              # metres
open_air_volume             = utils.cylinder_volume(open_air_radius, open_air_height)
total_volume                = hot_volume + gap_volume + cold_volume + open_air_volume*2 # metres^3
total_height                = pore_height + open_air_height*2 # metres
gap_bottom_height           = open_air_height + hot_coating_height
gap_top_height              = open_air_height + hot_coating_height + gap_height
#cell size
num_x_subdivions            = 7
num_y_subdivions            = 7
num_z_subdivions            = 148
dx                          = open_air_radius/num_x_subdivions # cell x
dy                          = open_air_radius/num_y_subdivions # cell y
dz                          = total_height/num_z_subdivions # cell z - total_height/num_z_subdivions + overlap = ~ 2 particle diameters high

#Physics
argon_mass                  = 6.63 * 10**-26        # kg
ar_molar_mass               = 0.039948              # Kg/mole
molecules_per_mole          = 6.02214179 * 10**23   # molecules per mole
ideal_gas_const             = 8.3145                # J/(mole*kelvin)
boltzman                    = 1.38064852 * 10**(-23)      # m^2Kg/s^2K
temp_ambient                = 298.0                   # kelvin
sigma                       = 3.6 * 10**(-19)       # 3.6*10^-19 m^2
argon_radius                = np.sqrt(sigma/(4*np.pi))  # 1.692568750643269 * 10^-10 m 
collision_radius            = argon_radius*1        #consider increase by 15% for collision detection purposes ~1.946 * 10^-10 m
collision_range             = collision_radius*2    # ~3.89 * 10^-10 m
pressure                    = 101325                # N/m^2
lambda_mfp                  = boltzman*temp_ambient/(np.sqrt(2)*sigma*pressure) # ~79.7 nm mean free path
v_mean                      = np.sqrt(3*ideal_gas_const*temp_ambient/ar_molar_mass) # mean speed
num_moles                   = total_volume*pressure/(ideal_gas_const*temp_ambient)
a_shape                     = np.sqrt(boltzman*temp_ambient/argon_mass) # argon boltzmann shaping factor
num_molecules               = np.round(num_moles * molecules_per_mole).astype(int)
t_cold                      = 293.0                 # kelvin
t_hot                       = 353.0                 # kelvin
t_debye_graphene            = 1813.0                # kelvin
t_debye_alumina             = 980.0                # kelvin
coated_accomodation_coeff   = 0.95 # graphene
gap_accomodation_coeff      = 0.8 # alumina
num_atoms_unitcell_graphene = 2
num_atoms_unitcell_alumina  = 10
debye_integrand             = lambda x: (x**3)/(exp(x)-1)
debye_quadrature_cold       = quad( debye_integrand, [0 , t_debye_graphene/t_cold] )
debye_quadrature_hot        = quad( debye_integrand, [0 , t_debye_graphene/t_hot] )
surface_energy_cold         = 9 * t_cold * num_atoms_unitcell_graphene * boltzman * (t_cold/t_debye_graphene)**3 * debye_quadrature_cold
surface_energy_hot          = 9 * t_hot * num_atoms_unitcell_graphene * boltzman * (t_hot/t_debye_graphene)**3 * debye_quadrature_hot

#Keep values in bounds
open_air_collision_radius   = open_air_radius - argon_radius
gap_collision_radius        = gap_radius - argon_radius
pore_collision_radius       = pore_coated_radius - argon_radius

#Time
tau                         = lambda_mfp / v_mean   # mean-free time
nmft                        = 20                    # number of mean-free times to run simulation
nmft_slice                  = 1000                  # num timesteps per mean-free time (25 -> ~28 atom lengths per timestep) use 1000 max (3/4 of an atom length per timestep)
num_timesteps               = nmft*nmft_slice       # number of time steps 
dt                          = nmft*tau/num_timesteps# timestep

# Simulation
open_air_particles          = np.floor(num_molecules * (open_air_volume/total_volume)).astype(int)
cold_pore_particles         = np.floor(num_molecules * (cold_volume/total_volume)).astype(int)
hot_pore_particles          = np.floor(num_molecules * (hot_volume/total_volume)).astype(int)
gap_particles               = np.floor(num_molecules * (gap_volume/total_volume)).astype(int)
remaining_particles         = num_molecules - gap_particles - hot_pore_particles - cold_pore_particles - open_air_particles*2
num_sims                    = 1                     # number of simulations to run
num_workers                 = cpu_count() + 1

# set the random number generator seed
np.random.seed(17)
seed(17) 

# prep figure
num_bins            = 200
fig                 = plt.figure(figsize=(4*(num_sims+1),16))

def cylinder_volume(radius, height):
    return np.pi * radius ** 2 * height

# 3d speed -> velocity vector
def random_components(r):
    costheta = np.random.uniform(low=-1.0, high=1.0)
    phi = uniform(0,pi)
    theta = acos(costheta)
    Fx = r*cos(phi)*sin(theta)
    Fy = r*sin(phi)*sin(theta) * np.random.choice([-1,1])
    Fz = r*cos(theta)
    return Fx,Fy,Fz

def kinetic_energy(velocity_magnitude):
    return 0.5 * argon_mass * velocity_magnitude**2

# perturbs a vector by up to 85 degrees
def random_inbounds_direction(norm):
    while True:
        nx,ny,nz = random_components(1)
        new_direction = np.array([nx,ny,nz])
        if abs(np.dot(new_direction, norm)) < cos(85*pi/180):
            continue
        if np.dot(new_direction, norm) < cos(85*pi/180):
            new_direction = -new_direction
        break
    return new_direction

def gap_surface_temperature(z_value):
    m = (t_cold - t_hot)/(gap_height)
    return m*(z_value-gap_bottom_height) + t_hot

def debye_quadrature_gap(z_value):
    return quad( debye_integrand, [0 , t_debye_alumina/gap_surface_temperature(z_value)] )

def surface_energy_gap(z_value):
    t_gap = gap_surface_temperature(z_value)
    return 9 * t_gap * num_atoms_unitcell_alumina * boltzman * (t_gap/t_debye_alumina)**3 * debye_quadrature_gap(z_value)

def init_positions():
    def x_func(radius, rand_rad, theta):
        return radius * np.sqrt(rand_rad) * cos(theta)

    def y_func(radius, rand_rad, theta):
        return radius * np.sqrt(rand_rad) * sin(theta)

    v_x_func = np.vectorize(x_func)
    v_y_func = np.vectorize(y_func)
    theta = np.random.uniform(0, 2*np.pi, num_molecules)
    rand_radius = np.random.uniform(0, 1, num_molecules)
    x_vals = np.zeros(num_molecules)
    y_vals = np.zeros(num_molecules)
    z_vals = np.zeros(num_molecules)
    hot_start_index = open_air_particles
    hot_end_index = hot_start_index + hot_pore_particles
    gap_start_index = hot_end_index 
    gap_end_index = gap_start_index + gap_particles
    cold_starting_index = gap_end_index
    cold_end_index = cold_starting_index + cold_pore_particles
    cold_open_air_start_index = cold_end_index
    #hot open air cylinder (bottom)
    x_vals[0:open_air_particles] = v_x_func(open_air_radius-argon_radius, rand_radius[0:open_air_particles], theta[0:open_air_particles])
    y_vals[0:open_air_particles] = v_y_func(open_air_radius-argon_radius, rand_radius[0:open_air_particles], theta[0:open_air_particles])
    z_vals[0:open_air_particles] = np.random.uniform(0+argon_radius, open_air_height-argon_radius, open_air_particles)
    #hot coating pore cylinder
    x_vals[hot_start_index:hot_end_index] = v_x_func(pore_coated_radius-argon_radius, rand_radius[hot_start_index:hot_end_index], theta[hot_start_index:hot_end_index])
    y_vals[hot_start_index:hot_end_index] = v_y_func(pore_coated_radius-argon_radius, rand_radius[hot_start_index:hot_end_index], theta[hot_start_index:hot_end_index])
    z_vals[hot_start_index:hot_end_index] = np.random.uniform(open_air_height, open_air_height+hot_coating_height, hot_pore_particles)
    #gap cylinder
    x_vals[gap_start_index:gap_end_index] = v_x_func(gap_radius-argon_radius, rand_radius[gap_start_index:gap_end_index], theta[gap_start_index:gap_end_index])
    y_vals[gap_start_index:gap_end_index] = v_y_func(gap_radius-argon_radius, rand_radius[gap_start_index:gap_end_index], theta[gap_start_index:gap_end_index])
    z_vals[gap_start_index:gap_end_index] = np.random.uniform(gap_bottom_height+argon_radius, gap_top_height-argon_radius, gap_particles)
    #cold coating cylinder
    x_vals[cold_starting_index:cold_end_index] = v_x_func(pore_coated_radius-argon_radius, rand_radius[cold_starting_index:cold_end_index], theta[cold_starting_index:cold_end_index])
    y_vals[cold_starting_index:cold_end_index] = v_y_func(pore_coated_radius-argon_radius, rand_radius[cold_starting_index:cold_end_index], theta[cold_starting_index:cold_end_index])
    z_vals[cold_starting_index:cold_end_index] = np.random.uniform(open_air_height+hot_coating_height+gap_height, open_air_height+hot_coating_height+gap_height+cold_coating_height, cold_pore_particles)
    #cold open air cylinder (top)
    x_vals[cold_open_air_start_index:] = v_x_func(open_air_radius-argon_radius, rand_radius[cold_open_air_start_index:], theta[cold_open_air_start_index:])
    y_vals[cold_open_air_start_index:] = v_y_func(open_air_radius-argon_radius, rand_radius[cold_open_air_start_index:], theta[cold_open_air_start_index:])
    z_vals[cold_open_air_start_index:] = np.random.uniform(open_air_height+hot_coating_height+gap_height+cold_coating_height+argon_radius, total_height-argon_radius, open_air_particles+remaining_particles)
    return x_vals,y_vals,z_vals

def init_velocities():
    # 3d directionless speeds for argon boltzmann distribution
    speeds = maxwell.rvs(loc=0, scale=a_shape, size=num_molecules)  
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

def pairwise_particles_in_cell(completed_paths, completed_x_paths, completed_y_paths, completed_z_paths, in_cell, continue_path, continue_x_path, continue_y_path, continue_z_path, has_collided, x_positions_in_cell, y_positions_in_cell, z_positions_in_cell, x_velocities_in_cell, y_velocities_in_cell, z_velocities_in_cell):
    num_collisions_in_cell = 0
    num_particles_in_cell = np.sum( in_cell )
    completed_paths_temp = []
    completed_x_paths_temp = []
    completed_y_paths_temp = []
    completed_z_paths_temp = []
    for i in range(num_particles_in_cell): #for each particle
        for j in range(i): #for each combination (not permutations) - don't compare pairs twice
            if i != j: #ignore comparing particles to themselves
                #calculate gemoetric distance between particles
                x1, x2, y1, y2, z1, z2 = x_positions_in_cell[j], x_positions_in_cell[i], y_positions_in_cell[j], y_positions_in_cell[i], z_positions_in_cell[j], z_positions_in_cell[i]
                particle_separation = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)                              
                if particle_separation < collision_range: #collision detected

                    #FOR EACH DETECTED COLLISION

                    #pull both particle's values for readability
                    vx1, vx2, vy1, vy2, vz1, vz2 = x_velocities_in_cell[j], x_velocities_in_cell[i], y_velocities_in_cell[j], y_velocities_in_cell[i], z_velocities_in_cell[j], z_velocities_in_cell[i]            

                    # quadratic solve for t
                    a = (-vx2+vx1)**2 + (-vy2+vy1)**2 + (-vz2+vz1)**2
                    b = 2*((x2-x1)*(-vx2+vx1) + (y2-y1)*(-vy2+vy1) + (z2-z1)*(-vz2+vz1))
                    c = (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 - collision_range**2
                    t = np.max([(-b + np.sqrt(b**2 - 4*a*c))/(2*a), (-b - np.sqrt(b**2 - 4*a*c))/(2*a)])
                    if has_collided[j]: # a full path has been observed
                        completed_paths_temp.append(abs(continue_path[j] - abs(np.sqrt(vx1**2 + vy1**2 + vz1**2)*t)))
                        completed_x_paths_temp.append(abs(continue_x_path[j] - abs(vx1*t)))
                        completed_y_paths_temp.append(abs(continue_y_path[j] - abs(vy1*t)))
                        completed_z_paths_temp.append(abs(continue_z_path[j] - abs(vz1*t)))
                    else: #this was the end of a partial path
                        has_collided[j] = True
                    if has_collided[i]: # a full path has been observed
                        completed_paths_temp.append(abs(continue_path[i] - abs(np.sqrt(vx2**2 + vy2**2 + vz2**2)*t)))
                        completed_x_paths_temp.append(abs(continue_x_path[i] - abs(vx2*t)))
                        completed_y_paths_temp.append(abs(continue_y_path[i] - abs(vy2*t)))
                        completed_z_paths_temp.append(abs(continue_z_path[i] - abs(vz2*t)))
                    else: #this was the end of a partial path
                        has_collided[i] = True
                
                    #New positions pre collision (collision range apart)
                    new_x1, new_y1, new_z1, new_x2, new_y2, new_z2 = x1-vx1*t, y1-vy1*t, z1-vz1*t, x2-vx2*t, y2-vy2*t, z2-vz2*t
                    
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
                    num_collisions_in_cell += 1

    #update global num_collisions per timestep
    with num_collisions_per_step.get_lock():
        num_collisions_per_step.value+=num_collisions_in_cell
        
    # Extend shared lists by noted completed full paths
    completed_paths.extend(completed_paths_temp)
    completed_x_paths.extend(completed_x_paths_temp)
    completed_y_paths.extend(completed_y_paths_temp)
    completed_z_paths.extend(completed_z_paths_temp)

    # Return new particle values to main arrays
    return in_cell, continue_path, continue_x_path, continue_y_path, continue_z_path, has_collided, x_positions_in_cell, \
        y_positions_in_cell, z_positions_in_cell, x_velocities_in_cell, y_velocities_in_cell, z_velocities_in_cell

def hit_vertical_specular_wall(hits, z_plane):
    global z_vals, z_velocities
    dt_ac = (z_vals[hits]-z_plane) / z_velocities[hits] # time after collision
    z_velocities[hits] = -z_velocities[hits]  # reverse normal component of velocity
    z_vals[hits] = z_plane + dt_ac * z_velocities[hits]

def hit_cylinder_specular_side_wall(hits, collision_radius, total_errs):    
    global x_vals, y_vals, x_velocities, y_velocities
    x_positions_in_case = x_vals[hits]
    y_positions_in_case = y_vals[hits]
    x_velocities_in_case = x_velocities[hits]
    y_velocities_in_case = y_velocities[hits]
    num_particles_in_case = np.sum( hits )
    for p in range(num_particles_in_case):
        try:
            x, y, vx, vy = x_positions_in_case[p], y_positions_in_case[p], x_velocities_in_case[p], y_velocities_in_case[p]
            a = (-vx)**2 + (-vy)**2
            b = 2 * (x * (-vx) + y * (-vy))
            c = x**2 + y**2 - collision_radius**2
            t = np.min([(-b + np.sqrt(b**2 - 4*a*c))/(2*a), (-b - np.sqrt(b**2 - 4*a*c))/(2*a)])
            col_x, col_y = x-vx*t, y-vy*t
            normal_vect = np.array([col_x, col_y])
            normalized_norm_vect = normal_vect/collision_radius
            vel_vect = np.array([vx, vy])
            scalar = np.dot(vel_vect, normalized_norm_vect)
            reflected_vel_vect = vel_vect - 2 * scalar * normalized_norm_vect
            new_vx, new_vy = reflected_vel_vect[0], reflected_vel_vect[1]
            new_x, new_y = col_x + new_vx*t, col_y + new_vy*t
            x_positions_in_case[p], y_positions_in_case[p], x_velocities_in_case[p], y_velocities_in_case[p] = new_x, new_y, new_vx, new_vy
        except:
            print([a, b, c])
            total_errs += 1
    x_vals[hits] = x_positions_in_case
    y_vals[hits] = y_positions_in_case
    x_velocities[hits] = x_velocities_in_case
    y_velocities[hits] = y_velocities_in_case
    return total_errs

def hit_vertical_coated_wall(hits, surface_energy, z_plane, inbounds_direction, completed_paths, completed_x_paths, completed_y_paths, completed_z_paths):
    global x_vals, y_vals, z_vals, x_velocities, y_velocities, z_velocities, full_path_traveled
    global dist_since_collision, dist_x_since_collision, dist_y_since_collision, dist_z_since_collision
    global num_collisions_per_step
    dt_ac = (z_vals[hits]-z_plane) / z_velocities[hits] # time after collision
    x_positions_in_case = x_vals[hits]
    y_positions_in_case = y_vals[hits]
    z_positions_in_case = z_vals[hits]
    x_velocities_in_case = x_velocities[hits]
    y_velocities_in_case = y_velocities[hits]
    z_velocities_in_case = z_velocities[hits]
    continue_path = dist_since_collision[hits]
    continue_x_path = dist_x_since_collision[hits]
    continue_y_path = dist_y_since_collision[hits]
    continue_z_path = dist_z_since_collision[hits]
    has_collided = full_path_traveled[hits]
    num_particles_in_case = np.sum( hits )
    momentem_z_change_in_case = 0
    energy_change_in_case = 0
    for p in range(num_particles_in_case):
        t = dt_ac[p]
        x, y, z = x_positions_in_case[p], y_positions_in_case[p], z_positions_in_case[p]
        vx, vy, vz = x_velocities_in_case[p], y_velocities_in_case[p], z_velocities_in_case[p]
        col_x, col_y, col_z = x-vx*t, y-vy*t, z_plane

        normalized_norm_vect = np.array([0, 0, inbounds_direction])
        new_velocity_vector = random_inbounds_direction(normalized_norm_vect)

        v_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
        old_z_momentum = argon_mass * vz
        particle_energy = kinetic_energy(v_magnitude)
        energy_difference = surface_energy - particle_energy
        new_particle_energy = particle_energy + energy_difference*coated_accomodation_coeff
        unit_vx, unit_vy, unit_vz = new_velocity_vector[0], new_velocity_vector[1], new_velocity_vector[2]
        new_velocity_magnitude = np.sqrt(new_particle_energy * 2 / argon_mass)
        energy_transfer = new_particle_energy - particle_energy
        energy_change_in_case += energy_transfer
        new_vx, new_vy, new_vz = unit_vx*new_velocity_magnitude, unit_vy*new_velocity_magnitude, unit_vz*new_velocity_magnitude
        new_z_momentum = argon_mass * new_vz
        momentum_z_difference = new_z_momentum - old_z_momentum
        momentem_z_change_in_case += momentum_z_difference

        if has_collided[p]: # a full path has been observed
            completed_paths.append(abs(continue_path[p] - abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)))
            completed_x_paths.append(abs(continue_x_path[p] - abs(vx*t)))
            completed_y_paths.append(abs(continue_y_path[p] - abs(vy*t)))
            completed_z_paths.append(abs(continue_z_path[p] - abs(vz*t)))
        else: #this was the end of a partial path
            has_collided[p] = True
        continue_path[p] = 0
        continue_x_path[p] = 0
        continue_y_path[p] = 0
        continue_z_path[p] = 0
        x_positions_in_case[p], y_positions_in_case[p], z_positions_in_case[p] = col_x, col_y, col_z
        x_velocities_in_case[p], y_velocities_in_case[p], z_velocities_in_case[p] = new_vx, new_vy, new_vz
    x_vals[hits], y_vals[hits], z_vals[hits] = x_positions_in_case, y_positions_in_case, z_positions_in_case
    x_velocities[hits], y_velocities[hits], z_velocities[hits] = x_velocities_in_case, y_velocities_in_case, z_velocities_in_case   
    dist_since_collision[hits] = continue_path
    dist_x_since_collision[hits] = continue_x_path
    dist_y_since_collision[hits] = continue_y_path
    dist_z_since_collision[hits] = continue_z_path
    full_path_traveled[hits] = has_collided
    num_collisions_per_step.value += hits.sum()
    return momentem_z_change_in_case, energy_change_in_case

def hit_cylinder_coated_side_wall(hits, surface_energy, collision_radius, completed_paths, completed_x_paths, completed_y_paths, completed_z_paths, total_errs):    
    global x_vals, y_vals, z_vals, x_velocities, y_velocities, z_velocities, full_path_traveled
    global dist_since_collision, dist_x_since_collision, dist_y_since_collision, dist_z_since_collision
    global num_collisions_per_step
    x_positions_in_case = x_vals[hits]
    y_positions_in_case = y_vals[hits]
    z_positions_in_case = z_vals[hits]
    x_velocities_in_case = x_velocities[hits]
    y_velocities_in_case = y_velocities[hits]
    z_velocities_in_case = z_velocities[hits]
    continue_path = dist_since_collision[hits]
    continue_x_path = dist_x_since_collision[hits]
    continue_y_path = dist_y_since_collision[hits]
    continue_z_path = dist_z_since_collision[hits]
    has_collided = full_path_traveled[hits]
    num_particles_in_case = np.sum( hits )
    momentem_z_change_in_case = 0
    energy_change_in_case = 0
    for p in range(num_particles_in_case):
        try:
            x, y, z = x_positions_in_case[p], y_positions_in_case[p], z_positions_in_case[p]
            vx, vy, vz = x_velocities_in_case[p], y_velocities_in_case[p], z_velocities_in_case[p]
            a = (-vx)**2 + (-vy)**2
            b = 2 * (x * (-vx) + y * (-vy))
            c = x**2 + y**2 - collision_radius**2
            t = np.min([(-b + np.sqrt(b**2 - 4*a*c))/(2*a), (-b - np.sqrt(b**2 - 4*a*c))/(2*a)])
            col_x, col_y, col_z = x-vx*t, y-vy*t, z-vz*t

            normal_vect = np.array([col_x, col_y, 0])
            normalized_norm_vect = normal_vect/collision_radius
            new_velocity_vector = random_inbounds_direction(-normalized_norm_vect)

            v_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
            old_z_momentum = argon_mass * vz
            particle_energy = kinetic_energy(v_magnitude)
            energy_difference = surface_energy - particle_energy
            new_particle_energy = particle_energy + energy_difference*coated_accomodation_coeff
            unit_vx, unit_vy, unit_vz = new_velocity_vector[0], new_velocity_vector[1], new_velocity_vector[2]
            new_velocity_magnitude = np.sqrt(new_particle_energy * 2 / argon_mass)
            energy_transfer = new_particle_energy - particle_energy
            energy_change_in_case += energy_transfer
            new_vx, new_vy, new_vz = unit_vx*new_velocity_magnitude, unit_vy*new_velocity_magnitude, unit_vz*new_velocity_magnitude
            new_z_momentum = argon_mass * new_vz
            momentum_z_difference = new_z_momentum - old_z_momentum
            momentem_z_change_in_case += momentum_z_difference
            if has_collided[p]: # a full path has been observed
                completed_paths.append(abs(continue_path[p] - abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)))
                completed_x_paths.append(abs(continue_x_path[p] - abs(vx*t)))
                completed_y_paths.append(abs(continue_y_path[p] - abs(vy*t)))
                completed_z_paths.append(abs(continue_z_path[p] - abs(vz*t)))
            else: #this was the end of a partial path
                has_collided[p] = True
            continue_path[p] = 0
            continue_x_path[p] = 0
            continue_y_path[p] = 0
            continue_z_path[p] = 0
            x_positions_in_case[p], y_positions_in_case[p], z_positions_in_case[p] = col_x, col_y, col_z
            x_velocities_in_case[p], y_velocities_in_case[p], z_velocities_in_case[p] = new_vx, new_vy, new_vz
        except:
            print([a, b, c])
            total_errs += 1
    x_vals[hits], y_vals[hits], z_vals[hits] = x_positions_in_case, y_positions_in_case, z_positions_in_case
    x_velocities[hits], y_velocities[hits], z_velocities[hits] = x_velocities_in_case, y_velocities_in_case, z_velocities_in_case
    dist_since_collision[hits] = continue_path
    dist_x_since_collision[hits] = continue_x_path
    dist_y_since_collision[hits] = continue_y_path
    dist_z_since_collision[hits] = continue_z_path
    full_path_traveled[hits] = has_collided
    num_collisions_per_step.value += num_particles_in_case
    return momentem_z_change_in_case, energy_change_in_case, total_errs

def hit_cylinder_gap_side_wall(hits, collision_radius, completed_paths, completed_x_paths, completed_y_paths, completed_z_paths, total_errs):    
    global x_vals, y_vals, z_vals, x_velocities, y_velocities, z_velocities, full_path_traveled
    global dist_since_collision, dist_x_since_collision, dist_y_since_collision, dist_z_since_collision
    global num_collisions_per_step
    x_positions_in_case = x_vals[hits]
    y_positions_in_case = y_vals[hits]
    z_positions_in_case = z_vals[hits]
    x_velocities_in_case = x_velocities[hits]
    y_velocities_in_case = y_velocities[hits]
    z_velocities_in_case = z_velocities[hits]
    continue_path = dist_since_collision[hits]
    continue_x_path = dist_x_since_collision[hits]
    continue_y_path = dist_y_since_collision[hits]
    continue_z_path = dist_z_since_collision[hits]
    has_collided = full_path_traveled[hits]
    num_particles_in_case = np.sum( hits )
    momentem_z_change_in_case = 0
    for p in range(num_particles_in_case):
        try:
            x, y, z = x_positions_in_case[p], y_positions_in_case[p], z_positions_in_case[p]
            vx, vy, vz = x_velocities_in_case[p], y_velocities_in_case[p], z_velocities_in_case[p]
            a = (-vx)**2 + (-vy)**2
            b = 2 * (x * (-vx) + y * (-vy))
            c = x**2 + y**2 - collision_radius**2
            t = np.min([(-b + np.sqrt(b**2 - 4*a*c))/(2*a), (-b - np.sqrt(b**2 - 4*a*c))/(2*a)])
            col_x, col_y, col_z = x-vx*t, y-vy*t, z-vz*t

            normal_vect = np.array([col_x, col_y, 0])
            normalized_norm_vect = normal_vect/collision_radius
            new_velocity_vector = random_inbounds_direction(-normalized_norm_vect)

            v_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
            old_z_momentum = argon_mass * vz
            particle_energy = kinetic_energy(v_magnitude)
            surface_energy = surface_energy_gap(col_z) 
            energy_difference = surface_energy - particle_energy
            new_particle_energy = particle_energy + energy_difference*gap_accomodation_coeff
            unit_vx, unit_vy, unit_vz = new_velocity_vector[0], new_velocity_vector[1], new_velocity_vector[2]
            new_velocity_magnitude = np.sqrt(new_particle_energy * 2 / argon_mass)
            new_vx, new_vy, new_vz = unit_vx*new_velocity_magnitude, unit_vy*new_velocity_magnitude, unit_vz*new_velocity_magnitude
            new_z_momentum = argon_mass * new_vz
            momentum_z_difference = new_z_momentum - old_z_momentum
            momentem_z_change_in_case += momentum_z_difference

            if has_collided[p]: # a full path has been observed
                completed_paths.append(abs(continue_path[p] - abs(np.sqrt(vx**2 + vy**2 + vz**2)*t)))
                completed_x_paths.append(abs(continue_x_path[p] - abs(vx*t)))
                completed_y_paths.append(abs(continue_y_path[p] - abs(vy*t)))
                completed_z_paths.append(abs(continue_z_path[p] - abs(vz*t)))
            else: #this was the end of a partial path
                has_collided[p] = True
            continue_path[p] = 0
            continue_x_path[p] = 0
            continue_y_path[p] = 0
            continue_z_path[p] = 0
            x_positions_in_case[p], y_positions_in_case[p], z_positions_in_case[p] = col_x, col_y, col_z
            x_velocities_in_case[p], y_velocities_in_case[p], z_velocities_in_case[p] = new_vx, new_vy, new_vz
        except:
            print([a, b, c])
            total_errs += 1
    x_vals[hits], y_vals[hits], z_vals[hits] = x_positions_in_case, y_positions_in_case, z_positions_in_case
    x_velocities[hits], y_velocities[hits], z_velocities[hits] = x_velocities_in_case, y_velocities_in_case, z_velocities_in_case
    dist_since_collision[hits] = continue_path
    dist_x_since_collision[hits] = continue_x_path
    dist_y_since_collision[hits] = continue_y_path
    dist_z_since_collision[hits] = continue_z_path
    full_path_traveled[hits] = has_collided
    num_collisions_per_step.value += num_particles_in_case
    return momentem_z_change_in_case, total_errs

def init_globals(counter):
    global num_collisions_per_step
    num_collisions_per_step = counter
    
def num_out_of_bounds():
    global x_vals, y_vals, z_vals
    num_particles_out_of_bounds = 0
    found = z_vals < 0
    for (x, y, z) in zip(x_vals[found], y_vals[found], z_vals[found]):
        print(x,y,z) 
    num_particles_out_of_bounds += found.sum()
    found = z_vals > total_height
    for (x, y, z) in zip(x_vals[found], y_vals[found], z_vals[found]):
        print(x,y,z)
    num_particles_out_of_bounds += found.sum()
    found = np.logical_and(x_vals**2 + y_vals**2 > open_air_radius**2, np.logical_and( z_vals >= 0, z_vals <=open_air_height))
    for (x, y, z) in zip(x_vals[found], y_vals[found], z_vals[found]):
        print(x,y,z)
    num_particles_out_of_bounds += found.sum()
    found = np.logical_and(x_vals**2 + y_vals**2 > open_air_radius**2, np.logical_and( z_vals >= total_height-open_air_height, z_vals <=total_height))
    for (x, y, z) in zip(x_vals[found], y_vals[found], z_vals[found]):
        print(x,y,z)
    num_particles_out_of_bounds += found.sum()
    found = np.logical_and(x_vals**2 + y_vals**2 > gap_radius**2, np.logical_and(z_vals >= gap_bottom_height, z_vals <= gap_top_height))
    for (px, py, pz, x, y, z) in zip(prior_x_vals[found],  prior_y_vals[found], prior_z_vals[found], x_vals[found], y_vals[found], z_vals[found]):
        print(px, py, pz, x, y, z)
    num_particles_out_of_bounds += found.sum()
    found = np.logical_and(x_vals**2 + y_vals**2 > pore_coated_radius**2, np.logical_and(z_vals > open_air_height, z_vals < gap_bottom_height))
    for (px, py, pz, x, y, z) in zip(prior_x_vals[found],  prior_y_vals[found], prior_z_vals[found], x_vals[found], y_vals[found], z_vals[found]):
        print(px, py, pz, x, y, z)
    num_particles_out_of_bounds += found.sum()
    found = np.logical_and(x_vals**2 + y_vals**2 > pore_coated_radius**2, np.logical_and(z_vals > gap_top_height, z_vals < (total_height - open_air_height)))
    for (px, py, pz, x, y, z) in zip(prior_x_vals[found],  prior_y_vals[found], prior_z_vals[found], x_vals[found], y_vals[found], z_vals[found]):
        print(px, py, pz, x, y, z)
    num_particles_out_of_bounds += found.sum()
    return num_particles_out_of_bounds

def recapture_out_of_bounds():
    global x_vals, y_vals, z_vals
    num_particles_out_of_bounds = 0
    found = z_vals < 0
    z_vals[found] = 50 * 10 ** -9
    num_particles_out_of_bounds += found.sum()
    found = z_vals > total_height
    z_vals[found] = total_height - (50 * 10 ** -9)
    num_particles_out_of_bounds += found.sum()
    found = x_vals**2 + y_vals**2 > open_air_radius**2
    x_vals[found] = 0
    y_vals[found] = 0
    num_particles_out_of_bounds += found.sum()
    found = np.logical_and(x_vals**2 + y_vals**2 > gap_radius**2, np.logical_and(z_vals > open_air_height, z_vals < (total_height - open_air_height)))
    x_vals[found] = 0
    y_vals[found] = 0
    num_particles_out_of_bounds += found.sum()
    found = np.logical_and(x_vals**2 + y_vals**2 > pore_coated_radius**2, np.logical_or(np.logical_and(z_vals > open_air_height, z_vals < (open_air_height+hot_coating_height)), np.logical_and(z_vals > (open_air_height+hot_coating_height+gap_height), z_vals < (total_height - open_air_height))))
    x_vals[found] = 0
    y_vals[found] = 0
    num_particles_out_of_bounds += found.sum()
    return num_particles_out_of_bounds
       
total_cols = 0
total_errs = 0

# Simulation Main Loop
if __name__ == "__main__":

    global dist_since_collision, dist_x_since_collision, dist_y_since_collision, dist_z_since_collision, full_path_traveled, x_vals, y_vals, z_vals, prior_x_vals, prior_y_vals, prior_z_vals, x_velocities, y_velocities, z_velocities

    #free path tracking
    dist_since_collision = np.zeros(num_molecules)
    dist_x_since_collision = np.zeros(num_molecules)
    dist_y_since_collision = np.zeros(num_molecules)
    dist_z_since_collision = np.zeros(num_molecules)
    full_path_traveled = np.zeros(num_molecules, dtype=bool)

    #momentum tracking
    momentum_z_change_per_step = []

    #energy tracking
    energy_transfer_hot_per_step = []
    energy_transfer_cold_per_step = []

    # Initialize positions
    x_vals,y_vals,z_vals = init_positions()
    prior_x_vals = np.zeros(num_molecules)
    prior_y_vals = np.zeros(num_molecules)
    prior_z_vals = np.zeros(num_molecules)

    x_velocities, y_velocities, z_velocities = init_velocities()

    # Grab Currrent Time After Running the initalization
    end_init = time()
    init_time = end_init - start
    print( 'Initialization Runtime: '+ str(init_time) + ' seconds')
    print('  There are {} particles out of bounds after initialization.'.format(num_out_of_bounds()))    

    with Manager() as manager:
        #MP safe lists for making the particle vs particle comparisons parallel
        completed_paths = manager.list()
        completed_x_paths = manager.list()
        completed_y_paths = manager.list()
        completed_z_paths = manager.list()

        # Evolve
        for step in range(num_timesteps):

            # timestamp of the start of a timestep
            step_start = time()
            
            print('  timestep',step,'of',num_timesteps,'  (sim',1,'/',num_sims,')')

            #reset num_collisions_per_step to MP safe value of zero for new timestep
            num_collisions_per_step = Value('i', 0)
            
            # DRIFT
            prior_x_vals = np.copy(x_vals)
            prior_y_vals = np.copy(y_vals)
            prior_z_vals = np.copy(z_vals)
            x_vals += dt*x_velocities
            y_vals += dt*y_velocities
            z_vals += dt*z_velocities
            #increase distance since last collision
            dist_since_collision += abs(np.sqrt(np.square(dt*x_velocities) + np.square(dt*y_velocities) + np.square(dt*z_velocities)))
            dist_x_since_collision += abs(dt*x_velocities)
            dist_y_since_collision += abs(dt*y_velocities)
            dist_z_since_collision += abs(dt*z_velocities)

            momentum_z_change_in_step = 0
            energy_change_hot_in_step = 0
            energy_change_cold_in_step = 0


            # WALL COLLISIONS
            # CASE 1 
            # collide specular side of open air cylinder
            hit_side_open_air = np.sqrt(x_vals**2 + y_vals**2) > open_air_radius #true/false array
            total_errs = hit_cylinder_specular_side_wall(hit_side_open_air, open_air_collision_radius, total_errs)            

            # CASE 2
            # collide specular exterier vertical limits of described open air shape
            #top side
            hit_vertical_ext_open_air_min = z_vals < 0
            hit_vertical_specular_wall(hit_vertical_ext_open_air_min, 0)
            #bottom side
            hit_vertical_ext_open_air_max = z_vals > total_height
            hit_vertical_specular_wall(hit_vertical_ext_open_air_max, total_height)    

            # CASE 3
            # collide coated interior vertical limits of described open air shape
            #cold
            hit_vertical_int_open_air_cold = np.logical_and(prior_z_vals >= total_height - open_air_height + argon_radius, np.logical_and(z_vals < total_height - open_air_height + argon_radius, x_vals**2 + y_vals**2 > pore_coated_radius**2)) #cold side
            momentum_z_change_in_case, energy_change_cold_in_case = hit_vertical_coated_wall(hit_vertical_int_open_air_cold, surface_energy_cold, (total_height - open_air_height + argon_radius), 1, completed_paths, completed_x_paths, completed_y_paths, completed_z_paths)
            momentum_z_change_in_step += momentum_z_change_in_case
            energy_change_cold_in_step += energy_change_cold_in_case
            #hot
            hit_vertical_int_open_air_hot = np.logical_and(prior_z_vals <= open_air_height - argon_radius, np.logical_and(z_vals > open_air_height - argon_radius, x_vals**2 + y_vals**2 > pore_coated_radius**2)) #hot side
            momentum_z_change_in_case, energy_change_hot_in_case = hit_vertical_coated_wall(hit_vertical_int_open_air_hot, surface_energy_hot, open_air_height - argon_radius, -1, completed_paths, completed_x_paths, completed_y_paths, completed_z_paths)
            momentum_z_change_in_step += momentum_z_change_in_case
            energy_change_hot_in_step += energy_change_hot_in_case

            # CASE 4
            # collide with gap interior side wall
            hit_gap_cylinder_wall = np.logical_and( np.logical_and(prior_z_vals < gap_top_height - argon_radius, prior_z_vals > gap_bottom_height + argon_radius), 
                                                    np.logical_and(prior_x_vals**2 + prior_y_vals**2 <= gap_collision_radius**2, x_vals**2 + y_vals**2 > gap_collision_radius**2))
            momentum_z_change_in_case, total_errs = hit_cylinder_gap_side_wall(hit_gap_cylinder_wall, gap_collision_radius, completed_paths, completed_x_paths, completed_y_paths, completed_z_paths, total_errs)
            momentum_z_change_in_step += momentum_z_change_in_case

            # CASE 5
            # collide with top or bottom bases of gap cylinder
            #bottom/hot
            hit_gap_cylinder_base_bottom = np.logical_and(  np.logical_and( prior_x_vals**2 + prior_y_vals**2 >= pore_collision_radius**2, z_vals < gap_bottom_height + argon_radius),
                                                            np.logical_and( prior_z_vals <= gap_top_height - argon_radius, prior_z_vals >= gap_bottom_height + argon_radius))
            momentum_z_change_in_case, energy_change_hot_in_case = hit_vertical_coated_wall(hit_gap_cylinder_base_bottom, surface_energy_hot, gap_bottom_height + argon_radius, 1, completed_paths, completed_x_paths, completed_y_paths, completed_z_paths)
            momentum_z_change_in_step += momentum_z_change_in_case
            energy_change_hot_in_step += energy_change_hot_in_case
            #top/cold
            hit_gap_cylinder_base_top = np.logical_and( np.logical_and( prior_x_vals**2 + prior_y_vals**2 >= pore_collision_radius**2, z_vals > gap_top_height - argon_radius),
                                                        np.logical_and( prior_z_vals <= gap_top_height - argon_radius, prior_z_vals >= gap_bottom_height + argon_radius))
            momentum_z_change_in_case, energy_change_cold_in_case = hit_vertical_coated_wall(hit_gap_cylinder_base_top, surface_energy_cold, gap_top_height - argon_radius, -1,  completed_paths, completed_x_paths, completed_y_paths, completed_z_paths)
            momentum_z_change_in_step += momentum_z_change_in_case
            energy_change_cold_in_step += energy_change_cold_in_case                                              

            # CASE 6
            # collide with coated pore side wall
            #hot
            hit_pore_coating_hot  = np.logical_and( np.logical_and( prior_x_vals**2 + prior_y_vals**2 <= pore_collision_radius**2, x_vals**2 + y_vals**2 > pore_collision_radius**2), #always necessary for a collision with the coating wall
                                                    np.logical_and(z_vals <= gap_bottom_height + argon_radius, z_vals >= open_air_height - argon_radius)) #hot coating reflection - treated as specular in this simulation
            momentum_z_change_in_case, energy_change_hot_in_case, total_errs = hit_cylinder_coated_side_wall(hit_pore_coating_hot, surface_energy_hot, pore_collision_radius, completed_paths, completed_x_paths, completed_y_paths, completed_z_paths, total_errs)
            momentum_z_change_in_step += momentum_z_change_in_case
            energy_change_hot_in_step += energy_change_hot_in_case
            #cold
            hit_pore_coating_cold = np.logical_and( np.logical_and( prior_x_vals**2 + prior_y_vals**2 <= pore_collision_radius**2, x_vals**2 + y_vals**2 > pore_collision_radius**2), #always necessary for a collision with the coating wall
                                                    np.logical_and(z_vals < total_height-open_air_height + argon_radius, z_vals > gap_top_height-argon_radius)) #cold coating reflection - treated as specular in this simulation                                                  
            momentum_z_change_in_case, energy_change_cold_in_case, total_errs = hit_cylinder_coated_side_wall(hit_pore_coating_cold, surface_energy_cold, pore_collision_radius, completed_paths, completed_x_paths, completed_y_paths, completed_z_paths, total_errs)
            momentum_z_change_in_step += momentum_z_change_in_case
            energy_change_cold_in_step += energy_change_cold_in_case

            #track momentum change per timestep over time
            momentum_z_change_per_step.append(momentum_z_change_in_step)
            energy_transfer_hot_per_step.append(energy_change_hot_in_step)
            energy_transfer_cold_per_step.append(energy_change_cold_in_step)

            # #Checks for piece of mind that particles are not lost over time - uncomment out to run, and change modulus to affect frequency of check (mod 1 => run each timestep)
            # if (step%1 == 0):
            #     hit = np.sqrt(x_vals**2 + y_vals**2) > open_air_radius #true/false array
            #     print('             ',hit.sum(),' missed case 1\'s')
            #     for (x, y, z) in zip(x_vals[hit], y_vals[hit], z_vals[hit]):
            #         print(x,y,z)
            #     hit = z_vals < 0
            #     print('             ',hit.sum(),' missed case 2a\'s')
            #     for (x, y, z) in zip(x_vals[hit], y_vals[hit], z_vals[hit]):
            #         print(x,y,z)
            #     hit = z_vals > total_height
            #     print('             ',hit.sum(),' missed case 2b\'s')
            #     for (x, y, z) in zip(x_vals[hit], y_vals[hit], z_vals[hit]):
            #         print(x,y,z)
            #     hit = np.logical_and(prior_z_vals >= total_height - open_air_height + argon_radius, np.logical_and(z_vals < total_height - open_air_height + argon_radius, np.sqrt(x_vals**2 + y_vals**2) > pore_coated_radius)) #cold side
            #     print('             ',hit.sum(),' missed case 3a\'s')
            #     for (x, y, z) in zip(x_vals[hit], y_vals[hit], z_vals[hit]):
            #         print(x,y,z)
            #     hit = np.logical_and(prior_z_vals <= open_air_height - argon_radius, np.logical_and(z_vals > open_air_height - argon_radius, np.sqrt(x_vals**2 + y_vals**2) > pore_coated_radius)) #hot side
            #     print('             ',hit.sum(),' missed case 3b\'s')
            #     for (x, y, z) in zip(x_vals[hit], y_vals[hit], z_vals[hit]):
            #         print(x,y,z)
            #     hit_gap_cylinder_wall = np.logical_and( np.logical_and(prior_z_vals < gap_top_height - argon_radius, prior_z_vals > gap_bottom_height + argon_radius), 
            #                                             np.logical_and(np.sqrt(prior_x_vals**2 + prior_y_vals**2) <= gap_collision_radius, np.sqrt(x_vals**2 + y_vals**2) > gap_collision_radius))
            #     print('             ',hit.sum(),' missed case 4\'s')
            #     for (px, py, pz, x, y, z) in zip(prior_x_vals[hit],  prior_y_vals[hit], prior_z_vals[hit], x_vals[hit], y_vals[hit], z_vals[hit]):
            #         print(px, py, pz, x, y, z)
            #     hit = hit_gap_cylinder_base_bottom = np.logical_and(    np.logical_and( np.sqrt(prior_x_vals**2 + prior_y_vals**2) >= pore_collision_radius, z_vals < gap_bottom_height + argon_radius),
            #                                                             np.logical_and( prior_z_vals <= gap_top_height - argon_radius, prior_z_vals >= gap_bottom_height + argon_radius))
            #     print('             ',hit.sum(),' missed case 5a\'s')
            #     for (px, py, pz, x, y, z) in zip(prior_x_vals[hit],  prior_y_vals[hit], prior_z_vals[hit], x_vals[hit], y_vals[hit], z_vals[hit]):
            #         print(px, py, pz, x, y, z)
            #     hit = hit_gap_cylinder_base_top = np.logical_and(   np.logical_and( np.sqrt(prior_x_vals**2 + prior_y_vals**2) >= pore_collision_radius, z_vals > gap_top_height - argon_radius),
            #                                                         np.logical_and( prior_z_vals <= gap_top_height - argon_radius, prior_z_vals >= gap_bottom_height + argon_radius))
            #     print('             ',hit.sum(),' missed case 5b\'s')
            #     for (px, py, pz, x, y, z) in zip(prior_x_vals[hit],  prior_y_vals[hit], prior_z_vals[hit], x_vals[hit], y_vals[hit], z_vals[hit]):
            #         print(px, py, pz, x, y, z)
            #     hit_pore_coating_hot  = np.logical_and( np.logical_and( np.sqrt(prior_x_vals**2 + prior_y_vals**2) <= pore_collision_radius, np.sqrt(x_vals**2 + y_vals**2) > pore_collision_radius), #always necessary for a collision with the coating wall
            #                                             np.logical_and(z_vals <= gap_bottom_height + argon_radius, z_vals >= open_air_height - argon_radius)) #hot coating reflection - treated as specular in this simulation
            #     print('             ',hit.sum(),' missed case 6a\'s')
            #     hit_pore_coating_cold = np.logical_and( np.logical_and( np.sqrt(prior_x_vals**2 + prior_y_vals**2) <= pore_collision_radius, np.sqrt(x_vals**2 + y_vals**2) > pore_collision_radius), #always necessary for a collision with the coating wall
            #                                             np.logical_and(z_vals < total_height-open_air_height + argon_radius, z_vals > gap_top_height-argon_radius)) #cold coating reflection - treated as specular in this simulation
            #     print('             ',hit.sum(),' missed case 6b\'s')
            print('    There are {} particles out of bounds after handling wall collisions.'.format(num_out_of_bounds()))
            recapture_out_of_bounds()
            print('    There are {} particles out of bounds after post wall collision recapture.'.format(num_out_of_bounds()))
            
            # Grab Currrent Time After Running the initalization
            end_step_walls = time()
            wall_time = end_step_walls - step_start
            print( '    Wall Step Runtime: '+ str(wall_time) + ' seconds')
            print( '    Num collisions from walls: ' + str(num_collisions_per_step.value))

            # PARTICLE-PARTICLE COLLISIONS
            # For each independant group of cells
            for x_group in range(2):
                for y_group in range(2):
                    for z_group in range(2):
                        
                        # loop over cells                   
                        in_x_layers = [(((2*x_layer+x_group-num_x_subdivions)*dx - collision_range) < x_vals) & (x_vals < ((2*x_layer+x_group-num_x_subdivions+1)*dx)) for x_layer in range(num_x_subdivions)]
                        in_y_layers = [(((2*y_layer+y_group-num_y_subdivions)*dy - collision_range) < y_vals) & (y_vals < ((2*y_layer+y_group-num_y_subdivions+1)*dy)) for y_layer in range(num_y_subdivions)]       
                        in_z_layers = [(((2*z_layer+z_group)*dz  - collision_range) < z_vals) & (z_vals < ((2*z_layer+z_group+1)*dz)) for z_layer in range(int(num_z_subdivions/2))]
                        list_of_cells_in_group = [x_layer & y_layer & z_layer for x_layer in in_x_layers for y_layer in in_y_layers for z_layer in in_z_layers if np.sum( x_layer & y_layer & z_layer )>0]
                        
                        #Format cell data            
                        continue_paths = list(dist_since_collision[cell] for cell in list_of_cells_in_group)
                        continue_x_paths = list(dist_x_since_collision[cell] for cell in list_of_cells_in_group)
                        continue_y_pats = list(dist_y_since_collision[cell] for cell in list_of_cells_in_group)
                        continue_z_paths = list(dist_z_since_collision[cell] for cell in list_of_cells_in_group)
                        cells_of_has_collided = list(full_path_traveled[cell] for cell in list_of_cells_in_group)
                        x_positions_in_cells = list(x_vals[cell] for cell in list_of_cells_in_group)
                        y_positions_in_cells = list(y_vals[cell] for cell in list_of_cells_in_group)
                        z_positions_in_cells = list(z_vals[cell] for cell in list_of_cells_in_group)
                        x_velocities_in_cells = list(x_velocities[cell] for cell in list_of_cells_in_group)
                        y_velocities_in_cells = list(y_velocities[cell] for cell in list_of_cells_in_group)
                        z_velocities_in_cells = list(z_velocities[cell] for cell in list_of_cells_in_group)
                        # Solve cell particle vs particle collisions in parallel
                        pool = Pool(num_workers, initializer=init_globals, initargs=(num_collisions_per_step,))
                        for in_cell, continue_path, continue_x_path, continue_y_path, continue_z_paths, has_collided, x_positions_in_cell, y_positions_in_cell, z_positions_in_cell, x_velocities_in_cell, y_velocities_in_cell, z_velocities_in_cell in pool.starmap(partial(pairwise_particles_in_cell, completed_paths, completed_x_paths, completed_y_paths, completed_z_paths), list(zip(list_of_cells_in_group, continue_paths, continue_x_paths, continue_y_pats, continue_z_paths, cells_of_has_collided, x_positions_in_cells, y_positions_in_cells, z_positions_in_cells, x_velocities_in_cells, y_velocities_in_cells, z_velocities_in_cells))):
                            dist_since_collision[in_cell], dist_x_since_collision[in_cell], dist_y_since_collision[in_cell], dist_z_since_collision[in_cell], full_path_traveled[in_cell], x_vals[in_cell], y_vals[in_cell], z_vals[in_cell], x_velocities[in_cell], y_velocities[in_cell], z_velocities[in_cell] = continue_path, continue_x_path, continue_y_path, continue_z_paths, has_collided, x_positions_in_cell, y_positions_in_cell, z_positions_in_cell, x_velocities_in_cell, y_velocities_in_cell, z_velocities_in_cell
                        pool.close()
                        pool.join()
            print('    There are {} particles out of bounds after particle-particle collisions.'.format(num_out_of_bounds()))
            recapture_out_of_bounds()
            print('    There are {} particles out of bounds after post particle-particle recapture.'.format(num_out_of_bounds()))
            # Grab Currrent Time After Running the initalization
            end_step_pvp = time()
            pvp_time = end_step_pvp - end_step_walls
            print( '    Particle-Particle step Runtime: '+ str(pvp_time) + ' seconds')
            # Update total collisions and print the completed timestep's collisions
            total_cols += num_collisions_per_step.value
            print('   ',num_collisions_per_step.value,' collisions from this timestep')
            print(' ',total_errs,' errors/warnings so far')

        # Note relevant end of sim values
        print(' ',total_errs,' errors/warnings - potential lost particles')        
        print(' ',total_cols,' collisions')
        print(' ',len(completed_paths),' completed paths')

        # print mean free path
        print('Simulation mean free path: ' + str(np.average(completed_paths)))
        print('Simulation mean x free path: ' + str(np.average(completed_x_paths)))
        print('Simulation mean y free path: ' + str(np.average(completed_y_paths)))
        print('Simulation mean z free path: ' + str(np.average(completed_z_paths)))
        print('Num of measured full paths total: '+ str(len(completed_paths)))
        
        #generate figure for graphing
        #Subplot for total distance
        ax1 = fig.add_subplot(4, 1, 1)
        data = completed_paths
        n_total, bins_total, patches_total = ax1.hist(data, range=(0,10**-6), bins=num_bins, density=True, color ='green', label='3d distance data') # color = numpy.random.rand(3,)
        ax1.set_xlabel('Path length before collision (m)')
        ax1.set_ylabel('Probability')
        ax1.legend()
        #Subplot for x distance
        ax2 = fig.add_subplot(4, 1, 2)
        x_data = completed_x_paths
        n_x, bins_x, patches_x = ax2.hist(x_data, range=(0,10**-6), bins=num_bins, density=True, color ='green', label='x data') # color = numpy.random.rand(3,)
        ax2.set_xlabel('X Path length before collision (m)')
        ax2.set_ylabel('Probability')
        ax2.legend()
        #Subplot for y distance
        ax3 = fig.add_subplot(4, 1, 3)
        y_data = completed_y_paths
        n_y, bins_y, patches_y = ax3.hist(y_data, range=(0,10**-6), bins=num_bins, density=True, color ='green', label='y data') # color = numpy.random.rand(3,)
        ax3.set_xlabel('Y Path length before collision (m)')
        ax3.set_ylabel('Probability')
        ax3.legend()
        #Subplot for z distance
        ax4 = fig.add_subplot(4, 1, 4)
        z_data = completed_z_paths
        n_z, bins_z, patches_z = ax4.hist(z_data, range=(0,10**-6), bins=num_bins, density=True, color ='green', label='z data') # color = numpy.random.rand(3,)  
        ax4.set_xlabel('Z Path length before collision (m)')
        ax4.set_ylabel('Probability')
        ax4.legend()

        # Grab Currrent Time After Running the Code
        end = time()
        runtime = end - start
        print( 'Runtime: '+ str(runtime/60.0) + ' minutes (pre histogram data rewrite)')

        #save histogram data to text files
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

        #output momentum and energy transfer data to CSV
        momentum_energy_data = {'Momentum': momentum_z_change_per_step,
                                'EnergyCold': energy_transfer_cold_per_step,
                                'EnergyHot': energy_transfer_hot_per_step}
        df = pd.DataFrame.from_dict(momentum_energy_data)
        df.to_csv('momentum_energy.csv')

        print(sum(momentum_z_change_per_step))
        print(sum(energy_transfer_cold_per_step))
        print(sum(energy_transfer_hot_per_step))

        # Grab Currrent Time After Running the Code
        end = time()
        runtime = end - start
        print( 'Runtime: '+ str(runtime/60.0) + ' minutes')

        #show graph
        plt.show()