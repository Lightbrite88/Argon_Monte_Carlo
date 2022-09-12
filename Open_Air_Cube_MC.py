import matplotlib.pyplot as plt
import numpy as np
import random
import math
import sys
import time
import tracemalloc
import linecache
import os
from scipy import stats
from scipy.optimize import curve_fit

np.set_printoptions(threshold=sys.maxsize)

"""
1:1 Time based Hard Sphere Argon Particle Collision Monte Carlo
Author: Jeff Hatton (Science methodology by Sean Wagner)
Initially inspired by a DSMC by Philip Mocz (2021) Princeton Univeristy, @PMocz
"""

# Grab Currrent Time Before Running the Code and start tracking memory usage
# start = time.time()
# tracemalloc.start()

#Shape description
cube_x              = 100 * 10 ** -9              # metres
cube_y              = 100 * 10 ** -9              # metres
cube_z              = 100 * 10 ** -9              # metres
cube_volume         = cube_x * cube_y * cube_z
num_x_subdivions    = 15
num_y_subdivions    = 15
num_z_subdivions    = 15              
dx                  = cube_x/num_x_subdivions # cell x
dy                  = cube_y/num_y_subdivions # cell y
dz                  = cube_z/num_z_subdivions # cell z
collision_x_overlap = dx/10
collision_y_overlap = dy/10
collision_z_overlap = dz/10
cell_volume         = dx * dy * dz

#Physics
argon_mass          = 6.63 * 10**-26        # kg
ar_molar_mass       = 0.039948              # Kg/mole
molecules_per_mole  = 6.02214179 * 10**23   # molecules per mole
ideal_gas_const     = 8.3145                # J/(mole*kelvin)
boltzman            = 1.38 * 10**(-23)      # m^2Kg/s^2K
temp_ambient        = 298                   # kelvin
sigma               = 3.6 * 10**(-19)       # 3.6*10^-19 m^2
argon_radius        = np.sqrt(sigma/(4*np.pi))  # 1.692568750643269 * 10^-10 m 
collision_radius    = argon_radius*1        #consider increase by 15% for collision detection purposes ~1.946 * 10^-10 m
collision_range     = collision_radius*2    # ~3.89 * 10^-10 m
pressure            = 101325                # N/m^2
lambda_mfp          = boltzman*temp_ambient/(np.sqrt(2)*sigma*pressure) # ~79.7 nm mean free path
v_mean              = np.sqrt(3*ideal_gas_const*temp_ambient/ar_molar_mass) # mean speed
num_moles           = cube_volume*pressure/(ideal_gas_const*temp_ambient)
a_shape             = np.sqrt(boltzman*temp_ambient/argon_mass) # argon boltzmann shaping factor
num_molecules       = np.round(num_moles * molecules_per_mole).astype(int)
print(num_molecules)

#Time
tau                 = lambda_mfp / v_mean   # mean-free time
Nmft                = 20                    # number of mean-free times to run simulation
num_timesteps       = Nmft*25               # number of time steps (25 per mean-free time)
dt                  = Nmft*tau/num_timesteps  # timestep

# Simulation
min_num_particles_per_cell = np.floor(num_molecules/(num_x_subdivions*num_y_subdivions*num_z_subdivions)).astype(int)
N                   = min_num_particles_per_cell*num_x_subdivions*num_y_subdivions*num_z_subdivions     # number of sampling particles
remaining_particles = num_molecules - N
#molecules_per_sample_particle = np.round(num_molecules/N).astype(int)
num_particles_per_x_layer = int(N/num_x_subdivions)
num_particles_per_y_layer = int(N/num_y_subdivions)
num_particles_per_z_layer = int(N/num_z_subdivions)
Nsim                = 1         # number of simulations to run

# set the random number generator seed
np.random.seed(127) 
random.seed(127)

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

# def display_top(snapshot, key_type='lineno', limit=3):
#     snapshot = snapshot.filter_traces((
#         tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
#         tracemalloc.Filter(False, "<unknown>"),
#     ))
#     top_stats = snapshot.statistics(key_type)

#     print("Top %s lines" % limit)
#     for index, stat in enumerate(top_stats[:limit], 1):
#         frame = stat.traceback[0]
#         # replace "/path/to/module/file.py" with "module/file.py"
#         filename = os.sep.join(frame.filename.split(os.sep)[-2:])
#         print("#%s: %s:%s: %.1f KiB"
#               % (index, filename, frame.lineno, stat.size / 1024))
#         line = linecache.getline(frame.filename, frame.lineno).strip()
#         if line:
#             print('    %s' % line)

#     other = top_stats[limit:]
#     if other:
#         size = sum(stat.size for stat in other)
#         print("%s other: %.1f KiB" % (len(other), size / 1024))
#     total = sum(stat.size for stat in top_stats)
#     print("Total allocated size: %.1f KiB" % (total / 1024))

#Intended curve for fitting (exponential decay)
def fit_exp_function(independant_variable, coeff_1, coeff_2):
    return coeff_1 * np.exp(coeff_2 * np.array(independant_variable))

#Intended curve for fitting (inverse)
def fit_inv_function(independant_variable, coeff_1, coeff_2, coeff_3):
    return coeff_1 * (independant_variable-coeff_2)**coeff_3



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
    x_vals = cube_x * np.random.random(remaining_particles)
    y_vals = cube_y * np.random.random(remaining_particles)
    z_vals = cube_z * np.random.random(remaining_particles)
    for x in range(num_x_subdivions):
        for y in range(num_y_subdivions):
            for z in range(num_z_subdivions):
                xv =  dx * (np.random.random(min_num_particles_per_cell) + x)
                yv =  dy * (np.random.random(min_num_particles_per_cell) + y)
                zv =  dz * (np.random.random(min_num_particles_per_cell) + z)
                x_vals = np.concatenate( (x_vals, xv), axis=0 )
                y_vals = np.concatenate( (y_vals, yv), axis=0 )
                z_vals = np.concatenate( (z_vals, zv), axis=0 )
                
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

    # Evolve
    for i in range(num_timesteps):
        
        print('  timestep',i,'of',num_timesteps,'  (sim',sim+1,'/',Nsim,')')
        
        # drift
        x_vals += dt*x_velocities
        y_vals += dt*y_velocities
        z_vals += dt*z_velocities
        #increase distance since last collision
        dist_since_collision[sim] += abs(np.sqrt(np.square(dt*x_velocities) + np.square(dt*y_velocities) + np.square(dt*z_velocities)))
        dist_x_since_collision[sim] += abs(dt*x_velocities)
        dist_y_since_collision[sim] += abs(dt*y_velocities)
        dist_z_since_collision[sim] += abs(dt*z_velocities)

        # collide specular walls perpendicular to x axis
        # trace the straight-line trajectory to the top wall, bounce it back
        # max value wall (x=cube_x)
        hit_right = x_vals > cube_x #true/false array
        dt_ac = (x_vals[hit_right]-cube_x) / x_velocities[hit_right] # time after collision
        x_velocities[hit_right] = -x_velocities[hit_right]  # reverse normal component of velocity
        x_vals[hit_right] = cube_x + dt_ac * x_velocities[hit_right]
        # min value wall (x=0)
        hit_left = x_vals < 0 #true/false array
        dt_ac = x_vals[hit_left] / x_velocities[hit_left] # time after collision
        x_velocities[hit_left] = -x_velocities[hit_left]  # reverse normal component of velocity
        x_vals[hit_left] = dt_ac * x_velocities[hit_left]

        # collide specular walls perpendicular to y axis
        # trace the straight-line trajectory to the top wall, bounce it back
        # max value wall (y=cube_y)
        hit_back = y_vals > cube_y #true/false array
        dt_ac = (y_vals[hit_back]-cube_y) / y_velocities[hit_back] # time after collision
        y_velocities[hit_back] = -y_velocities[hit_back]  # reverse normal component of velocity
        y_vals[hit_back] = cube_y + dt_ac * y_velocities[hit_back]
        # min value wall (y=0)
        hit_front = y_vals < 0 #true/false array
        dt_ac = y_vals[hit_front] / y_velocities[hit_front] # time after collision
        y_velocities[hit_front] = -y_velocities[hit_front]  # reverse normal component of velocity
        y_vals[hit_front] = dt_ac * y_velocities[hit_front]
        
        # collide specular walls perpendicular to z axis
        # trace the straight-line trajectory to the top wall, bounce it back
        # max value wall (z=cube_z)
        hit_top = z_vals > cube_z #true/false array
        dt_ac = (z_vals[hit_top]-cube_z) / z_velocities[hit_top] # time after collision
        z_velocities[hit_top] = -z_velocities[hit_top]  # reverse normal component of velocity
        z_vals[hit_top] = cube_z + dt_ac * z_velocities[hit_top]
        # min value wall (z=0)
        hit_bottom = z_vals < 0 #true/false array
        dt_ac = z_vals[hit_bottom] / z_velocities[hit_bottom] # time after collision
        z_velocities[hit_bottom] = -z_velocities[hit_bottom]  # reverse normal component of velocity
        z_vals[hit_bottom] = dt_ac * z_velocities[hit_bottom]
        
        # collide particles using acceptance--rejection scheme
        N_collisions = 0

        # loop over cells      
        for x_layer in range(num_x_subdivions):
            in_x_layer = ((x_layer*dx - collision_x_overlap) < x_vals) & (x_vals < ((x_layer+1)*dx))
            for y_layer in range(num_y_subdivions):
                in_y_layer = ((y_layer*dy - collision_y_overlap) < y_vals) & (y_vals < ((y_layer+1)*dy))
                for z_layer in range(num_z_subdivions):         
                    in_z_layer = ((z_layer*dz  - collision_z_overlap) < z_vals) & (z_vals < ((z_layer+1)*dz))
                    in_cell = in_x_layer & in_y_layer & in_z_layer #true false array of whether particle is in cell
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

        print('    ',N_collisions,' collisions')

    #generate figure for graphing
    #Subplot for total distance
    ax1 = fig.add_subplot(4, Nsim, sim+1)
    data = completed_paths[sim]
    n_total, bins_total, patches_total = ax1.hist(data, range=(0,10**-6), bins=num_bins, density=True, color ='green', label='3d distance data') # color = numpy.random.rand(3,)
    # popt, pcov = curve_fit(fit_function, bins[0:len(n)]*(10**6), n/(10**6), p0=[14.0, -11.0], maxfev=25000) #po is first guess, maxfev is number of guesses
    popt, pcov = curve_fit(fit_exp_function, bins_total[0:len(n_total)], n_total, p0=[14000000.0, -11000000.0], maxfev=25000) #po is first guess, maxfev is number of guesses
    # ax.plot(bins[0:len(n)], small_data_fit_function(bins[0:len(n)], *popt), 'r--', label='fit: a=%5.8f, b=%5.8f' % tuple(popt))
    ax1.plot(bins_total[0:len(n_total)], fit_exp_function(bins_total[0:len(n_total)], *popt), 'r--', label='fit: a=%5.8f, b=%5.8f' % tuple(popt))
    ax1.set_xlabel('Path length before collision (m)')
    ax1.set_ylabel('Probability')
    ax1.legend()
    #Subplot for x distance
    ax2 = fig.add_subplot(4, Nsim, sim+2)
    x_data = completed_x_paths[sim]
    n_x, bins_x, patches_x = ax2.hist(x_data, range=(0,10**-6), bins=num_bins, density=True, color ='green', label='x data') # color = numpy.random.rand(3,)
    # popt, pcov = curve_fit(fit_function, bins[0:len(n)]*(10**6), n/(10**6), p0=[14.0, -11.0], maxfev=25000) #po is first guess, maxfev is number of guesses
    popt_x, pcov_x = curve_fit(fit_inv_function, bins_x[0:len(n_x)], n_x, p0=[1.0, 0.0, -3.0], maxfev=25000) #po is first guess, maxfev is number of guesses
    # ax.plot(bins[0:len(n)], small_data_fit_function(bins[0:len(n)], *popt), 'r--', label='fit: a=%5.8f, b=%5.8f' % tuple(popt))
    #ax2.plot(bins_x[0:len(n_x)], fit_inv_function(bins_x[0:len(n_x)], *popt_x), 'r--', label='fit: a=%5.8f, b=%5.8f, c=%5.8f' % tuple(popt_x))
    ax2.set_xlabel('X Path length before collision (m)')
    ax2.set_ylabel('Probability')
    ax2.legend()
    #Subplot for y distance
    ax3 = fig.add_subplot(4, Nsim, sim+3)
    y_data = completed_y_paths[sim]
    n_y, bins_y, patches_y = ax3.hist(y_data, range=(0,10**-6), bins=num_bins, density=True, color ='green', label='y data') # color = numpy.random.rand(3,)
    # popt, pcov = curve_fit(fit_function, bins[0:len(n)]*(10**6), n/(10**6), p0=[14.0, -11.0], maxfev=25000) #po is first guess, maxfev is number of guesses
    popt_y, pcov = curve_fit(fit_inv_function, bins_y[0:len(n_y)], n_y, p0=[1.0, 0.0, -3.0], maxfev=25000) #po is first guess, maxfev is number of guesses
    # ax.plot(bins[0:len(n)], small_data_fit_function(bins[0:len(n)], *popt), 'r--', label='fit: a=%5.8f, b=%5.8f' % tuple(popt))
    #ax3.plot(bins_y[0:len(n_y)], fit_inv_function(bins_y[0:len(n_y)], *popt_y), 'r--', label='fit: a=%5.8f, b=%5.8f, c=%5.8f' % tuple(popt_y))
    ax3.set_xlabel('Y Path length before collision (m)')
    ax3.set_ylabel('Probability')
    ax3.legend()
    #Subplot for z distance
    ax4 = fig.add_subplot(4, Nsim, sim+4)
    z_data = completed_z_paths[sim]
    n_z, bins_z, patches_z = ax4.hist(z_data, range=(0,10**-6), bins=num_bins, density=True, color ='green', label='z data') # color = numpy.random.rand(3,)
    # popt, pcov = curve_fit(fit_function, bins[0:len(n)]*(10**6), n/(10**6), p0=[14.0, -11.0], maxfev=25000) #po is first guess, maxfev is number of guesses
    popt_z, pcov = curve_fit(fit_inv_function, bins_z[0:len(n_z)], n_z, p0=[1.0, 0.0, -3.0], maxfev=25000) #po is first guess, maxfev is number of guesses
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
# end = time.time()
# runtime = end - start
# print( 'Runtime: '+ str(runtime/60.0) + ' minutes')

# 3 highest memory lines
# snapshot = tracemalloc.take_snapshot()
# display_top(snapshot)

#show graph
plt.show()
