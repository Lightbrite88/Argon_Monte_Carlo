# Argon_Monte_Carlo
A time based 1:1 hard sphere argon particle collision monte carlo in python.
 - Originally based on a DSMC, the entire methodology was rewritten to use custom volumes, a mix of specular and energized surface collisions, and 1:1 particle collision tracking (given that we wanted to ensure elastic collisions) since momentum tracking is the focus and the final objective. 

## Simulations
All sims return histogram x and y values as text files for mean free path tracking. Additionally, each simulation displays a window figure of the histogram graphs.
Only the final simulation writes a csv of the momentum change per timestep.
 - utils.py is simply a separate file for a function to keep the main simulation files a little cleaner and readable

### Open_Air_Cube_MC.py
1:1 specular wall collisions in a cube to establish mean free path validity
- this was the first stage, and had few enough particles to be run in serial

### Open_Air_Pore_MC.py
1:1 specular wall collisions in a thruster pore to establish new mean free path given the thruster's shape
- this was the second stage, and had enough particles to require parallelization for the p-p collisions

### Temperature_Pore_MC.py
1:1 energized wall collisions in a thruster pore with open air enclosures at endpoints to establish momentum change of thruster
- this was the third and final stage, and had enough particles to require parallelization for the p-p collisions

## Usage
As long as you have a copy of any of the above listed simulation files, as well as the utils.py file, you may run the simulation. The txt files are not necessary, and are merely the most current values from the auther for the latest simulation.

## Credits
Author: Jeff Hatton (Science methodology by Sean Wagner)

Initially inspired by DSMC by Philip Mocz (2021) Princeton Univeristy




