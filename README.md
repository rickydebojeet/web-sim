# Closed Loop Web Server Simulator
It is a simulator which implements a closed loop system of Users and Web Server. The simulator is implemented in python and follows a modular design.

## Project Structure
`classes.py` file contains class definitions for all internal modules of the system.

`simulator.py` file contains simulator class definition, which uses modules from `classes.py` to instantiate a simulator.

`lcgrand.py` file defines the pseudo random number generator used by the simulator.

`data_generator.py` file runs the simulator for different configurations and generates csv files of the metrics.

`graph_gen.py` file reads the csv files and generated graphs.

## Generating Simulation Data
Requirements: 
 - `python3`

`data_generator.py` defines variable for the simulation metrics. Change this metrics as per requirement and then save the file.

Execute `python3 data_generator.py` to generate the data.

## Generating Graphs
Requirements: 
 - `python3`
 - `matplotlib (pip3 install matplotlib)`
 - Generated csv data files.

Execute `python3 graph_gen.py`. Graphs will be generated.
