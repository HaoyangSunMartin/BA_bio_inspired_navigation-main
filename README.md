# BA_bio_inspired_navigation

Welcome to the Git Repo for the Bachelor Thesis on

> Biologically Inspired Spatial Navigation using vector-based and topology-based path planning 

In this file we will cover the installation process to get the code running and touch upon the meaning of the different files.
For a detailed description of the functionality and thoughts behind this code, please refer to the thesis.

## Install packages
The code is based on Python3. You will need to install the following packages to run the code:

- pip
- pybullet
- matplotlib
- numpy
- scipy

Also make sure to have latex installed to be able to export plots.

## Some more setup
If you want to initialize the model with the data described in the thesis, go to:
https://syncandshare.lrz.de/getlink/fiFTBuDHjvcNosMhEEiddMEC/
There you can download the model of grid cells, place cells and cognitive map. Copy-paste it in the data folder. 
Make sure to set in main.py the variable **from_data = True** in the gc model and other models.

When running experiments, double check where the data will be saved and avoid that it overwrites your previous runs.

## What to do with the code?
The code was designed to perform and analyze the experiments described in the thesis.
With the **main.py** the main experiments can be executed.

### Grid Cell Decoder Test
- As environment choose **env_model = "single_line_traversal"**. 
- Pick a decoder of your choice via the variable **vector_model**.
- Set the number of trials you want to execute.
- Set **nr_steps = 8000** and **nr_steps_exploration = 3500**.
- Double check what data you want to save of each trial.

### Maze test
Here it makes sense to decouple exploration and navigation phase.
- Set the number of trials you want to execute to 1.
- First do a run with **nr_steps = 15000** and **nr_steps_exploration = nr_steps**.
- Save the cognitive map at the end of the run (or use the data provided with lrz folder)
- Double check what data you want to save of each trial.
- Adapt the environment if wanted and do a run with **nr_steps = 8000** and **nr_steps_exploration = 0**.

### What else?
In the folder **/other_exec** you'll find a file to evaluate the data from your runs, 
as well as simplified executables to test subcomponents of the system.

You also have the option to create videos with the **main.py** file. This is why the loop is structured in this specific way.

## Code Structure
Note that some folders are git-ignored but will be created when running the script.

    .
    ├── data                    # Optional initialization data for gc, pc, cm
    ├── environment             # Environment files for pybullet scene. Edit .urdf files if wished
    ├── experiments             # Here it will save plots, when you analyze run data 
    ├── ffmpeg                  # Needed for video creation
    ├── other_exec              # Executables to analyze run data or test subcomponents
    ├── p3dx                    # Agent files for pybullet
    ├── plots                   # Here it will save plots during linear lookaheads if you want
    ├── plotting                # Script to create plots
    │   ├── plotThesis.py       # TUM themed and labeled plots used for the thesis
    │   ├── plotResults.py      # WIP plots to check results and scripts to create video
    │   ├── plotHelper.py       # Some helper functions used by plotResults.py
    │   └── tum_logo.png        # TUM Logo    
    ├── system                  # Scripts that make up the system
    │   ├── bio_model           # Scripts modeling grid cells, place cells and the cognitive map cells
    │   │── controller          # Navigation phase, Exploration Phase controller and Agent controller
    │   ├── decoder             # Scripts for different grid cell decoder mechanism
    │   └── helper.py           # Some helper function used across scripts
    ├── videos                  # Here it will save the video
    ├── main.py                 # Execute this file for main experiments
    ├── README.md               # You are here. Overview of project
    └── Biologically_Inspired...pdf           # Bachelor thesis of Haoyang Sun explaining thoughts and theory behind code

## Questions left?
Any questions left or having troubles with the code? Feel free to reach out to tim.engelmann@tum.de



