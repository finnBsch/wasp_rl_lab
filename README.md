# Lab - Reinforcement Learning with a Quadruped

## Overview

In this lab you will be asked complete and tune a code base for training a four legged robot in the Mujoco Simulator (the simulation environment is based on JAX, a python library by Google that is designed for numerical computing and machine learning tasks).
This includes implementing a component of the PPO algorithm, and optionally tuning the cost function to instruct the robot to walk over an elevated step.

The lab will have two different levels, 'rl' and 'extra' (optional). You need to finish the rl part to pass this lab, and you can optionally try the extra level.

## Getting Started
We provide instructions for running this on Berzelius, using the VSCode browser version.

### Log into Berzelius
We will provide you with a project for Berzelius.

You will need three terminal windows: One to launch the vscode server, one to forward the VSCode port, and one to fetch the vscode password.

In the first window, login to Berzelius:

```ssh <username>@berzelius2.nsc.liu.se```

Then, clone the lab repository to your home directory:

```
git clone https://github.com/finnBsch/wasp_rl_lab.git
```

and start an interactive session on Berzelius (see [here](https://www.nsc.liu.se/support/systems/berzelius-gpu/#3-interactive-sessions) for more info):

```
interactive --gpus=1 -A Berzelius-2025-241 -t 00-03:00:00
```
You can replace the project (Berzelius-2025-241) with your own, if you have access to another project.

This will give you an interactive GPU session for 3hours. Please don't forget to close this terminal once you're done to free ressources.

Once logged in, we need to start the VSCode-Server (see [here](https://www.nsc.liu.se/support/systems/berzelius-software/berzelius-vscode/) for more details):

```
module load VSCode-Server/latest-bdist
code-server --bind-addr node0xy:9988 ./wasl_rl_lab
```
where 'node0xy' should be the node you were assigned.

In a new terminal, forward the vscode port using

```
ssh -N -L localhost:9988:node0xy:9988 <your username>@berzelius1.nsc.liu.se
```

and in your browser open
```
http://localhost:9988
```

Open on last terminal window and again ssh into berzelius and read the contents of
```
/home/<your username>/.config/code-server/config.yaml
```
to get the vscode password.

### Inside VSCode
Inside VSCode, firstly open the cloned repo. Then, inside VSCode, open a terminal, and load the following modules:

```
module load Mambaforge
module load buildenv-gcccuda
module load CMake
```

Then, create a new mamba environment:
```
mamba create -n rl_env python=3.11
mamba activate rl_env
```

Inside the mamba env, please run the following commands in the terminal:
```
mamba install ffmpeg
python3 -m pip install matplotlib opencv-python mediapy jax[cuda12] jupyterlab
```

Furthermore install these dependencies (note: the order matters here, so please stick to it, if something seems to go wrong, just restart from the first one and run them all again):
```
python3 -m pip install git+https://github.com/finnBsch/mujoco_playground.git@lab1_rl
python3 -m pip uninstall mujoco-mjx
python3 -m pip install git+https://github.com/finnBsch/mujoco.git@lab#subdirectory=mjx
```

Lastly, install the VSCode python extension and configure the created mamba/conda env as the interpreter. Then, open 'lab_rl.ipynb' and follow the tasks in the notebook.

Note that all these setup steps will be persistent and hence, when logging into Berzelius again, you won't have to redo them, except for loading the modules and activating the mamba env. Don't forget to save the notebook to save the outputs/changes. Also, make sure to download the result videos to avoid losing them.


## Related Resources

- PPO paper (OpenAI): https://arxiv.org/pdf/1707.06347
- Perceptive Quadruped RL (ETH Zurich): https://arxiv.org/pdf/2201.08117
- Non-perceptive Quadruped RL (ETH Zurich): https://arxiv.org/pdf/2010.11251
