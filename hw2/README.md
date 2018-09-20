# CS294-112 HW 2: Policy Gradient

Modification:

In general, we followed the code structure of the original version and modified the neural network part to pytorch. 

Because of the different between the static graphs framework and the dynamic graphs framework, we merged and added some code in `train_pg_f18.py`. We also adapted the instructions of this assignment for pytorch. (Thanks to CS294-112 for offering ![equation](http://latex.codecogs.com/gif.latex?\LaTeX) code for the instructions) And you can just follow the pytorch version instructions we wrote.

------

Dependencies:

 * Python **3.5**
 * Numpy version **1.14.5**
 * Pytorch version **0.4.0**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==**2.3.2**

Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

The only file that you need to look at is `train_pg_f18.py`, which you will implement.

See the [HW2 PDF](./hw2_instructions.pdf) for further instructions.
