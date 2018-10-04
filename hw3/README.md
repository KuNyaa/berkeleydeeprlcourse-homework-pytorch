# CS294-112 HW 3: Q-Learning

Modifications:

In general, we followed the code structure of the original version and modified the neural network part to pytorch. 

Because of the different between the static graphs framework and the dynamic graphs framework, we merged and added some code. For the instructions, you can generally follow the original PDF version, and we have adapted the comments in the code for pytorch to help you finish this assignment.

------

Dependencies:

 * Python **3.5**
 * Numpy version **1.14.5**
 * Pytorch version **0.4.0**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==**2.3.2**
 * OpenCV
 * ffmpeg

Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

The only files that you need to look at are `dqn.py` and `train_ac_f18.py`, which you will implement.

See the [HW3 PDF](./hw3_instructions.pdf) for further instructions.

The starter code was based on an implementation of Q-learning for Atari generously provided by Szymon Sidor from OpenAI.
