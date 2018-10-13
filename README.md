# Berkeley DeepRLcourse Homework in Pytorch
## Introduction

In recent years, with the booming of deep learning, reinforcement learning has made great progress in solving complex tasks and has attracted more and more people`s attention. Also, many researchers start applying reinforcement learning algorithms to solve the problem in other fields (such as Natural Language Processing).

So, there is a big need for learning those classic reinforcement learning algorithms in an easy way.

As beginners in reinforcement learning, we found that [CS 294-112](http://rail.eecs.berkeley.edu/deeprlcourse/) at UC Berkeley is a great course where we can learn a lot of classic and advanced reinforcement learning algorithms.

As the saying goes, “talk is cheap, show me your code.” It is very important to write algorithm in code correctly, instead of just knowing the algorithm. Luckily, CS 294-112 also provides programming assignments for those reinforcement learning algorithms. While, these assignments are mainly implemented in **TensorFlow**, which might be bad news for people who are more familiar with other deep learning frameworks.

For the reasons above, we modified those assignments (for Fall 2018) and implemented in **PyTorch**, which is a framework that we often use in our research. 

Moreover, we also provide [solutions](https://github.com/KuNyaa/berkeleydeeprlcourse-homework-pytorch-solution) to these assignments, and you can use them when you get stuck.

Hope you will enjoy it : )



## What can you learn from it?

- ### HW1: Imitation Learning

  In this assignment, you will implement the **Behavioral Cloning** and **DAgger** algorithm. 

  In the experiments, you will see the case where Behavioral Cloning work well, and the case where DAgger can learn a better policy than Behavioral Cloning.

- ### HW2: Policy Gradients

  In this assignment, you will implement the **Policy Gradients** algorithm.

  In the experiments, you will compare the difference between gradient estimators(full-trajectory case and reward-to-go case) and learn how batch size and learning rate can affect the algorithm performance. Moreover, you will implement a **neural network baseline** to help the gradient estimator to reduce variance and assist the agent to learn a better policy.

- ### HW3: Q-Learning and Actor-Critic

  In this assignment, you will implement the **Deep Q-learning** and **Actor-Critic** algorithm.

  In the Deep Q-learning part, you will implement **vanilla DQN** and **double DQN** and compare their performance in different atari game environments. Also, you will experiment how hyperparameters affect the final results.

  In the Actor-Critic part, you will implement a **Actor-Critic** model based on your Policy Gradients implementations in HW2. Additionally, you will learn how to tune the hyperparameters for the Actor-Critic model, and make it outperform your previous Policy Gradients model which is equiped with reward-to-go gradient estimator and neural network baseline.

- ### HW4: Model-Based RL

  ###### Coming Soon......

- ### HW5: Advanced Topics

  ###### Coming Soon......



## How can you use it?

#### If you want to learn：

- ##### The whole course:

  You can just follow the course syllabus, and use this as programming assignments.

- ##### Policy Optimization style RL algorithm:

  You may want to finish the HW2 and the Actor-Critic part in HW3, and read relative material from the course website.

- ##### Dynamic Programming style RL algorithm:

  You may want to finish the Deep Q-learning part in HW3, and read relative material from the course website.

#### Or you can just use it as you like : )