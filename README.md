# Berkeley DeepRLcourse Homework in Pytorch
## Introduction

In recent years, with the booming of deep learning, reinforcement learning has made great progress in solving real life tasks and has attracted more and more people`s attention. Also, many researchers start applying reinforcement learning algorithms to solve the problem in other fields (such as Natural Language Processing).

So, there is a big need for learning those classic reinforcement learning algorithms in an easy way.

As beginners in reinforcement learning, we found that [CS 294-112](http://rail.eecs.berkeley.edu/deeprlcourse/) at UC Berkeley is a great course where we can learn a lot of classic and advanced reinforcement learning algorithms.

As the saying goes, “talk is cheap, show me your code.” It is very important to write algorithm in code correctly, instead of just knowing the algorithm. Luckily, CS 294-112 also provides programming assignments for those reinforcement learning algorithms. While, these assignments are mainly implemented in **TensorFlow**, which might be bad news for people who are familiar with other deep learning frameworks.

For the reasons above, we modified those assignments (for Fall 2018) and implemented in **Pytorch**, which is a framework that we often use in our research. 

Moreover, we also wrote [solutions](https://github.com/KuNyaa/berkeleydeeprlcourse-homework-pytorch-solution) to these assignments, and you can use them when you get stuck.

Hope you can enjoy it : )



## What can you learn from it?

- ### HW1: Imitation Learning

  In this assignment, you will implement **Behavioral Cloning** and **DAgger** algorithm. 

  In the experiments, you will see the case where Behavioral Cloning work well, and the case where DAgger can learn a better policy than Behavioral Cloning.

- ### HW2: Policy Gradients

  In this assignment, you will implement **Policy Gradients** algorithm.

  In the experiments, you will compare the difference between gradient estimators(full-trajectory case and reward-to-go case) and learn how batch size and learning rate can affect the algorithm performance. Moreover, you will implement a **neural network baseline** to help the gradient estimator to reduce variance and assist the agent to learn a better policy.

- ### HW3: Q-Learning and Actor-Critic

  ###### Coming Soon......

- ### HW4: Model-Based RL

  ###### Coming Soon......

- ### HW5: Advanced Topics

  ###### Coming Soon......