# Missile Guidance Using Reinforcement Learning

This repository contains the implementation of a Reinforcement Learning Based Missile Guidance Algorithm, as described in the research paper titled "Missile Guidance and Control Against Maneuvering Targets Using Deep Reinforcement Learning" by Arijit Bhowmik, Twinkle Tripathy, and Tushar Sandhan. You can access the research paper [here](https://acrobat.adobe.com/link/review?uri=urn:aaid:scds:US:81dc3665-0d7b-31be-a30b-fc05cf4c6ee0).

The project is ongoing, and the repository currently includes the code for a Proximity Policy Optimization (PPO) based reinforcement learning model. This model is being used to achieve two-dimensional target interception, as mentioned in the research paper.

## Algorithms Implemented

### Deep Deterministic Policy Gradient (DDPG) based Implementation

The deep deterministic policy gradient (DDPG) algorithm is an online, model-free, off-policy reinforcement learning (RL) method. It explores an optimal policy that maximizes the expected cumulative long-term reward. The DDPG agent is an actor-critic RL agent, with both the actor and critic functions constructed using neural networks.

In our missile guidance scenario, we have successfully implemented the DDPG algorithm based on the work described in the research paper "Missile Guidance and Control Against Maneuvering Targets Using Deep Reinforcement Learning" by Arijit Bhowmik, Twinkle Tripathy, and Tushar Sandhan. The implementation has achieved the desired results in terms of effective missile guidance and control against maneuvering targets.

The DDPG algorithm utilizes the policy gradient method to find a deterministic policy. The critic network evaluates a given policy using the action-value function, and the actor network updates the policy using the critic's evaluation. This iterative process continues until the algorithm converges.

To address the convergence issue when implementing Q learning with neural networks, we employ soft update rules for the target actor and critic networks. This means that the weights of the target networks (w′c and w′a) are gradually updated based on the weights of the parent networks (wc and wa) using an update rate (τ). This soft update approach helps stabilize the learning process.

In DDPG, we use temporal difference to approximate the error of the action-value function, denoted as tde = (y − Q(s, a|wc)), where y is the target action-value function obtained using the target actor and critic networks. The loss function (l) is defined as the square of the tde, i.e., l = tde^2. This loss function is used to update the critic network.

By successfully implementing the DDPG algorithm based on the research paper's work and achieving the desired results, we have demonstrated the effectiveness of reinforcement learning for missile guidance and control against maneuvering targets.



### Proximity Policy Optimization (PPO) based Implementation
The PPO algorithm with Generalized Advantage Estimation has been fully implemented. However, we are currently working on leveraging this algorithm to achieve the desired results. PPO has gained popularity in the reinforcement learning community, and we aim to utilize its potential for successful missile target capture.

Please note that the repository is a work in progress, and updates will be made to enhance the algorithm's performance and achieve optimal results.

Feel free to explore the code and provide any feedback or suggestions.

## References

Please find below the references for the research papers and implementations related to the Missile Guidance Using Reinforcement Learning project:

1. "Missile Guidance and Control Against Maneuvering Targets Using Deep Reinforcement Learning" by Arijit Bhowmik, Twinkle Tripathy, and Tushar Sandhan.

2. "Continuous Control with Deep Reinforcement Learning" by Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Google DeepMind.

3. "Computational Missile Guidance: A Deep Reinforcement Learning Approach" by Shaoming He, Hyo-Sang Shin, and Antonios Tsourdos. Beijing Institute of Technology and Cranfield University.

4. "Proximal Policy Optimization Algorithms" by John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. OpenAI.

5. "High-Dimensional Continuous Control using Generalized Advantage Estimation" by John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter Abbeel. University of California, Berkeley.

### Implementations:

1. [Continuous Action State PPO Algorithm implementation](https://github.com/nric/ProximalPolicyOptimizationContinuousKeras)  By [nric](https://github.com/nric/ProximalPolicyOptimizationContinuousKeras/commits?author=nric)

2. [DDPG Algorithm TensorFlow Implementation](https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning) By [philtabor](https://github.com/philtabor/Youtube-Code-Repository/commits?author=philtabor)

Please refer to these sources for further details on the research papers and implementations related to the project.

