import numpy as np

"""
author: Julia Kraus, 05.12.2017
forward algorithm for Hidden Markov Models
N: number of hidden states

"""

"""
initial_prob: initial probability, shape = 1*N
transition_prob: transition probability between the N states, shape: N*N
observation_prob: observation probability, shape: N* # of possible observations
"""


class ForwardLikelihood(object):
    def __init__(self, initial_prob, transition_prob, observation_prob):
        self.N = initial_prob.shape[0]
        self.initial_prob = initial_prob
        self.transition_prob = transition_prob
        self.observation_prob = observation_prob

    def get_observation_prob(self, observation):
        return self.observation_prob[:, observation]

    def get_likelihood_observation_sequence(self, observations):
        forward_prob = np.zeros((self.N, len(observations)))
        print(forward_prob.shape)
        # initialization step
        forward_prob[:, 0] = self.initial_prob * self.get_observation_prob(observations[0])

        # recursion step
        for t in range(1, len(observations)):
            forward_prob[:, t] = np.sum(forward_prob[:, t - 1, None] * (
                self.get_observation_prob(observations[t]).reshape(1, 2)) \
                                        * self.transition_prob, axis=0)

        # termination step
        likelihood = np.sum(forward_prob[:, len(observations) - 1])

        return likelihood
