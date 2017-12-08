import numpy as np

"""
hidden_markov.py
Author: Julia Kraus
Date: 8 Dec 2017

Implementation of the Hidden Markov Model after "A Tutorial on Hidden Markov Models and Selected Applications in Speech
Recognition" by Daniel Jurafsky and James H. Martin. 

"""


class HMM(object):
    def __init__(self, A, B, pi):
        # HMM is (A, B, pi) with A = transition probabilities, B = emission probabilities, pi = initial distribution
        self.A = A
        self.B = B
        self.pi = pi
        self.N = pi.shape[0]  # number of hidden states in the model

    def viterbi(self, obs):
        """Returns most likely sequence of hidden states, given the observations and the HMM"""
        T = len(obs)
        trellis = np.zeros((self.N, T))
        backpointer = np.ones((self.N, T), 'int32') * -1
        # initialization
        trellis[:, 0] = np.squeeze(self.pi * self.B[:, obs[0]].reshape(-1, 1))

        for t in range(1, T):
            trellis[:, t] = np.max(
                np.dot(trellis[:, t - 1].reshape(-1, 1), self.B[:, obs[t]].reshape(1, -1)) * self.A,
                axis=0)
            backpointer[:, t] = (np.tile(trellis[:, t - 1, None], [1, self.N]) * self.A).argmax(0)
        # termination
        max_prob_sequence = [trellis[:, -1].argmax()]
        for i in range(T - 1, 0, -1):
            max_prob_sequence.append(backpointer[max_prob_sequence[-1], i])
            # return sequence with maximum probability and maximum probability
        return max_prob_sequence[::-1], trellis[:, -1].max()

    def forward(self, obs):
        T = len(obs)
        forward_trellis = np.zeros((self.N, T))
        # initialization step
        forward_trellis[:, 0] = np.squeeze(self.pi * self.B[:, obs[0]].reshape(-1, 1))
        # recursion step
        for t in range(1, T):
            forward_trellis[:, t] = np.sum(
                np.dot(forward_trellis[:, t - 1].reshape(-1, 1), self.B[:, obs[t]].reshape(1, -1)) * self.A,
                axis=0)
        # non-vectorized implementation
        # for t in range(1, T):
        #     for j in range(0, self.N):
        #         forward_trellis[j, t] = np.dot(forward_trellis[:, t - 1].reshape(1, self.N),
        #                                        self.A[:, j].reshape(self.N, 1)) * self.B[j, obs[t]]

        # termination step
        forward_likelihood = np.sum(forward_trellis[:, T - 1])

        return forward_likelihood, forward_trellis

    def backward(self, obs):

        T = len(obs)
        backward_trellis = np.zeros((self.N, T))

        # initialization step
        backward_trellis[:, T - 1] = 1

        # recursion step
        # for t in reversed(range(0, T - 1)):
        #     for i in range(0, N):
        #         backward_trellis[:, t] = self.B[:, backward_trellis[:, t + 1].reshape(1, -1)

        for t in reversed(range(0, T - 1)):
            backward_trellis[:, t] = np.sum(np.dot(backward_trellis[:, t + 1].reshape(-1, 1),
                                                   self.B[:, obs[t + 1]].reshape(1, -1))
                                            * self.A, axis=0)

        # termination step

        backward_likelihood = np.sum(backward_trellis[:, 0].reshape(-1, 1) * self.pi * self.B[:, obs[0]].reshape(-1, 1))

        return backward_likelihood, backward_trellis



        #
        # # def forward_backward(self, observations):
        # #     T = len(observations)
        # #     # initialize transition_prob, observation_prob:
        # #     A = np.ones((self.N, self.N))
        # #     # normalize
        # #     A = A/np.sum(A, 1)
        # #     B = np.ones((self.N, T))
        # #     B = B/np.sum(B, 1)
        # #
        # #     # iterate until convergence:
        # #     while True:
        # #         old_A = A
        # #         old_B = B
        #
        #         # expectation step
        #         # gamma_values
        #         gamma_ =
