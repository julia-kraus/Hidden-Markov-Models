import numpy as np
import hidden_markov

"""
simple test: three different states, two different outputs (N=2, np.unique(observations) = 2)
"""

pi = np.array([[0.3, 0.7]]).T

trans = np.array([[0.2, 0.8],
                  [0.5, 0.5]])

obs = np.array([[0.1, 0.9],
                [0.6, 0.4]])

data = [0, 1]

d = hidden_markov.HMM(trans, obs, pi)
print(d.forward(data))
print(d.viterbi(data))
# print(d.backward(data))

# pi = np.array([[0.3, 0.7]]).T
# e = viterbi.Decoder(pi, trans, obs)
# e.decode(data)
