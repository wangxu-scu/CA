import numpy as np
from scipy.stats import ortho_group


class IBQ:
    def __init__(self, num_iterations):
        self.num_iterations = num_iterations

    def fit(self, V):
        (num_rows, num_bits) = V.shape

        self.R = ortho_group.rvs(num_bits)

        self.quantization_losses = []
        for i in range(self.num_iterations):
            # Fix R and update B.
            B = np.ones((num_rows, num_bits))
            B[np.dot(V, self.R) < 0] = 0

            # Fix B and update R.
            S, _, S_hat_T = np.linalg.svd(np.dot(B.T, V))
            self.R = np.dot(S, S_hat_T).T

            # Append quantization loss.
            self.quantization_losses.append(np.linalg.norm(B - np.dot(V, self.R)))

    def quantize(self, V):
        B = np.ones(V.shape)
        B[np.dot(V, self.R) < 0] = 0
        return B

    def fit_quantize(self, V):
        self.fit(V)
        return self.quantize(V)