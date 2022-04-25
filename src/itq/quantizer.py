from .pca import PCA
from .ibq import IBQ


class IterativeQuantizer:
    def __init__(self, num_bits, num_iterations):
        self.pca = PCA(num_bits)
        self.ibq = IBQ(num_iterations)

    def fit(self, X):
        V = self.pca.fit_transform(X)
        return self.ibq.fit(V)

    def quantize(self, X):
        V = self.pca.transform(X)
        return self.ibq.quantize(V)