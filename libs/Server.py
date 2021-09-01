import copy
import numpy as np
from scipy.spatial.distance import cdist


class Server:
    def __init__(self, parameters):
        super().__init__()
        self.k = parameters['kernel_size']
        self.alpha = 1.0
        self.d_out = parameters['d_out']
        # centers, spreads, w and b can be broadcast to clients
        self.centers, self.std = None, None
        self.w = np.random.randn(self.k, self.d_out)
        self.b = np.random.randn(1, self.d_out)

    @staticmethod
    def _sort_centers(centers):
        """
        To sort the centers according to the distance from zero vector
        Please note that this fun has not consider the direction of the centers, should be change
        :param centers:
        :return: sorted centers & index
        """
        tmp_centers = copy.deepcopy(centers)
        distance = np.sum(tmp_centers ** 2, axis=1)

        sorted_index = np.argsort(distance)

        tmp_centers = tmp_centers[sorted_index, :]

        return tmp_centers, sorted_index

    @staticmethod
    def _dist(Mat1, Mat2):
        """
        rewrite euclidean distance function in Matlab: dist
        :param Mat1: matrix 1, M x N
        :param Mat2: matrix 2, N x R
        output: Mat3. M x R
        """
        Mat2 = Mat2.T

        return cdist(Mat1, Mat2)

    def average(self, selected_clients):
        stack_c, stack_w, stack_b, stack_s = [], [], [], []
        num_data = 0

        for i, client in enumerate(selected_clients):
            tmp_c, tmp_w, tmp_b, tmp_s = client.compute_update()
            nk = client.data_size()
            num_data += nk

            if i == 0:
                stack_c, stack_w, stack_b, stack_s = nk * tmp_c, nk * tmp_w, \
                                                     nk * tmp_b, nk * tmp_s
            else:
                # stack_c = np.vstack((stack_c, tmp_c))
                stack_c += nk * tmp_c
                stack_w += nk * tmp_w
                stack_b += nk * tmp_b
                stack_s += nk * tmp_s

        # k_means_c = KMeans(n_clusters=self.k).fit(stack_c)
        # self.centers = k_means_c.cluster_centers_
        self.centers = stack_c / num_data
        self.centers, tmp_index = self._sort_centers(self.centers)
        # self.w, self.b, self.std = stack_w / num_data, stack_b / num_data, stack_s / num_data
        self.w, self.b, self.std = stack_w[tmp_index] / num_data, stack_b / num_data, stack_s[tmp_index] / num_data

    def predict(self, test_x):
        N = test_x.shape[0]
        TestDistance = self._dist(self.centers, test_x.T)
        TestSpreadMat = np.tile(self.std.reshape(-1, 1), (1, N))
        TestHiddenOut = np.exp(-(TestDistance / TestSpreadMat) ** 2).T
        Test_y = np.dot(TestHiddenOut, self.w) + self.b
        return Test_y

    def broadcast(self):
        tmp_c, tmp_w, tmp_b, tmp_s = copy.deepcopy(self.centers), copy.deepcopy(self.w), \
                                     copy.deepcopy(self.b), copy.deepcopy(self.std)
        return {'centers': tmp_c, 'w': tmp_w, 'b': tmp_b, 'std': tmp_s}
