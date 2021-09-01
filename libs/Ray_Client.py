import copy
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from utils.data_utils import index_bootstrap, mini_batches

import ray


@ray.remote
class RayClient:
    """
    params = {'E': 100, 'kernel_size': 10, 'lr': 0.01, 'opt': 'sgd', 'd_out': 1}
    """

    def __init__(self, parameters, id_num=0):
        self.k = parameters['kernel_size']
        self.local_E = parameters['E']
        self.lr = parameters['lr']
        self.optimizer = parameters['opt']
        self.d_out = parameters['d_out']
        self.alpha = 1.0
        self.id_num = id_num
        self.nk = 0

        # centers, spreads, w and b can be synchronized with the server
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

    @staticmethod
    def sort_data(original_x, original_y, num_select):
        data_index = np.argsort(original_y.flatten())
        return original_x[data_index[0:num_select], :], original_y[data_index[0:num_select], :]

    def data_size(self):
        """
        output the data size
        call after train function
        :return:
        """
        return copy.deepcopy(self.nk)

    def find_center(self, train_x):
        """
        :param train_x:
        :return:
        """
        k_means = KMeans(n_clusters=self.k).fit(train_x)
        centers = k_means.cluster_centers_
        centers, tmp_index = self._sort_centers(centers)
        AllDistances = self._dist(centers, centers.T)
        dMax = AllDistances.max(initial=-10)

        for i in range(self.k):
            AllDistances[i, i] = dMax + 1
        AllDistances = np.where(AllDistances != 0, AllDistances, 0.000001)
        std = self.alpha * np.min(AllDistances, axis=0)
        return centers, std

    def synchronize(self, server):
        """
        synchronize with the server
        :param server:
        :return:
        """
        # self.centers = copy.deepcopy(server['centers'])
        self.w = copy.deepcopy(server['w'])
        self.b = copy.deepcopy(server['b'])
        # self.std = copy.deepcopy(server['std'])

        # print(self.centers[0])

        # This will show the impact of center-mismatching
        # tmp = copy.deepcopy(self.centers[0])
        # tmp1 = copy.deepcopy(self.centers[1])
        # self.centers[1] = tmp
        # self.centers[0] = tmp1

    def compute_update(self):
        tmp_c, tmp_w, tmp_b, tmp_s = copy.deepcopy(self.centers), copy.deepcopy(self.w), \
                                     copy.deepcopy(self.b), copy.deepcopy(self.std)
        return tmp_c, tmp_w, tmp_b, tmp_s, self.nk

    def train(self, train_x, train_y):
        # training
        self.nk = train_x.shape[0]

        self.centers, self.std = self.find_center(train_x)
        # self.std = np.ones(self.std.shape)

        epoch_id = (i for i in range(self.local_E))
        Distance = self._dist(self.centers, train_x.T)
        SpreadsMat = np.tile(self.std.reshape(-1, 1), (1, self.nk))
        A = np.exp(-(Distance / SpreadsMat) ** 2).T

        if self.optimizer == 'sgd':
            # SGD
            # for each sample, loss = (y' - y)**2 / 2 = (wx+b - y)**2/2
            # dw = (wx+b -y)*x, db = wx+b
            samp_id = [i for i in range(train_x.shape[0])]

            for _ in epoch_id:
                for i in samp_id:
                    F = A[i].T.dot(self.w) + self.b
                    error = (F - train_y[i]).flatten()
                    dw = A[i].reshape(-1, 1) * error.reshape(1, self.d_out)
                    db = error
                    # update
                    self.w = self.w - self.lr * dw
                    self.b = self.b - self.lr * db


        elif self.optimizer == 'max-gd':
            # max error gd
            # for each batch, loss = sum(y' - y)**2 / (2m) = sum(wx+b - y)**2/(2m), m is the batch size
            # dw = sum(wx+b - y)/m, db = sum(wx+b)/m

            for _ in epoch_id:
                F = A.dot(self.w) + self.b
                error = F - train_y
                sum_error = np.sum(error ** 2, axis=1)
                max_index = np.argmax(sum_error)
                dw = A[max_index].reshape(-1, 1) * error[max_index].reshape(1, self.d_out)
                db = error[max_index].reshape(1, self.d_out)

                self.w = self.w - self.lr * dw
                self.b = self.b - self.lr * db

        elif self.optimizer == 'm-sgd':
            # mini-batch SGD
            # for each batch, loss = sum(y' - y)**2 / (2m) = sum(wx+b - y)**2/(2m), m is the batch size
            # dw = sum(wx+b - y)/m, db = sum(wx+b)/m
            batch_size = 12
            for _ in epoch_id:

                batches = mini_batches(train_x, train_y, A, batch_size, 1234)

                for batch_ in batches:
                    batch_x, batch_y, A_tmp = batch_[0], batch_[1], batch_[2]
                    real_bs = batch_x.shape[0]
                    F = A_tmp.dot(self.w) + self.b
                    error = F - batch_y
                    tmp_A = A_tmp.reshape(real_bs, -1, 1)
                    tmp_error = error.reshape(real_bs, -1, self.d_out)
                    out = tmp_A * tmp_error
                    dw = np.sum(out, axis=0) / real_bs
                    db = np.sum(error, axis=0).reshape(1, self.d_out) / real_bs

                    self.w = self.w - self.lr * dw
                    self.b = self.b - self.lr * db

        elif self.optimizer == '1-sgd':
            # one batch SGD
            # for each batch, loss = sum(y' - y)**2 / (2m) = sum(wx+b - y)**2/(2m), m is the batch size
            # dw = sum(wx+b - y)/m, db = sum(wx+b)/m
            wanted_size = 32
            prob = wanted_size / self.nk
            for _ in epoch_id:
                batch_index = index_bootstrap(self.nk, prob)
                batch_x, batch_y = train_x[batch_index], train_y[batch_index]
                batch_size = batch_x.shape[0]
                A_tmp = A[batch_index]

                F = A_tmp.dot(self.w) + self.b
                error = F - batch_y
                tmp_A = A_tmp.reshape(batch_size, -1, 1)
                tmp_error = error.reshape(batch_size, -1, self.d_out)
                out = tmp_A * tmp_error
                dw = np.sum(out, axis=0) / batch_size
                db = np.sum(error, axis=0).reshape(1, self.d_out) / batch_size

                self.w = self.w - self.lr * dw
                self.b = self.b - self.lr * db

        else:
            print('Error in loss name')

    def predict(self, test_x):
        N = test_x.shape[0]
        TestDistance = self._dist(self.centers, test_x.T)
        TestSpreadMat = np.tile(self.std.reshape(-1, 1), (1, N))
        TestHiddenOut = np.exp(-(TestDistance / TestSpreadMat) ** 2).T
        Test_y = np.dot(TestHiddenOut, self.w) + self.b
        return Test_y
