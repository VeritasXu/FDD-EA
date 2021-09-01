import numpy as np


class Single_AF:
    def __init__(self, server):
        self.server = server

    def LCB(self, new_pops, *arg):
        """
        :param new_pops:
        :param arg: selected_clients, ac_type: fe_type, str, contains: LCB, EI, combined
        :return:
        """
        selected_clients = arg[0]
        ac_type = arg[1]
        LG_type = arg[2]

        num_pop = new_pops.shape[0]
        num_clients = len(selected_clients)

        f_g_hat = self.server.predict(new_pops).reshape(num_pop, 1)

        f_l_hat = np.zeros((num_pop, num_clients))
        for i, c in enumerate(selected_clients):
            f_l_hat[:, i] = c.predict(new_pops).flatten()

        if LG_type == 'L':
            f_mean_hat = np.mean(f_l_hat, axis=1).reshape(num_pop, 1)
            tmp = (f_l_hat - f_mean_hat) ** 2
            s_2_hat = np.sum(tmp, axis=1, keepdims=True) / (f_l_hat.shape[1] - 1)


        elif LG_type == 'G':
            f_mean_hat = f_g_hat
            tmp = (f_l_hat - f_mean_hat) ** 2
            s_2_hat = np.sum(tmp, axis=1, keepdims=True) / (f_l_hat.shape[1] - 1)

        elif LG_type == 'LG':
            local_mean = np.mean(f_l_hat, axis=1).reshape(num_pop, 1)
            combined_sum = np.hstack((local_mean, f_g_hat))
            f_mean_hat = np.mean(combined_sum, axis=1).reshape(-1, 1)

            combined_f_hat = np.hstack((f_l_hat, f_g_hat))
            tmp = (combined_f_hat - f_mean_hat) ** 2

            s_2_hat = np.sum(tmp, axis=1, keepdims=True) / (combined_f_hat.shape[1] - 1)

        else:
            f_mean_hat = None
            s_2_hat = None
            print('Error! The type of mean ± std should be chosen from {L, G, LG}')

        s_hat = np.sqrt(s_2_hat)

        # step 2: LCB
        w = 2
        LCB_matrix = (f_mean_hat - w * s_hat).flatten()

        # step 3: ExI

        if ac_type == 'LCB':
            return LCB_matrix
        else:
            print('Error! The type of acquisition function should be chosen from {LCB}')