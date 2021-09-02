import copy
import time
import numpy as np
import pandas as pd
from EA.RCGA import RCGA
from libs.AF import Single_AF
from datetime import datetime
from config import args_parser
from libs.Server import Server
from libs.Client import Client
from libs.Ray_Client import RayClient
from utils.data_sop import init_samples
from utils.data_utils import sort_data

import ray

ray.init(include_dashboard=False)

# -------------Debug mode---------------------------#
# --------Modify hyper-parameters in config----------------#

# parse args
args = args_parser()

if args.func == 'Ackley':
    from SOP.Ackley import Ackley as Function

    lb, ub = -32.768, 32.768
elif args.func == 'Griewank':
    from SOP.Griewank import Griewank as Function

    lb, ub = -600, 600
elif args.func == 'Ellipsoid':
    from SOP.Ellipsoid import Ellipsoid as Function

    lb, ub = -5.12, 5.12
elif args.func == 'Rastrigin':
    from SOP.Rastrigin import Rastrigin as Function

    lb, ub = -5, 5
elif args.func == 'Rosenbrock':
    from SOP.Rosenbrock import Rosenbrock as Function

    lb, ub = -2.048, 2.048
elif args.func == 'Schwefel':
    from SOP.Schwefel import Schwefel as Function

    lb, ub = -500, 500
elif args.func == 'F13':
    from SOP.cec2005.F13 import CEC_F13 as Function

    lb, ub = -5, 5

else:
    Function = None
    lb, ub = None, None
    print('Error in Test Function')

# total number of clients
N = args.num_users
num_T = int(args.frac * N)
# dimension of decision variable
d = args.d
M = 1
num_b4_d = 5
num_data = num_b4_d * d
boot_prob = args.boot_prob
gens = args.max_gens

up_bound = 1e5

kernel_size = 2 * d + 1
best_pop = []
multi_ub = ub * np.ones(d)
multi_lb = lb * np.ones(d)

# 20 independent rounds
Max_IR = args.runs

round_best_fit = np.ones(Max_IR) * up_bound

# record 20 profiles
record_profiles = np.zeros((11 * d, Max_IR))

if __name__ == '__main__':

    t0 = time.time()

    params = {'E': args.E, 'kernel_size': kernel_size, 'lr': args.lr, 'd_out': M, 'opt': args.opt}

    ray_clients = [RayClient.remote(params, id_num=i) for i in range(num_T)]
    clients = [Client(params, id_num=i) for i in range(num_T)]

    for IR in range(Max_IR):
        # re-LHS training data
        print('\033[1;35;46m Re-LHS ' + args.func + ' d=' + str(d) + ' data, ' + str(IR + 1) + ' run\033[0m')

        # reset random seed
        now = datetime.now()
        clock = 100 * (now.year + now.month + now.day + now.hour + now.minute + now.second)
        np.random.seed(clock)

        chosen_pop = np.zeros((1, 1))

        # define local archives for each client
        Dl_samp, Dl_label = [np.ones((1, 2)) for i in range(N)], [np.ones((1, 1)) for i in range(N)]

        # define initial local dataset for training, never change
        D_samp, D_label = [[] for i in range(N)], [[] for i in range(N)]

        gap = (ub - lb) / N

        train_x, train_y = init_samples(func=Function,
                                        num_b4_d=num_b4_d,
                                        d=d, xlb=lb, xub=ub)

        copy_y = copy.deepcopy(train_y)
        copy_index = np.argsort(-copy_y.flatten())
        record_profiles[0:num_b4_d * d, IR] = copy_y[copy_index, :].flatten()

        current_best = np.min(train_y)

        for i in range(N):
            left_b = lb + i * gap
            right_b = lb + (i + args.tau) * gap

            for j in range(train_x.shape[0]):

                if not left_b <= train_x[j, 0] <= right_b:
                    D_samp[i].append(train_x[j])
                    D_label[i].append(train_y[j])
            D_samp[i] = np.array(D_samp[i]).reshape(-1, d)

            D_label[i] = np.array(D_label[i]).reshape(-1, 1)

        ############################################################################
        #                           optimization                                   #
        #                           optimization                                   #
        ############################################################################

        # count real fitness evaluations
        Real_FE = num_data

        server = Server(params)

        best_fit = up_bound

        idx_users = np.random.choice(range(N), num_T, replace=False)

        # start from 5d, end at 11d
        while Real_FE < 11 * d:

            print('Real FE ', Real_FE)

            t1 = time.time()

            # define a tmp client dataset, using for stack Dl and D
            D_x, D_y = [[] for i in range(num_T)], [[] for i in range(num_T)]

            # 1: get the model of the server
            server_model = server.broadcast()

            for i, idx in enumerate(idx_users):
                # real-evaluate the chosen sample, count the number
                if Dl_samp[idx].shape[1] == d:
                    D_x[i] = np.vstack((D_samp[idx], Dl_samp[idx]))
                    D_y[i] = np.vstack((D_label[idx], Dl_label[idx]))
                else:
                    D_x[i] = D_samp[idx]
                    D_y[i] = D_label[idx]

                if d <= 50:
                    num_select = Real_FE
                else:
                    num_select = 500
                D_x[i], D_y[i] = sort_data(D_x[i], D_y[i], num_select)

                # 2: overwrite the local model
                ray_clients[i].synchronize.remote(server_model)
                # note: it is better to warm up for mini-batch sgd in the first round
                # if args.opt == 'sgd' and Real_FE == num_b4_d * d:
                #     ray_clients[i].warm_up.remote(D_x[i], D_y[i])
                # 3: local training
                ray_clients[i].train.remote(D_x[i], D_y[i])

            all_models = ray.get([ray_clients[i].compute_update.remote() for i in range(num_T)])
            [clients[i].reload(all_models[i]) for i in range(num_T)]

            # local models ---> averaging ---> global model
            server.average(clients)
            t2 = time.time()

            print('update, train time: %.2f' % (t2 - t1))

            FU_LCB = Single_AF(server)
            chosen_pop, _, pop, _1 = RCGA(FU_LCB.LCB,
                                          multi_lb, multi_ub,
                                          args=(clients, 'LCB', args.ac_type),
                                          max_iter=gens,
                                          particle_output=True)

            t3 = time.time()
            print('optimization time: %.2f' % (t3 - t2))

            idx_users = np.random.choice(range(N), num_T, replace=False)

            for idx in idx_users:

                best_fit = Function(chosen_pop.flatten())

                if best_fit <= current_best:
                    current_best = best_fit
                # else:
                #     print('found: %.3f' % best_fit)
                chosen_pop = chosen_pop.reshape(-1, d)

                left_b = lb + idx * gap
                right_b = lb + (idx + args.tau) * gap

                if not left_b <= chosen_pop[0, 0] <= right_b:
                    if Dl_samp[idx].shape[1] != d:
                        Dl_samp[idx] = chosen_pop
                        Dl_label[idx] = best_fit

                    else:
                        Dl_samp[idx] = np.vstack((Dl_samp[idx], chosen_pop))
                        Dl_label[idx] = np.vstack((Dl_label[idx], best_fit))

            record_profiles[Real_FE, IR] = current_best
            Real_FE += 1

            print('best fit: %.3f' % current_best)

        round_best_fit[IR] = current_best
    t_final = time.time()

    fit_mean, fit_std = np.mean(round_best_fit), np.std(round_best_fit)

    print(args.func, ', d=', d, ', alpha=%.1f, ' % args.alpha)
    print('Mean ± std: %.3f' % fit_mean, '$\pm$ %.3f, ' % fit_std)
    print('20 runs elapsed time: %.2f' % (t_final - t0), 's\n')

    file_path = './results/t1/'

    file_name = args.func + '_' + str(d) + '.csv'

    mean_profiles = np.mean(record_profiles, axis=1)
    mean_profiles = pd.DataFrame(mean_profiles[0:11 * d:2])
    mean_profiles.to_csv(file_path + file_name, index=False, header=False)

    std_profiles = np.std(record_profiles, axis=1)
    std_profiles = pd.DataFrame(std_profiles[0:11 * d:2])
    std_profiles.to_csv(file_path + 'std_' + file_name, index=False, header=False)

    record_profiles = pd.DataFrame(record_profiles)
    record_profiles.to_csv(file_path + 'record_' + file_name, index=False, header=False)
