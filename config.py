#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse


def args_parser():
    """
    one thing must be mentioned: if mini-batch sgd is set to the optimizer, if you want a better performance
    for few rounds, local epoch or learning rates must be large, due to sgd only uses few samples while gd
    goes through all samples in the dataset
    """
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--runs',       type=int,   default=20,     help="how many runs")
    parser.add_argument('--max_gens',   type=int,   default=100,    help="SOP: 100, MOP: 50")
    parser.add_argument('--num_users',  type=int,   default=100,    help="number of users: K")
    parser.add_argument('--frac',       type=float, default=0.1,    help="the fraction: λ")
    parser.add_argument('--E',          type=int,   default=5,      help="local epoch")
    parser.add_argument('--opt',        type=str,   default='sgd',  help="optimizer, sgd/m-sgd/max-gd")
    parser.add_argument('--lr',         type=float, default=0.06,   help="learning rate")

    # optimization arguments
    parser.add_argument('--boot_prob',  type=float, default=1,      help='bootstrap probability')
    parser.add_argument('--init_type',  type=str,   default='same', help='clients holds the same data or not')
    parser.add_argument('--alpha',      type=float, default=0.0,    help='noisy')

    # Test function arguments
    parser.add_argument('--func',       type=str,   default='F13',  help='test function name')
    parser.add_argument('--d',          type=int,   default=10,     help='dimension of decision space')
    parser.add_argument('--M',          type=int,   default=1,      help="how many objectives")

    # other arguments
    parser.add_argument('--seed',       type=int,   default=1234,   help='random seed (default: 1)')
    args = parser.parse_args()
    return args
