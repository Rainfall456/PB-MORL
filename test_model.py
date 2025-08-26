import argparse
import os
import time
#import gymnasium as gym
#import mo_gymnasium as mo_gym
import gym
import numpy as np
import half_cheetah_v3
import hopper_v3
import humanoid_v3
import ant_v3
import walker2d_v3
import torch
import copy

import MOTD7
from torch.utils.tensorboard import SummaryWriter
from pymoo.factory import get_performance_indicator
import matplotlib.pyplot as plt
from pymoo.config import Config
Config.show_compile_hint = False
#from pymoo.indicators.hv import HV
from pymoo.visualization.scatter import Scatter


def get_HV(eval_env, args):
    if args.env == 'MO_half_cheetah-v0':
        ref_point = np.array([0, -4000])
    elif args.env == 'MO_walker-v0':
        ref_point = np.array([0, -2500])
    elif args.env == 'MO_hopper-v0':
        ref_point = np.array([0, -1000])
    elif args.env == 'MO_humanoid-v0':
        ref_point = np.array([0, -1500])
    elif args.env == 'MO_ant-v0':
        ref_point = np.array([0, -3000])

    #ind = HV(ref_point=ref_point)
    hv = get_performance_indicator("hv", ref_point=ref_point)
    objective_values =[]
    for i in range(pop_size):
        total_reward = np.zeros((args.eval_eps, eval_env.reward_num))
        for ep in range(args.eval_eps):
            state, ep_finished = eval_env.reset(), False
            weights = np.array([ep*(1/args.eval_eps), 1-ep*(1/args.eval_eps)])
            state = np.append(state, weights)
            step = 0
            while not ep_finished:
                action = pop[i].select_action(np.array(state), args.use_checkpoints, use_exploration=False)
                #state,  vector_reward, terminated, truncated, _ = eval_env.step(action)
                state, vector_reward, ep_finished, _ = eval_env.step(action)
                state = np.append(state, weights)
                #reward = np.dot(vector_reward, weights)
                total_reward[ep] += vector_reward
                step += 1
            objective_values.append(total_reward[ep])

    print("HV:%s" % hv.do(-np.array(objective_values)+2*ref_point))
    SP = get_SP(objective_values)
    print("SP:%s" % SP)

    objective_values_ = copy.deepcopy(objective_values)
    for i in range(0, len(objective_values)):
        objective_values_[i][0] = objective_values[i][1]
        objective_values_[i][1] = objective_values[i][0]
    draw_scatter(objective_values_)


def get_SP(objective_values):
    o = get_no_dominated_solutions(objective_values)
    # print("point_num:%d " % len(o))
    sp = 0
    for i in range(env.reward_num):
        sp_i = 0
        sorted_o = sorted(o, key=lambda x: x[i], reverse=True)
        for j in range(len(o) - 1):
            sp_i += pow(sorted_o[j][i] - sorted_o[j + 1][i], 2)
        sp += sp_i / (len(o) - 1)
    return sp


def get_no_dominated_solutions(objective_values):
    nds = []
    sorted_o = sorted(objective_values, key=lambda x: x[0], reverse=True)
    max2 = -99999
    for i in range(0, len(sorted_o)):
        if sorted_o[i][1] > max2:
            nds.append([sorted_o[i][0], sorted_o[i][1]])
            max2 = sorted_o[i][1]
    return np.array(nds)


def draw_scatter(objective_values):
    o1 = get_no_dominated_solutions(objective_values[0:200])
    o2 = get_no_dominated_solutions(objective_values[0:100])
    plt.scatter(o1[:, 0], o1[:, 1], color='red', marker='o', label='PB-MORL')
    plt.scatter(o2[:, 0], o2[:, 1], color='cornflowerblue', marker='v', label='PB-MORL (single agent)', alpha=1)
    plt.title('objective values')
    plt.xlabel('Control Cost')
    plt.ylabel('Forward Reward')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RL
    parser.add_argument("--env", default="MO_half_cheetah-v0", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--use_checkpoints', default=True)
    # Evaluation
    parser.add_argument("--timesteps_before_training", default=25e3, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--eval_eps", default=100, type=int)
    # File
    args = parser.parse_args()
    seed = args.seed
    env_tag = args.env
    cwd = os.getcwd()

    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    env.seed(args.seed)
    #_, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    eval_env.seed(args.seed+100)
    #_, _ = eval_env.reset(seed=args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0] + env.reward_num
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    RL_agent = MOTD7.Agent(state_dim, action_dim,  env.reward_num, max_action)
    model_name = "./models/"+env_tag
    pop = []
    pop_size = 2
    for i in range(pop_size):
        pop.append(MOTD7.Agent(state_dim, action_dim, env.reward_num, max_action))
        pop[i].checkpoint_actor.load_state_dict(torch.load(model_name + "_actor"+str(i), map_location=torch.device('cpu')))
        pop[i].checkpoint_encoder.load_state_dict(torch.load(model_name + "_fixed_encoder"+str(i), map_location=torch.device('cpu')))

    get_HV(eval_env, args)


