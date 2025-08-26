import argparse
import os
import time
import gym
import numpy as np
import torch
import half_cheetah_v3
import hopper_v3
import humanoid_v3
import ant_v3
import walker2d_v3
import ant3d_v3
import MOTD7
from torch.utils.tensorboard import SummaryWriter
import copy


def train_online(pop, env, eval_env, args):
	evals = []
	start_time = time.time()
	allow_train = False
	state, ep_finished = env.reset(), False
	#state, ep_finished = env.reset()[0], False

	weights = np.random.rand(env.reward_num)
	weights /= weights.sum()
	state = np.append(state, weights)

	ep_total_reward, ep_timesteps, ep_num = np.zeros(env.reward_num), 0, 1

	for t in range(int(args.max_timesteps+1)):
		maybe_evaluate_and_print(pop[0], eval_env, evals, t, start_time, args)
		
		if allow_train:
			action = pop[0].select_action(np.array(state))
		else:
			action = env.action_space.sample()

		#next_state, vector_reward, terminated, truncated, _ = env.step(action)
		next_state, vector_reward, ep_finished, _ = env.step(action)
		next_state = np.append(next_state, weights)

		#reward = np.dot(vector_reward, weights)
		ep_total_reward += vector_reward
		ep_timesteps += 1

		#done = 1.0 if terminated else 0
		done = float(ep_finished) if ep_timesteps < env.max_episode_steps else 0

		for i in range(pop_size):
			pop[i].replay_buffer.add(state, action, next_state, vector_reward*weight_bias[i], weights, done)

		state = next_state

		if allow_train and not args.use_checkpoints:
			for i in range(pop_size):
				pop[i].train()

		if ep_finished:
			f.write(f"Total T: %s Episode Num: %s Episode T: %s Reward: %s\n" % (t+1, ep_num, ep_timesteps, ep_total_reward))
			f.flush()

			if t >= args.timesteps_before_training:
				allow_train = True

			#state, done = env.reset()[0], False
			state, done = env.reset(), False
			weights = np.random.rand(env.reward_num)
			weights /= weights.sum()
			state = np.append(state, weights)
			ep_total_reward, ep_timesteps = np.zeros(env.reward_num), 0
			ep_num += 1

		if t % 100000 == 0:
			save_filename = f"{model_path}/{run_name}__{t}"
			for i in range(pop_size):
				torch.save(pop[i].actor.state_dict(), save_filename + "_actor"+str(i))
				torch.save(pop[i].fixed_encoder.state_dict(), save_filename + "_fixed_encoder"+str(i))


weight_bias = [[1, 1], [1, 1]]
def init_hyperparameters():
	global weight_bias
	if args.env == 'MO_half_cheetah-v0':
		weight_bias = [[1, 1], [1, 6]]
	elif args.env == 'MO_ant-v0':
		weight_bias = [[1, 1], [1, 6]]
	elif args.env == 'MO_hopper-v0' or args.env == 'MO_walker-v0':
		weight_bias = [[1, 1], [1, 2]]
	elif args.env == 'MO_humanoid-v0':
		weight_bias = [[1, 0], [1, 2]]
	elif args.env == 'MO_ant3d-v0':
		weight_bias = [[1, 1, 1], [1, 2, 1]]
	f.write("weight_bias=%s\n" % (weight_bias))


def maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args):
	if t % args.eval_freq == 0:
		f.write("---------------------------------------\n")
		f.write(f"Evaluation at %s time steps\n" % t)
		f.write(f"Total time passed: %.2f min(s)\n" % (round((time.time()-start_time)/60.,2)))

		total_reward = np.zeros((args.eval_eps, env.reward_num))
		if env.reward_num == 1:
			w = [1]*10
		elif env.reward_num == 2:
			w = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [1.0, 0.0]]   # 2 objectives
		elif env.reward_num == 3:
			w = [[0.9, 0.05, 0.05], [0.7, 0.15, 0.15], [0.5, 0.25, 0.25], [0.33, 0.33, 0.33], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
		elif env.reward_num == 4:
			w = [[0.9, 0.033, 0.033, 0.033], [0.7, 0.1, 0.1, 0.1], [0.5, 0.166, 0.166, 0.166], [0.033, 0.9,  0.033, 0.033], [0.033, 0.033, 0.9, 0.033], [0.033, 0.033, 0.033, 0.9], [0.25, 0.25, 0.25, 0.25], [0.166, 0.5, 0.166, 0.166], [0.166, 0.166, 0.5, 0.166], [0.166, 0.166, 0.166, 0.5]]
		for ep in range(args.eval_eps):
			state, done = eval_env.reset(), False
			#state, terminated, truncated = eval_env.reset()[0], False, False
			#weights = np.random.rand(eval_env.reward_num)
			#weights /= weights.sum()
			weights = w[ep]
			state = np.append(state, weights)
			while not done:
				action = RL_agent.select_action(np.array(state), args.use_checkpoints, use_exploration=False)
				#state,  vector_reward, terminated, truncated, _ = eval_env.step(action)
				state, vector_reward, done, _ = eval_env.step(action)
				state = np.append(state, weights)
				#reward = np.dot(vector_reward, weights)
				total_reward[ep] += vector_reward
		#mean_total_reward = np.mean(total_reward, axis=0)
			f.write(f"weights: %s rewards %s\n" % (weights, total_reward[ep]))
		
		f.write("---------------------------------------\n")
		evals.append(total_reward)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# RL
	parser.add_argument("--env", default="MO_hopper-v0", type=str)
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument('--use_checkpoints', default=False)
	# Evaluation
	parser.add_argument("--timesteps_before_training", default=25e3, type=int)
	parser.add_argument("--eval_freq", default=5e3, type=int)
	parser.add_argument("--eval_eps", default=10, type=int)
	parser.add_argument("--max_timesteps", default=15e5, type=int)
	# File
	args = parser.parse_args()
	seed = args.seed
	env_tag = args.env
	time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
	run_name = f"{env_tag}__PB-MORL__{seed}__{time_str}"
	cwd = os.getcwd()
	path = f'{cwd}/log/{env_tag}/{run_name}'
	model_path = f'{cwd}/models/{env_tag}/{run_name}'
	os.makedirs(model_path, mode=0o777)
	os.makedirs(path, mode=0o777)
	log_name = run_name + '_log.txt'
	f = open(f'{path}/{log_name}', 'w')
	init_hyperparameters()

	env = gym.make(args.env)
	eval_env = gym.make(args.env)

	f.write("---------------------------------------\n")
	f.write(f"Algorithm: PB-MORL, Env: %s, Seed: %s\n" %(args.env, args.seed))
	f.write("---------------------------------------\n")

	env.seed(args.seed)
	#_, _ = env.reset(seed=args.seed)
	env.action_space.seed(args.seed)
	eval_env.seed(args.seed+100)
	#_, _ = eval_env.reset(seed=args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0] + env.reward_num
	action_dim = env.action_space.shape[0]
	reward_dim = env.reward_num
	max_action = float(env.action_space.high[0])

	pop = []
	pop_size = 2
	pop.append(MOTD7.Agent(state_dim, action_dim, reward_dim, max_action))
	for _ in range(1, pop_size):
		pop.append(copy.deepcopy(pop[0]))

	train_online(pop, env, eval_env, args)

