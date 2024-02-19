import os
import time
import jericho
import logger
import argparse
import logging
import json
import subprocess
from jericho.util import clean
from random import choice
from collections import defaultdict
import numpy as np
import pickle
import shutil

from env_calm import EnvCALM
from memory import *
from lm import *

from agent_sac_meta import *

from conditioning_model import EthicsModel
import torch.nn as nn
import copy

logging.getLogger().setLevel(logging.CRITICAL)


def configure_logger(log_dir, add_tb=1, add_wb=1, args=None):
	logger.configure(log_dir, format_strs=['log'])
	global tb
	log_types = [logger.make_output_format('json', log_dir),
				 logger.make_output_format('stdout', log_dir)]
	if add_tb: log_types += [logger.make_output_format('tensorboard', log_dir)]
	if add_wb: log_types += [logger.make_output_format('wandb', log_dir, args=args)]
	tb = logger.Logger(log_dir, log_types)

	global log
	log = logger.log


def train(agent, lm,envs, max_steps, update_freq, checkpoint_freq, log_freq, args, cclm):
	start = time.time()
	cumulative_morality_vector = np.zeros(4, dtype=np.int_)
	obs, rewards, dones, infos, transitions,ethics_rewards = [], [], [], [], [],[]
	env_steps, max_score, d_in, d_out = 0, 0, defaultdict(list), defaultdict(list)
	if args.lm_top_k:  # small pre-trained classifier to filter invalid actions that CALM generates
		import fasttext
		detector = fasttext.load_model('/..your_path/valid_model.bin')
		reject = lambda ob: detector.predict(clean(ob))[0][0] == '__label__invalid'
	if args.lm_dict:

		print('LOADING LM GENERATION DICT')
		g_dict = json.load(open(args.lm_dict, 'r'))
		lm.generate_dict = {int(k):v for k,v in g_dict.items()}
	for env in envs:
		ob, info = env.reset()
		obs, rewards, dones, infos, transitions,ethics_rewards = \
			obs + [ob], rewards + [0], dones + [False], infos + [info], transitions + [[]],ethics_rewards+[[0]]

	valid_ids = [info['valid'] for info in infos]


	if args.using_LLM == True:
		states = LLM_build_state(agent._tokenize, lm,obs, infos, envs)
		valid_ids_tokenized = [[agent._tokenize(a) for a in info['valid']] for info in infos]
	else:
		states = build_state(lm, obs, infos, envs)
		valid_ids_tokenized = [[lm.act2ids(a) for a in info['valid']] for info in infos]

	for step in range(1, max_steps + 1):

		action_ids,action_idxs,action_values = agent.choose_action(states, valid_ids_tokenized)
		action_strs = [info['valid'][idx] for info, idx in zip(infos, action_idxs)]


		log('State  {}: {}'.format(step, lm.tokenizer.decode(states[0].state)))
		log('Actions{}: {}'.format(step, infos[0]['valid']))
		log('Qvalues{}: {}'.format(step, action_values[0].tolist()))

		# step with rejection
		next_obs, next_rewards, next_dones, next_infos,next_ethics_rewards = [], [], [], [],[]
		for i, (env, action) in enumerate(zip(envs, action_strs)):
			if dones[i]:
				env_steps += infos[i]['moves']
				env.env._write_to_log('<!DONE!>', 'step_log.txt')
				ob, info = env.reset()


				if (info['label_vectors'] != []) and (sum(info['label_vectors'])[0] > 0):
					eth_rew= 1
				else:eth_rew=0

				action_strs[i], action_ids[i], transitions[i] = 'reset', [], []
				next_obs, next_rewards, next_dones, next_infos,next_ethics_rewards = next_obs + [ob], next_rewards + [0], next_dones + [
					False], next_infos + [info],next_ethics_rewards+ [eth_rew]
				continue
			prev_inv, prev_look = infos[i]['inv'], infos[i]['look']

			ob, reward, done, info = env.step(action, write_to_log=True)
			env.env._write_to_log('STEP', 'step_log.txt')  # first action for this step
			markers, label_vectors = info['markers'], info['label_vectors']



			# ========================== #
			if i == 0:
				log('>> Action{}: {}'.format(step, action))
				log("Reward{}: {}, Score {}, Done {}, Markers: {}, Label_vectors: {}".format(step, reward, info['score'], done, markers, label_vectors))
			for vec in label_vectors:
				cumulative_morality_vector += vec

			if args.lm_top_k:  # deal with rejection
				key = hash(tuple(states[i][0] + states[i][1]+states[i][2]))
				l_in, l_out = d_in[key], d_out[key]
				actions = infos[i]['valid']
				actions_sorted = False
				rej = reject(ob) and prev_inv == info['inv'] and prev_look == info['look']

				# while action is invalid, pull another action from CALM generated candidates
				while not done and rej and len(actions) > 1:
					####### sorting actions based on q_values #######
					if not actions_sorted:
						actions = [x for _, x in sorted(zip(action_values[i].tolist(), actions))]
						actions.reverse() # descending
						actions_sorted = True
					#################################################

					if action not in l_out: l_out.append(action)
					actions.remove(action)
					# action = choice(actions)
					action = actions[0]
					ob, reward, done, info = env.step(action, write_to_log=True)
					env.env._write_to_log('', 'step_log.txt')  # alternate actions for this step
					markers, label_vectors = info['markers'], info['label_vectors']

					if i == 0:
						log('>> Action{}: {}'.format(step, action))
						log("Reward{}: {}, Score {}, Done {}, Markers: {}, Label_vectors: {}".format(step, reward, info['score'], done, markers, label_vectors))
					for vec in label_vectors:
						cumulative_morality_vector += vec

					rej = reject(ob) and prev_inv == info['inv'] and prev_look == info['look']
				action_strs[i] = action

				if not rej and action not in l_in: l_in.append(action)
				if reward < 0 and action not in l_out: l_out.append(action)  # screen negative-reward actions

			if (info['label_vectors'] != []) and (sum(info['label_vectors'])[0] > 0):
				eth_rew = 1
			else:
				eth_rew = 0
			next_obs, next_rewards, next_dones, next_infos,next_ethics_rewards = \
				next_obs + [ob], next_rewards + [reward], next_dones + [done], next_infos + [info],next_ethics_rewards+ [eth_rew]#next_ethic_rewards


			if info['score'] > max_score:  # new high score experienced
				max_score = info['score']
				#agent.memory.clear_alpha()
			if done: tb.logkv_mean('EpisodeScore', info['score'])
		rewards, dones, infos,ethics_rewards = next_rewards, next_dones, next_infos,next_ethics_rewards

		# ================= GENERATE VALID ACTIONS ================= #

		if args.using_LLM == True:
			next_states = LLM_build_state(agent._tokenize, lm,obs, infos, envs,prev_obs=obs, prev_acts=action_strs)
			#valid_ids_tokenized = [[agent._tokenize(a) for a in info['valid']] for info in infos]
		else:
			next_states = build_state(lm, obs, infos, envs,prev_obs=obs, prev_acts=action_strs)
		#next_states = build_state(agent._tokenize, next_obs, infos, envs, prev_obs=obs, prev_acts=action_strs)
		if args.lm_top_k:
			batched_states = [s.raw_state for s in next_states]
			batched_actions = lm.batched_generate(batched_states, k=args.lm_top_k, batch_size=args.lm_batch_size)
			# for env, info, state, done in zip(envs, infos, next_states, dones):
			for env, info, state, actions, done in zip(envs, infos, next_states, batched_actions, dones):

				if not done:
					#key = hash(tuple(state[0] + state[1] + state[2]))
					key = hash(tuple(state[0]+ state[1] + state[2]))#+ state[1] + state[2]
					if args.lm_type == 'ngram':
						objs = env.get_objects()
						actions = lm.generate(objs, k=args.lm_top_k).copy()
					else:
						# actions = lm.generate(state.state, k=args.lm_top_k)
						l_in, l_out = d_in[key], d_out[key]
						actions += [action for action in l_in if action not in actions]  # add extra valid
						actions = [action for action in actions if action and action not in l_out]  # remove invalid
					if not actions: actions = ['wait', 'yes', 'no']
					info['valid'] = actions

		next_valids = [info['valid'] for info in infos]

		if args.using_LLM is True:
			next_valids_tokenized = [[agent._tokenize(a) for a in info['valid']] for info in infos]
		else:
			next_valids_tokenized = [[lm.act2ids(a) for a in info['valid']] for info in infos]


		for state, act, act_id, rew, next_state, valids, poss_acts, done, transition,ethics_reward,action_idx in zip(states, valid_ids_tokenized, action_ids, rewards, next_states,
																		 next_valids_tokenized, action_strs, dones, transitions,ethics_rewards,action_idxs):
			if act_id:  # not [] (i.e. reset)
				if len(act_id)==0:
					print('####ERROR',valid_ids_tokenized,action_ids)
				transition.append(Transition(state, next_state, act_id, act, valids, poss_acts,rew,  done,ethics_reward,action_idx))
				agent.observe(transition[-1])
		obs, states, valid_ids, valid_ids_tokenized = next_obs, next_states, next_valids, next_valids_tokenized


		if step % log_freq == 0:
			tb.logkv('Step', env_steps)
			tb.logkv('ETHICS', cumulative_morality_vector)
			tb.logkv("FPS", round((step * args.num_envs) / (time.time() - start), 2))
			tb.logkv("Max score seen", max_score)
			tb.logkv("#dict", len(lm.generate_dict))
			tb.logkv("Last50EpisodeScores", sum(env.get_end_scores(last=50) for env in envs) / len(envs))
			tb.dumpkvs()

		if step % update_freq == 0:
			out = agent.update(args,step)


		if step % checkpoint_freq == 0:
			json.dump(d_in, open('%s/d_in.json' % args.output_dir, 'w'), indent=4)
			json.dump(d_out, open('%s/d_out.json' % args.output_dir, 'w'), indent=4)
			json.dump(lm.generate_dict, open('%s/lm.json' % args.output_dir, 'w'), indent=4)

		if step % args.save_freq == 0:
			# save agent model to output_dir
			agent.save_model(args.output_dir)
			if cclm:
				cclm.save_cache(args.output_dir)


		if (step == 5000) and (max_score == env.env.get_starting_score()):
			print('\n\nStopping training early due to lack of progress.\n\n')
			break  # cut off training prematurely if there is no progress after 5000 steps

	# ============================ FINAL SAVES ============================ #
	tb.logkv('Step', env_steps)
	tb.logkv('ETHICS', cumulative_morality_vector)
	tb.logkv("FPS", round((step * args.num_envs) / (time.time() - start), 2))
	tb.logkv("Max score seen", max_score)
	tb.logkv("#dict", len(lm.generate_dict))
	#tb.logkv("#dict_d", sum(len(env.d) for env in envs) / len(envs))
	tb.logkv("lam_value", lam)
	tb.logkv("Last50EpisodeScores", sum(env.get_end_scores(last=50) for env in envs) / len(envs))
	tb.dumpkvs()

	json.dump(d_in, open('%s/d_in.json' % args.output_dir, 'w'), indent=4)
	json.dump(d_out, open('%s/d_out.json' % args.output_dir, 'w'), indent=4)
	json.dump(lm.generate_dict, open('%s/lm.json' % args.output_dir, 'w'), indent=4)

	agent.save_model(args.output_dir)
	if cclm:
		cclm.save_cache(args.output_dir)
	# ===================================================================== #

	print('TOTAL TIME ELAPSED:', time.time() - start)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--output_dir', default='/path')
	parser.add_argument('--load_dir', default=None, type=str)
	parser.add_argument('--env_step_limit', default=100, type=int)
	parser.add_argument('--seed', default=1026, type=int)
	parser.add_argument('--num_envs', default=8, type=int)
	parser.add_argument('--max_steps', default=15000, type=int)
	parser.add_argument('--update_freq', default=1, type=int)
	parser.add_argument('--save_freq', default=15000, type=int)
	parser.add_argument('--checkpoint_freq', default=1000, type=int)
	parser.add_argument('--eval_freq', default=500, type=int)
	parser.add_argument('--log_freq', default=100, type=int)
	parser.add_argument('--memory_size', default=10000, type=int)
	parser.add_argument('--priority_fraction', default=0.5, type=float)
	parser.add_argument('--batch_size', default=64, type=int) #64
	parser.add_argument('--gamma', default=.9, type=float)
	parser.add_argument('--learning_rate', default=0.0001, type=float)
	parser.add_argument('--clip', default=5, type=float)
	parser.add_argument('--embedding_dim', default=128, type=int)
	parser.add_argument('--hidden_dim', default=128, type=int)

	# logger
	parser.add_argument('--tensorboard', default=0, type=int)
	parser.add_argument('--wandb', default=0, type=int)
	parser.add_argument('--wandb_project', default='textgame', type=str)

	# language model
	parser.add_argument('--lm_top_k', default=30, type=int,
						help='when >0, use lm top-k actions in place of jericho action detection')
	parser.add_argument('--lm_type', default='gpt', help='gpt | ngram')
	parser.add_argument('--lm_path', default='gpt2')
	parser.add_argument('--lm_dict', default='')
	parser.add_argument('--fp16', action='store_true')
	parser.add_argument('--lm_batch_size', default=None, type=int, help='batch size for CALM action generator; defaults to all envs at once')
	parser.add_argument('--model_parallel', action='store_true')

	# conditioning
	parser.add_argument('--cclm_path', default=None, type=str)
	parser.add_argument('--cclm_model', default='roberta-large', type=str)
	parser.add_argument('--conditioning_model_load_dir', default=None, type=str)
	parser.add_argument('--reward_shaping', action='store_true')
	parser.add_argument('--policy_shaping', action='store_true')
	parser.add_argument('--cclm_oracle', action='store_true')
	parser.add_argument('--condition_weight', default=0, type=float)
	parser.add_argument('--cond_threshold', default=0.5, type=float)
	parser.add_argument('--weight_conditioning_by_max_score', action='store_true')
	parser.add_argument('--adaptive_cond_weight', action='store_true')
	parser.add_argument('--adaptive_cond_weight2', action='store_true')

	# annotation
	parser.add_argument('--game_folder_path', type=str)

	# fast-forwarding
	parser.add_argument('--starting_percentage', type=int, default=0)

	# exploration
	parser.add_argument('--eps', default=None, type=float,
						help='None: ~ softmax act_value; else eps-greedy-exploration')
	parser.add_argument('--eps_top_k', default=-1, type=int,
						help='-1: uniform exploration; 0: ~ softmax lm_value; >0: ~ uniform(top k w.r.t. lm_value)')
	parser.add_argument('--alpha', default=0, type=float,
						help='act_value = alpha * bert_value + (1-alpha) * q_value; only used when eps is None now')
	parser.add_argument('--argmax', action='store_true')
	parser.add_argument('--using_LLM', default=False)

	parser.add_argument('--lambda_value', type=int, default=5)

	return parser.parse_args()


def main():
	start_time = time.time()

	assert jericho.__version__.startswith('3'), "This code is designed to be run with Jericho version >= 3.0.0."
	args = parse_args()

	if os.path.exists(os.path.join(args.output_dir, 'experiment_info.pkl')):
		print('found experiment_info.pkl, so this experiment has already been run; exiting')
		exit()
	else:
		if os.path.exists(args.output_dir): shutil.rmtree(args.output_dir)
		os.makedirs(args.output_dir)

	print('==============================================')
	print('============== RUNNING TRAINING FOR {} ==============\n'.format(args.game_folder_path))

	if args.reward_shaping:
		assert (not args.policy_shaping) and (args.cclm_path is not None)
		print(f'---> REWARD SHAPING with {args.cclm_path}')
	elif args.policy_shaping:
		assert not args.reward_shaping and (args.cclm_path is not None)
		print(f'---> POLICY SHAPING with {args.cclm_path}')
	else:
		assert (args.condition_weight == 0) and (args.cclm_path is None)
		print('---> UNCONDITIONED')

	if args.conditioning_model_load_dir:
		print(f'LOADING CONDITIONING MODEL FROM {args.conditioning_model_load_dir}')

	if (args.lm_dict != '') and (os.path.exists(args.lm_dict) == False):
		print('could not find lm_dict; defaulting to empty cache')
		args.lm_dict = ''

	lm = GPT2LM(args.lm_path, args.fp16, model_parallel=args.model_parallel)
	args.vocab_size = len(lm.tokenizer)

	print(args)
	print()

	if args.cclm_path:
		cclm = EthicsModel(args.cclm_model, args.cclm_path,
						   args.conditioning_model_load_dir or args.load_dir, oracle=args.cclm_oracle)
	else:
		cclm = None

	configure_logger(args.output_dir, args.tensorboard, args.wandb, args)
	agent = Agent(args)

	# =========================== CREATE ENVS =========================== #
	envs = [EnvCALM(game_folder_path=args.game_folder_path, step_limit=args.env_step_limit, seed=args.seed,
						  starting_percentage=args.starting_percentage,
						  log_dir=os.path.join(args.output_dir, 'log_files', 'env{}'.format(i)))
			for i in range(args.num_envs)]


	if args.weight_conditioning_by_max_score:
		args.condition_weight = args.condition_weight * envs[0].env.get_max_score()

	# =========================== START TRAINING =========================== #
	print('MAX SCORE FOR GAME: {}'.format(envs[0].env.get_max_score()))
	print('STARTING SCORE: {}'.format(envs[0].env.get_starting_score()))

	print('\n==============================================\n')

	train(agent, lm, envs, args.max_steps, args.update_freq,args.checkpoint_freq, args.log_freq, args, cclm)

	with open(os.path.join(args.output_dir, 'experiment_info.pkl'), 'wb') as f:
		pickle.dump({'status': 'Done', 'time': time.time() - start_time,
					 'peak_memory': torch.cuda.max_memory_allocated()}, f)


if __name__ == "__main__":
	main()
