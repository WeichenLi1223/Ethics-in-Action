import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update#, hard_update
from model_lr import *
import sentencepiece as spm
# from collections import namedtupl
from memory import *
from collections import defaultdict
import jericho
from jericho.util import clean

def build_state(lm, obs, infos, envs, prev_obs=None, prev_acts=None):
	"""
	Return a state representation built from various info sources.
	obs, prev_obs, prev_acts: list of strs.
	"""
	if prev_obs is None:
		return [State(lm.sent2ids(ob), lm.sent2ids(info['look']), lm.sent2ids(info['inv']),
					  lm.sent2ids(ob), ob, hash(str(env.env.get_state())))
				for ob, info, env in zip(obs, infos, envs)]
	else:
		states = []
		for prev_ob, ob, info, act, env in zip(prev_obs, obs, infos, prev_acts, envs):
			sent = "[CLS] %s [SEP] %s [SEP] %s [SEP]" % (prev_ob, act, ob + info['inv'] + info['look'])
			# sent = "[CLS] %s [SEP]" % (ob + info['inv'] + info['look'])
			states.append(State(lm.sent2ids(ob), lm.act2ids(info['look']),
						  lm.act2ids(info['inv']), lm.sent2ids(sent), sent, hash(str(env.env.get_state()))))
		return states

class Agent(object):
	def __init__(self, args):
		self.sp = spm.SentencePieceProcessor()
		#self.sp.Load(args.spm_path)

		#self.num_inputs = env.observation_space.shape[0]
		#self.action_space = env.action_space

		#self.policy_type = args.policy
		#self.target_update_interval = args.target_update_interval
		#self.automatic_entropy_tuning = argsautomatic_entropy_tuning

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.critic =  Q_Network(args.vocab_size,args.embedding_dim, args.hidden_dim).to(self.device)
		self.critic_optim = Adam(self.critic.parameters(), lr=0.0003)

		self.critic_target = Q_Network(args.vocab_size,args.embedding_dim, args.hidden_dim).to(self.device)

		soft_update(self.critic_target, self.critic,0.1)

		self.policy =DRRN(args.vocab_size,args.embedding_dim, args.hidden_dim).to(self.device)
		self.policy_optim = Adam(self.policy.parameters(), lr=0.0003)

		self.critic_target_update_frequency = 2

		self.log_alpha = torch.tensor(np.log(0.1)).to(self.device)
		self.critic_tau = 0.1

		self.moral_network = MoralNetwork(args.vocab_size,args.embedding_dim, args.hidden_dim).to(self.device)
		self.moral_optimizer = torch.optim.Adam(self.moral_network.parameters(), lr=0.0003)

		self.batch_size = args.batch_size
		self.discount = 0.9
		self.clip = args.clip

		self.lam = args.lambda_value #5.0
		self.lam_lr = args.lambda_lr #0.0001#1e-5
		self.delta = args.delta_value #0.1

		self.memory = ReplayMemory(args.memory_size) #PrioritizedReplayMemory(args.memory_size, args.priority_fraction)

	def save_model(self, save_dir):
		model_to_save = self.policy.module if hasattr(self.policy, 'module') else self.policy
		output_model_file = os.path.join(save_dir, 'model.pt')
		output_opt_file = os.path.join(save_dir, 'optimizer.pt')
		output_replay_file = os.path.join(save_dir, 'replay.pkl')
		torch.save(model_to_save.state_dict(), output_model_file)
		torch.save(self.moral_optimizer.state_dict(), output_opt_file)
		torch.save(self.policy_optim.state_dict(), output_opt_file)
		torch.save(self.critic_optim.state_dict(), output_opt_file)
		torch.save(self.memory, output_replay_file)

	def load_model(self, load_dir):
		model_file = os.path.join(load_dir, 'model.pt')
		opt_file = os.path.join(load_dir, 'optimizer.pt')
		replay_file = os.path.join(load_dir, 'replay.pkl')
		self.network.load_state_dict(torch.load(model_file))
		self.optimizer.load_state_dict(torch.load(opt_file))
		if os.path.isfile(replay_file):
			self.memory = torch.load(replay_file)

	@property
	def alpha(self):
		return self.log_alpha.exp()

	def observe(self, transition, is_prior=False):
		self.memory.push(transition)

	def choose_action(self,states, poss_acts, sample=True):
		""" Returns a string action from poss_acts. """

		with torch.no_grad():
			#print('###51 states',states)
			#print('###52 poss_acts',poss_acts)
			idxs,values,_ = self.policy.act(states, poss_acts)
			act_ids = [poss_acts[batch][idx] for batch, idx in enumerate(idxs)]
		return  act_ids,idxs,values


	def update_parameters(self, batch):
		index = [valids.index(x) for valids, x in zip(batch.valids,batch.act)]
		index = torch.LongTensor(index).to(device)
		nested_acts = tuple([[a] for a in batch.act])

		act_idxs, act_probs, log_prob = self.policy.act(batch.next_state,batch.next_valids)
		next_action = tuple([[next_valids[idx]] for next_valids,idx in  zip(batch.next_valids,act_idxs)])
		target_moral = self.moral_network(batch.next_state,next_action)
		#print('####target_moral',target_moral)
		target_moral = torch.cat(target_moral)
		#print('####target_moral',target_moral)
		target_moral = torch.tensor(batch.cost, dtype=torch.float, device=device) + ((1-torch.tensor(batch.done, dtype=torch.float, device=device)) * self.discount *target_moral).detach()
		current_moral = self.moral_network(batch.state,nested_acts)

		current_moral = torch.cat(current_moral)


		# Compute critic loss
		M_critic_loss = F.mse_loss(current_moral, target_moral)

		# Optimize the moral
		self.moral_optimizer.zero_grad()
		M_critic_loss.backward()
		self.moral_optimizer.step()

		with torch.no_grad():
			act_idxs, act_probs, log_prob = self.policy.act(batch.next_state,batch.next_valids)
			next_action = tuple([[next_valids[idx]] for next_valids,idx in  zip(batch.next_valids,act_idxs)])
			target_Q1,target_Q2 = self.critic_target(batch.next_state, batch.next_valids)
			target_V = [(act*(torch.min(t1,t2) - self.alpha.detach() * log)).sum(dim = 0, keepdim =True) for act,t1,t2,log in zip(act_probs,target_Q1,target_Q2,log_prob)]
			target_V = torch.cat((target_V),0)

			target_Q = torch.tensor(batch.rew, dtype=torch.float, device=device) + ((1-torch.tensor(batch.done, dtype=torch.float, device=device)) * self.discount *  target_V.clone())

		current_Q1,current_Q2  = self.critic(batch.state,nested_acts)

		#current_Q1 = [current_q1.gather(0,idx) for current_q1,idx in zip(current_Q1,index)]
		current_Q1= torch.cat(current_Q1)
		current_Q2= torch.cat(current_Q2)

		qf1_loss =torch.mean((current_Q1 - target_Q).pow(2) ) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
		qf2_loss = torch.mean((current_Q2 - target_Q).pow(2) ) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
		qf_loss = qf1_loss + qf2_loss

		self.critic_optim.zero_grad()
		qf_loss.backward()
		self.critic_optim.step()

		###Update policy/Actor###
		act_idxs, act_probs, log_prob = self.policy.act(batch.state, batch.valids)
		action = tuple([[valids[idx]] for valids,idx in  zip(batch.valids,act_idxs)])
		with torch.no_grad():
			actor_Q1, actor_Q2 = self.critic(batch.state,batch.valids)#,batch.valids
			actor_Q = [torch.min(q1, q2) for q1, q2 in zip(actor_Q1,actor_Q2)]
		#actor_Q = torch.cat(actor_Q)
		moral_score_ = self.moral_network(batch.state,batch.valids) #batch.valids
		#moral_score = torch.cat(moral_score)

		moral_score = [torch.sum(m_score*act, dim=0, keepdim=True)for m_score,act in zip(moral_score_,act_probs)]
		moral_score = torch.cat(moral_score)

		# Expectations of entropies.
		entropies = [torch.sum(
			act * log, dim=0, keepdim=True) for act,log in zip(act_probs,log_prob)] #-

		q = [torch.sum(torch.min(q1, q2) * act, dim=0, keepdim=True)for q1,q2,act in zip(actor_Q1,actor_Q2,act_probs)]
		entropies = torch.cat(entropies)
		q = torch.cat(q)
		#print('####q',q)



		policy_loss = ((self.alpha*entropies-q) + self.lam *(moral_score - self.delta)).mean()


		self.policy_optim.zero_grad()
		policy_loss.backward()
		nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip)
		self.policy_optim.step()

		self.lam = max(0,self.lam + self.lam_lr * (torch.cat(moral_score_).mean().item() -self.delta))
		#breakpoint()

		return self.lam, self.lam_lr, policy_loss.item()#, alpha_loss.item(), alpha_tlogs.item()

	def update(self, args,step):
		if len(self.memory) < self.batch_size:
			return self.lam, self.lam_lr, 0

		self.transitions = transitions = self.memory.sample(
			self.batch_size)#,self.device
	
		batch = Transition(*zip(*transitions))

		q1_loss,q2_loss,policy_loss = self.update_parameters(batch)

		if step % self.critic_target_update_frequency == 0:
			soft_update(self.critic, self.critic_target,
									 self.critic_tau)

		return q1_loss,q2_loss,policy_loss
	# Save model parameters
	def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
		if not os.path.exists('checkpoints/'):
			os.makedirs('checkpoints/')
		if ckpt_path is None:
			ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
		print('Saving models to {}'.format(ckpt_path))
		torch.save({'policy_state_dict': self.policy.state_dict(),
					'critic_state_dict': self.critic.state_dict(),
					'critic_target_state_dict': self.critic_target.state_dict(),
					'critic_optimizer_state_dict': self.critic_optim.state_dict(),
					'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

	# Load model parameters
	def load_checkpoint(self, ckpt_path, evaluate=False):
		print('Loading models from {}'.format(ckpt_path))
		if ckpt_path is not None:
			checkpoint = torch.load(ckpt_path)
			self.policy.load_state_dict(checkpoint['policy_state_dict'])
			self.critic.load_state_dict(checkpoint['critic_state_dict'])
			self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
			self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
			self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

			if evaluate:
				self.policy.eval()
				self.critic.eval()
				self.critic_target.eval()
			else:
				self.policy.train()
				self.critic.train()
				self.critic_target.train()
