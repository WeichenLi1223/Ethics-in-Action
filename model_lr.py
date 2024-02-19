import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
import itertools
from more_itertools import locate
from memory import State
from torch.distributions import Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight, gain=1)
		torch.nn.init.constant_(m.bias, 0)
	#enddef

class Q_Network(torch.nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim):
		super(Q_Network, self).__init__()

		#Q1 architecture
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.obs_encoder = nn.GRU(embedding_dim, hidden_dim)
		self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
		self.inv_encoder = nn.GRU(embedding_dim, hidden_dim)
		self.act_encoder = nn.GRU(embedding_dim, hidden_dim)
		self.hidden = nn.Linear(4 * hidden_dim, hidden_dim)
		self.act_scorer = nn.Linear(hidden_dim, 1)

		#Q2 architecture
		self.embedding2 = nn.Embedding(vocab_size, embedding_dim)
		self.obs_encoder2 = nn.GRU(embedding_dim, hidden_dim)
		self.look_encoder2 = nn.GRU(embedding_dim, hidden_dim)
		self.inv_encoder2 = nn.GRU(embedding_dim, hidden_dim)
		self.act_encoder2 = nn.GRU(embedding_dim, hidden_dim)
		self.hidden2 = nn.Linear(4 * hidden_dim, hidden_dim)
		self.act_scorer2 = nn.Linear(hidden_dim, 1)

	def packed_rnn(self, x, rnn):
		""" Runs the provided rnn on the input x. Takes care of packing/unpacking.
			x: list of unpadded input sequences
			Returns a tensor of size: len(x) x hidden_dim
		"""
		lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=device)
		# Sort this batch in descending order by seq length
		lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
		_, idx_unsort = torch.sort(idx_sort, dim=0)
		idx_sort = torch.autograd.Variable(idx_sort)
		idx_unsort = torch.autograd.Variable(idx_unsort)
		padded_x = pad_sequences(x)
		x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
		x_tt = x_tt.index_select(0, idx_sort)
		# Run the embedding layer
		embed = self.embedding(x_tt).permute(1, 0, 2)  # Time x Batch x EncDim
		# Pack padded batch of sequences for RNN module
		packed = nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu())
		# Run the RNN
		out, _ = rnn(packed)
		# Unpack
		out, _ = nn.utils.rnn.pad_packed_sequence(out)
		# Get the last step of each sequence
		idx = (lengths - 1).view(-1, 1).expand(len(lengths), out.size(2)).unsqueeze(0)
		out = out.gather(0, idx).squeeze(0)
		# Unsort
		out = out.index_select(0, idx_unsort)
		return out

	def packed_rnn2(self, x, rnn):
		""" Runs the provided rnn on the input x. Takes care of packing/unpacking.
			x: list of unpadded input sequences
			Returns a tensor of size: len(x) x hidden_dim
		"""
		lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=device)
		# Sort this batch in descending order by seq length
		lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
		_, idx_unsort = torch.sort(idx_sort, dim=0)
		idx_sort = torch.autograd.Variable(idx_sort)
		idx_unsort = torch.autograd.Variable(idx_unsort)
		padded_x = pad_sequences(x)
		x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
		x_tt = x_tt.index_select(0, idx_sort)
		# Run the embedding layer
		embed = self.embedding2(x_tt).permute(1, 0, 2)  # Time x Batch x EncDim
		# Pack padded batch of sequences for RNN module
		packed = nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu())
		# Run the RNN
		out, _ = rnn(packed)
		# Unpack
		out, _ = nn.utils.rnn.pad_packed_sequence(out)
		# Get the last step of each sequence
		idx = (lengths - 1).view(-1, 1).expand(len(lengths), out.size(2)).unsqueeze(0)
		out = out.gather(0, idx).squeeze(0)
		# Unsort
		out = out.index_select(0, idx_unsort)
		return out

	def forward(self, state_batch, act_batch, detach=False, cond_weight=0,
				cclm=None, cond_threshold=0, args=None, testing_flag=False):
		"""
			Batched forward pass.
			obs_id_batch: iterable of unpadded sequence ids
			act_batch: iterable of lists of unpadded admissible command ids
			Returns a tuple of tensors containing q-values for each item in the batch
		"""
		# Zip the state_batch into an easy access format
		state = State(*zip(*state_batch))
		# This is number of admissible commands in each element of the batch
		act_sizes = [len(a) for a in act_batch]
		# Combine next actions into one long list
		act_batch = list(itertools.chain.from_iterable(act_batch))
		act_out = self.packed_rnn(act_batch, self.act_encoder)
		# Encode the various aspects of the state
		obs_out = self.packed_rnn(state.obs, self.obs_encoder)
		look_out = self.packed_rnn(state.description, self.look_encoder)
		inv_out = self.packed_rnn(state.inventory, self.inv_encoder)
		state_out = torch.cat((obs_out, look_out, inv_out), dim=1)
		# Expand the state to match the batches of actions
		state_out = torch.cat([state_out[i].repeat(j, 1) for i, j in enumerate(act_sizes)], dim=0)
		z = torch.cat((state_out, act_out), dim=1)  # Concat along hidden_dim
		z = F.relu(self.hidden(z))
		q_values1 = self.act_scorer(z).squeeze(-1).split(act_sizes)

		#Q2 architecture
		act_out2 = self.packed_rnn2(act_batch, self.act_encoder2)
		# Encode the various aspects of the state
		obs_out2 = self.packed_rnn2(state.obs, self.obs_encoder2)
		look_out2 = self.packed_rnn2(state.description, self.look_encoder2)
		inv_out2 = self.packed_rnn2(state.inventory, self.inv_encoder2)
		state_out2 = torch.cat((obs_out2, look_out2, inv_out2), dim=1)
		# Expand the state to match the batches of actions
		state_out2 = torch.cat([state_out2[i].repeat(j, 1) for i, j in enumerate(act_sizes)], dim=0)
		z2 = torch.cat((state_out2, act_out2), dim=1)  # Concat along hidden_dim
		z2 = F.relu(self.hidden2(z2))
		q_values2 = self.act_scorer2(z2).squeeze(-1).split(act_sizes)

		# Split up the q-values by batch
		return q_values1 , q_values2

class DRRN(torch.nn.Module):
	"""
		Deep Reinforcement Relevance Network - He et al. '16
	"""

	def __init__(self, vocab_size, embedding_dim, hidden_dim):
		super(DRRN, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.obs_encoder = nn.GRU(embedding_dim, hidden_dim)
		self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
		self.inv_encoder = nn.GRU(embedding_dim, hidden_dim)
		self.act_encoder = nn.GRU(embedding_dim, hidden_dim)
		self.hidden = nn.Linear(4 * hidden_dim, hidden_dim)
		self.act_scorer = nn.Linear(hidden_dim, 1)

	def packed_rnn(self, x, rnn):
		""" Runs the provided rnn on the input x. Takes care of packing/unpacking.
			x: list of unpadded input sequences
			Returns a tensor of size: len(x) x hidden_dim
		"""
		#print('####136 x##',x)
		lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=device)
		# Sort this batch in descending order by seq length
		lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
		_, idx_unsort = torch.sort(idx_sort, dim=0)
		idx_sort = torch.autograd.Variable(idx_sort)
		idx_unsort = torch.autograd.Variable(idx_unsort)
		padded_x = pad_sequences(x)
		x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
		x_tt = x_tt.index_select(0, idx_sort)
		# Run the embedding layer
		embed = self.embedding(x_tt).permute(1, 0, 2)  # Time x Batch x EncDim
		# Pack padded batch of sequences for RNN module
		packed = nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu())
		# Run the RNN
		out, _ = rnn(packed)
		# Unpack
		out, _ = nn.utils.rnn.pad_packed_sequence(out)
		# Get the last step of each sequence
		idx = (lengths - 1).view(-1, 1).expand(len(lengths), out.size(2)).unsqueeze(0)
		out = out.gather(0, idx).squeeze(0)
		# Unsort
		out = out.index_select(0, idx_unsort)
		return out

	def forward(self, state_batch, act_batch):
		"""
			Batched forward pass.
			obs_id_batch: iterable of unpadded sequence ids
			act_batch: iterable of lists of unpadded admissible command ids
			Returns a tuple of tensors containing q-values for each item in the batch
		"""
		#print('#####state_batch',state_batch)
		# Zip the state_batch into an easy access format
		state = State(*zip(*state_batch))
		# This is number of admissible commands in each element of the batch
		act_sizes = [len(a) for a in act_batch]
		# Combine next actions into one long list
		act_batch = list(itertools.chain.from_iterable(act_batch))

		with torch.backends.cudnn.flags(enabled=False):
			act_out = self.packed_rnn(act_batch, self.act_encoder)
			# Encode the various aspects of the state
			obs_out = self.packed_rnn(state.obs, self.obs_encoder)
			look_out = self.packed_rnn(state.description, self.look_encoder)
			inv_out = self.packed_rnn(state.inventory, self.inv_encoder)
			state_out = torch.cat((obs_out, look_out, inv_out), dim=1)
			# Expand the state to match the batches of actions
			state_out = torch.cat([state_out[i].repeat(j, 1) for i, j in enumerate(act_sizes)], dim=0)
			z = torch.cat((state_out, act_out), dim=1)  # Concat along hidden_dim
			z = F.relu(self.hidden(z))
			q_values = self.act_scorer(z) #F.relu(
			q_values = q_values.squeeze(-1).split(act_sizes)

		return q_values

	def act(self, states, act_ids, sample=True):
		act_values= self.forward(states, act_ids)
		if sample:
			act_probs = [F.softmax(vals, dim=0) for vals in act_values]
			probs = [Categorical(probs) \
						for probs in act_probs]
			act_idxs = [Categorical(probs).sample() \
						for probs in act_probs]
			Z =[(act == 0.0).float() * 1e-8 for act in act_probs]
			log_action_probs = [torch.log(a_p + z)for a_p,z in zip(act_probs,Z)]
		else:
			act_idxs = [vals.argmax(dim=0).item() for vals in act_values]
			act = [vals[idx] for vals,idx in zip(act_values,act_idxs)]
			log_prob = [torch.log(a) for a in act]
		return act_idxs,act_probs,log_action_probs


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
	lengths = [len(s) for s in sequences]
	nb_samples = len(sequences)
	if maxlen is None:
		maxlen = np.max(lengths)
	# take the sample shape from the first non empty sequence
	# checking for consistency in the main loop below.
	sample_shape = tuple()
	for s in sequences:
		if len(s) > 0:
			sample_shape = np.asarray(s).shape[1:]
			break
	x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
	for idx, s in enumerate(sequences):
		if len(s) == 0:
			continue  # empty list was found
		# pre truncating
		trunc = s[-maxlen:]
		# check `trunc` has expected shape
		trunc = np.asarray(trunc, dtype=dtype)
		if trunc.shape[1:] != sample_shape:
			raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
							 (trunc.shape[1:], idx, sample_shape))
		# post padding
		x[idx, :len(trunc)] = trunc
	return x

class MoralNetwork(nn.Module):
	def __init__(self, vocab_size, embedding_dim, dim_action, hidden_dim=128):
		super(MoralNetwork, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.obs_encoder = nn.GRU(embedding_dim, hidden_dim)
		self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
		self.inv_encoder = nn.GRU(embedding_dim, hidden_dim)
		self.act_encoder = nn.GRU(embedding_dim, hidden_dim)
		#self.hidden = nn.Linear(4 * hidden_dim, hidden_dim)
		self.network = nn.Sequential(
				nn.Linear(hidden_dim*4, hidden_dim),
				nn.ReLU(),
				nn.Linear(hidden_dim, 1)#,
		)


	def packed_rnn(self, x, rnn):
		""" Runs the provided rnn on the input x. Takes care of packing/unpacking.
			x: list of unpadded input sequences
			Returns a tensor of size: len(x) x hidden_dim
		"""
		#print('####x',x)
		lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=device)
		# Sort this batch in descending order by seq length
		lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
		_, idx_unsort = torch.sort(idx_sort, dim=0)
		idx_sort = torch.autograd.Variable(idx_sort)
		idx_unsort = torch.autograd.Variable(idx_unsort)
		padded_x = pad_sequences(x)
		x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
		x_tt = x_tt.index_select(0, idx_sort)
		# Run the embedding layer
		embed = self.embedding(x_tt).permute(1, 0, 2)  # Time x Batch x EncDim
		# Pack padded batch of sequences for RNN module
		packed = nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu())
		# Run the RNN
		out, _ = rnn(packed)
		# Unpack
		out, _ = nn.utils.rnn.pad_packed_sequence(out)
		# Get the last step of each sequence
		idx = (lengths - 1).view(-1, 1).expand(len(lengths), out.size(2)).unsqueeze(0)
		out = out.gather(0, idx).squeeze(0)
		# Unsort
		out = out.index_select(0, idx_unsort)
		return out

	def forward(self, state_batch, act_batch):
		# Zip the state_batch into an easy access format
		state = State(*zip(*state_batch))
		# This is number of admissible commands in each element of the batch
		act_sizes = [len(a) for a in act_batch]
		# Combine next actions into one long list
		act_batch = list(itertools.chain.from_iterable(act_batch))
		act_out = self.packed_rnn(act_batch, self.act_encoder)
		# Encode the various aspects of the state
		obs_out = self.packed_rnn(state.obs, self.obs_encoder)
		look_out = self.packed_rnn(state.description, self.look_encoder)
		inv_out = self.packed_rnn(state.inventory, self.inv_encoder)
		state_out = torch.cat((obs_out, look_out, inv_out), dim=1)
		# Expand the state to match the batches of actions
		state_out = torch.cat([state_out[i].repeat(j, 1) for i, j in enumerate(act_sizes)], dim=0)
		state_out = torch.cat((state_out, act_out), dim=1)
		rewards = self.network(state_out).squeeze(-1).split(act_sizes) #torch.FloatTensor(
		return rewards
	#enddef
#enddef
