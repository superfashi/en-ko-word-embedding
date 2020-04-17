# Imports
import torch
import random
import gzip

class BiCVM():
	def __init__(self, wordVecSize=1, num_sent=-1, m=1, neg_samples=1, lamb=1, num_iters=50, v=False):
		self.wordVecSize = wordVecSize
		self.num_sent = num_sent
		self.m = m
		self.neg_samples = neg_samples
		self.lamb = lamb
		self.num_iters = num_iters
		self.en = dict()
		self.ko = dict()
		self.enSents = []
		self.koSents = []
		self.theta = []
		self.v = v

	def debug_print(self, args):
		if self.v:
			print(args)

	# Load the training data, sets our dictionaries of words to embeddings and our list
	# of sentences to their initial values
	def load_data(self, train_file):
		count = 0
		with gzip.open(train_file, 'rt', encoding='utf_8') as f:
		  for line in f:
		  	# break if we've read the required number of sentences
		  	if self.num_sent != -1 and count >= self.num_sent:
		  		break

		  	# display completion percentage to the user
		  	if self.num_sent == -1 and (count + 1) % 100000 == 0:
		  		percent = (count / 2561937) * 100
		  		print(str(percent) + "% complete")

		  	# process each line
		  	data = line[:-1].split('\t')
		  	enWords = data[0].split()
		  	koWords = data[1].split()

		  	# store tensors 
		  	for i in range(len(enWords)):
		  		if enWords[i] not in self.en:
		  			self.en[enWords[i]] = torch.rand(1, self.wordVecSize, requires_grad = True)

		  	for i in range(len(koWords)):
		  		if koWords[i] not in self.ko:
		  			self.ko[koWords[i]] = torch.rand(1, self.wordVecSize, requires_grad = True)

		  	self.enSents.append(enWords)
		  	self.koSents.append(koWords)

		  	count += 1

		self.theta = list(self.en.values()) + list(self.ko.values())

	# calculates the regularization term and returns it
	def regularization(self):
		return (self.lamb / 2) * (torch.norm(torch.stack(self.theta))**2)

	# energy is defined as || f(a) - g(b) ||^2
	def energy(self, a, b):
		return torch.norm(torch.sum(a,0) - torch.sum(b,0))**2

	# Define loss function, see paper (https://arxiv.org/abs/1312.6173) for details
	def loss(self, j):
		# J(theta) = sum over all pairs (sum over all negative samples (Ehl(a, b, n_i) + regularization))
		# Ehl(a, b, n_i) = max(0, m + Ebi(a, b) -- Ebi(a, n))
		# Ebi(a, b) = energy(a, b)
		# f = g = vector sum of all words in sentence
		obj = 0

		# convert sentences to torch matrices
		enTensor = torch.empty(len(self.enSents[j]), self.wordVecSize)
		koTensor = torch.empty(len(self.koSents[j]), self.wordVecSize)

		for i in range(len(self.enSents[j])):
			enTensor[i] = self.en[self.enSents[j][i]]
		for i in range(len(self.koSents[j])):
			koTensor[i] = self.ko[self.koSents[j][i]]

		# compute energy of the positive sample
		goodEnergy = self.energy(enTensor, koTensor)

		# for all required negative pairs, compute energy
		for i in range(self.neg_samples): 
			neg_samp = random.randint(0, len(self.enSents) - 1)
			while neg_samp == j:
				neg_samp = random.randint(0, len(self.enSents) - 1)

			# convert negative sentence to torch
			# TODO: should we turn off gradients for the negative sample (if even possible)?
			negTensor = torch.empty(len(self.enSents[neg_samp]), self.wordVecSize)

			for k in range(len(self.enSents[neg_samp])):
				negTensor[k] = self.en[self.enSents[neg_samp][k]]

			# compute negative energy
			negEnergy = self.energy(enTensor, negTensor)

			# update objective fctn
			obj += max(0, self.m + goodEnergy - negEnergy)

		# add regularization
		obj += self.regularization()

		return obj

	# Define main training loop
	def train(self):
		optimizer = torch.optim.Adagrad(self.theta)
		count = 0

		for _ in range(self.num_iters):
			if (count + 1) % 10 == 0:
				percent = (count / self.num_iters) * 100
				print("**** " + str(percent) + "% complete ****")

			sent_count = 0
			optimizer.zero_grad()
			loss = 0
			for i in range(len(self.enSents)):
				if (sent_count + 1) % 10 == 0:
					percent = (sent_count / len(self.enSents)) * 100
					print(str(percent) + "% of the sentences processed")
				loss += self.loss(i)
				sent_count += 1
			
			loss.backward()
			optimizer.step()

			count += 1

	# Generate predictions based on validation data
	def gen_predictions(self, val, out):
		cos = torch.nn.CosineSimilarity()
		predictions = []

		# make predictions
		# TODO: perhaps a better method for prediction is needed
		with open(val, "r") as f:
			for line in f:
				data = line[:-1].split("\t")
				if data[1] in self.ko and data[0] in self.en:
					similarity = cos(self.ko[data[1]], self.en[data[0]])
					if similarity[0] >= 0.7:
						predictions.append(True)
					else:
						predictions.append(False)

					self.debug_print(data[1])
					self.debug_print(data[0])
					self.debug_print(similarity)
					self.debug_print(' ')
					# Doesn't work: predictions.append(similarity[0] >= 0.9)
				else:
					# for out of vocab words, just predict false for now
					predictions.append(False)

		# write predictions to output
		with open(out, "w") as f:
			for line in predictions:
				f.write(str(line) + "\n")

# Train and test the model
def main():
	model = BiCVM(wordVecSize=128, num_sent=10000, num_iters=1, v=0)

	print("---Loading Training Data---")
	model.load_data('train.txt.gz')

	print("---Training Model---")
	model.train()

	print("---Writing Predictions---")
	model.gen_predictions("val.txt", "bicvm_pred.txt")



if __name__ == '__main__':
	main()




