# Imports
import random
import gzip
import numpy as np
import pickle
from scipy.spatial.distance import cosine
import time, math

class BiCVM():
	def __init__(self, wordVecSize=1, num_sent=-1, 
		   		 m=1, neg_samples=1, lamb=1, num_iters=50, v=False,
		   		 tensors="", idxLookup="", lr=0.1):
		self.wordVecSize = wordVecSize
		self.num_sent = num_sent
		self.m = m
		self.neg_samples = neg_samples
		self.lamb = lamb
		self.num_iters = num_iters
		self.idxLookup = dict()
		self.enSents = []
		self.koSents = []
		self.theta = []
		self.lr = lr
		self.v = v
		self.grad = dict()
		self.G = dict()
		self.preTense = False
		self.preIdx = False

		if tensors:
			self.preTense = True
			with open(tensors, 'rb') as f:
				self.theta = pickle.load(f)
		if idxLookup:
			self.preIdx = True
			with open(idxLookup, 'rb') as f:
				self.idxLookup = pickle.load(f)

	def debug_print(self, args):
		if self.v:
			print(args)

	# Load the training data, sets our dictionaries of words to embeddings and our list
	# of sentences to their initial values
	def load_data(self, train_file):
		count = 0
		prevLen = len(self.idxLookup)
		idx = len(self.idxLookup)
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

		  	# store words
		  	for word in enWords:
		  		if word not in self.idxLookup:
		  			self.idxLookup[word] = idx
		  			idx += 1
		  		

		  	for word in koWords:
		  		if word not in self.idxLookup:
		  			self.idxLookup[word] = idx
		  			idx += 1


		  	self.enSents.append(enWords)
		  	self.koSents.append(koWords)

		  	count += 1

		print(str(idx) + " words in corpus")
		if not self.preTense:
			self.theta = np.random.rand(idx, self.wordVecSize)
		elif idx - prevLen > 0:
			print(self.theta.shape)
			newWrds = np.random.rand(idx - prevLen, self.wordVecSize)
			self.theta = np.concatenate((self.theta, newWrds), axis=0)
			print(newWrds.shape)
			print(self.theta.shape)
	# Define loss function, see paper (https://arxiv.org/abs/1312.6173) for details
	def loss(self, j):
		obj = 0

		# create vector representation of each sentence
		a = np.zeros((1, self.wordVecSize))
		b = np.zeros((1, self.wordVecSize))

		# using the ADD method in the paper (naive but effective)
		for word in self.enSents[j]:
			a += self.theta[self.idxLookup[word]]

		for word in self.koSents[j]:
			b += self.theta[self.idxLookup[word]]

		# generate some negative samples
		for _ in range(self.neg_samples):
			nidx = random.randint(0, len(self.enSents) - 1)
			while nidx == j:
				nidx = random.randint(0, len(self.enSents) - 1)

			# create the negative sentence
			n = np.zeros((1, self.wordVecSize))
			for word in self.enSents[nidx]:
				n += self.theta[self.idxLookup[word]]

			# calculate loss
			innerHinge = self.m + np.linalg.norm(a - b) - np.linalg.norm(a - n)
			obj += max(0, innerHinge)

			# accumulate gradients
			agrad = np.zeros((1, self.wordVecSize))
			bgrad = np.zeros((1, self.wordVecSize))
			ngrad = np.zeros((1, self.wordVecSize))

			# TODO: are these the right gradients?
			if innerHinge > 0:
				agrad += (2 * (a - b) - 2 * (a - n))
				bgrad += (-2 * (a - b))
				ngrad += (2 * (a - n))

				# print(agrad)
				# print(bgrad)
				# print(ngrad)

			# apply gradients to relavant words
			if ngrad.any() > 0:
				for word in self.enSents[nidx]:
					if word in self.grad:
						self.grad[word] += ngrad
					else:
						self.grad[word] = ngrad
			if agrad.any() > 0:
				for word in self.enSents[j]:
					if word in self.grad:
						self.grad[word] += agrad
					else:
						self.grad[word] = agrad
			if bgrad.any() > 0:
				for word in self.koSents[j]:
					if word in self.grad:
						self.grad[word] += bgrad
					else:
						self.grad[word] = bgrad

		return obj

	# Define main training loop
	def train(self):
		def time_since(since):
		    s = time.time() - since
		    m = math.floor(s / 60)
		    s -= m * 60
		    return '%dm %ds' % (m, s)

		start = time.time() # track the time

		# perform the specified number of iterations
		for count in range(self.num_iters):
			# reset loss and gradients
			loss = 0
			self.grad.clear()

			# pass through every sentence and accumulate gradients
			for i in range(len(self.enSents)):
				if (i + 1) % (self.num_sent / 10) == 0:
					percent = (i / len(self.enSents)) * 100
					print(str(percent) + "% of the sentences processed")
				loss += self.loss(i)

			# account for regularization
			loss += (self.lamb / 2) * (np.linalg.norm(self.theta)**2)

			# gradient descent
			for word in self.grad:
				# compute gradient
				regGrad = 2 * self.theta[self.idxLookup[word]]
				trueGrad = self.grad[word] + regGrad

				# adagrad
				if word in self.G:
					self.G[word] += trueGrad**2
				else:
					self.G[word] = trueGrad**2

				adaLR = self.lr / (self.G[word] + 1e-8)**0.5
				update = adaLR * trueGrad

				# update theta
				self.theta[self.idxLookup[word]] -= update[0]

			# print training info
			current_time = time_since(start)
			print('Iteration ' + str(count + 1) + '/' 
				  + str(self.num_iters) 
				  + ' complete. Time elapsed: ' 
				  + str(current_time) + '. Loss: ' 
				  + str(loss / len(self.enSents)))

			# checkpoint
			with open('bicvm_theta', 'wb') as f:
				pickle.dump(self.theta, f)
			with open('bicvm_lookup', 'wb') as f:
				pickle.dump(self.idxLookup, f)

	# Generate predictions based on validation data
	def gen_predictions(self, val, out):
		predictions = []

		# make predictions
		# TODO: change how out of vocab is handled and maybe how predictions are made
		with open(val, "r") as f:
			for line in f:
				data = line[:-1].split("\t")

				enVector = np.zeros((1, self.wordVecSize))
				koVector = np.zeros((1, self.wordVecSize))

				for word in data[0].split():
					if word in self.idxLookup:
						enVector += self.theta[self.idxLookup[word]]
				for word in data[1].split():
					if word in self.idxLookup:
						koVector += self.theta[self.idxLookup[word]]

				similarity = 1 - cosine(enVector, koVector) if enVector.all() != 0 and koVector.all() != 0 else 0

				if similarity >= 0.5:
					predictions.append(True)
				else:
					predictions.append(False)

				if similarity != 0:
					self.debug_print(data[1])
					self.debug_print(data[0])
					self.debug_print(similarity)
					self.debug_print(' ')

		# write predictions to output
		with open(out, "w") as f:
			for line in predictions:
				f.write(str(line) + "\n")

# Train and test the model
def main():
	# model = BiCVM(wordVecSize=40, num_sent=500000, num_iters=9,
	# 			  neg_samples=5, m=5, lr=0.1)

	model = BiCVM(wordVecSize=40, num_sent=-1, num_iters=15,
				  neg_samples=5, m=5, tensors='bicvm_theta',
				  idxLookup='bicvm_lookup')

	print("---Loading Training Data---")
	model.load_data('train.txt.gz')
	print(model.theta[0])

	print("---Training Model---")
	model.train()
	print(model.theta[0])

	print("---Writing Predictions---")
	model.gen_predictions("val.txt", "bicvm_pred.txt")



if __name__ == '__main__':
	main()




