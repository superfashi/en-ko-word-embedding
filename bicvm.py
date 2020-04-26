# Imports
import random
import gzip
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity as cosine
import time, math
import string

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--vec', type=int, required=False)  # embedding dimension
parser.add_argument('--sen', type=int, required=False)  # number of sentences to use 
parser.add_argument('--neg', type=int, required=False)  # number of negative samples
parser.add_argument('--mar', type=int, required=False)  # margin size
parser.add_argument('--lam', type=int, required=False)  # lambda
parser.add_argument('--itr', type=int, required=False)  # number of iterations
parser.add_argument('--ver', type=bool, required=False) # verbose mode
parser.add_argument('--ten', type=str, required=False)  # previous model weights
parser.add_argument('--idx', type=str, required=False)  # previous model word2idx
parser.add_argument('--lrt', type=float, required=False)# learning rate
parser.add_argument('--bat', type=int, required=False)  # number of batches

class BiCVM():
	def __init__(self, wordVecSize=1, num_sent=-1, 
		   		 m=1, neg_samples=1, lamb=1, num_iters=50, v=False,
		   		 tensors="", idxLookup="", lr=0.1, batch=10):
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
		self.grad = None
		self.G = np.zeros(2)
		self.preTense = False
		self.preIdx = False
		self.batch = 10

		# optionally load previous model
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
		a = np.zeros(self.wordVecSize)
		b = np.zeros(self.wordVecSize)

		# using the ADD method in the paper (naive but effective)
		for word in self.enSents[j]:
			a += self.theta[self.idxLookup[word]]

		for word in self.koSents[j]:
			b += self.theta[self.idxLookup[word]]

		# generate some negative samples
		for _ in range(self.neg_samples):
			nidx = random.randint(0, len(self.koSents) - 1)
			while nidx == j:
				nidx = random.randint(0, len(self.koSents) - 1)

			# create the negative sentence
			n = np.zeros(self.wordVecSize)
			for word in self.enSents[nidx]:
				n += self.theta[self.idxLookup[word]]

			# calculate loss
			innerHinge = self.m + 0.5 * np.linalg.norm(a - b) - 0.5 * np.linalg.norm(a - n)
			obj += max(0, innerHinge)

			# accumulate gradients
			agrad = np.zeros(self.wordVecSize)
			bgrad = np.zeros(self.wordVecSize)
			ngrad = np.zeros(self.wordVecSize)

			# TODO: are these the right gradients?
			if innerHinge > 0:
				agrad += (n - b)
				bgrad += (b - a)
				ngrad += (a - n)

				# print(agrad)
				# print(bgrad)
				# print(ngrad)

			# apply gradients to relavant words
			if ngrad.any() > 0:
				for word in self.koSents[nidx]:
					if word in self.idxLookup:
						self.grad[self.idxLookup[word]] += ngrad
			if agrad.any() > 0:
				for word in self.enSents[j]:
					if word in self.idxLookup:
						self.grad[self.idxLookup[word]] += agrad
			if bgrad.any() > 0:
				for word in self.koSents[j]:
					if word in self.idxLookup:
						self.grad[self.idxLookup[word]] += bgrad


		return obj

	# update parameters
	def update_param(self):
		self.grad = self.grad + self.lamb * self.theta
		if self.G.any() > 0:
			self.G = self.G + self.grad * self.grad
		else:
			self.G = self.grad * self.grad

		self.theta = self.theta - (self.lr / np.sqrt(self.G + 1e-8)) * self.grad

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
			self.grad = np.zeros((len(self.idxLookup), self.wordVecSize))


			# pass through every sentence and accumulate gradients
			for i in range(len(self.enSents)):
				loss += self.loss(i)
				if i != 0 and i % (len(self.enSents) / self.batch) == 0:
					# gradient descent
					self.update_param()
					self.grad = np.zeros((len(self.idxLookup), self.wordVecSize))

			self.update_param()

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

	# Dump word vectors to txt file
	def vec_to_txt(self):
		def iseng(word):
			printable = set(string.printable)
			for char in word:
				if char not in printable:
					return False
			return True

		# get relevant english and korean dictionaries
		print('---Splitting dictionaries---')
		en = dict()
		ko = dict()

		for word in self.idxLookup:
			if not iseng(word):
				ko[word] = self.theta[self.idxLookup[word]]
			else:
				en[word] = self.theta[self.idxLookup[word]]

		# dump to txt
		with open('en.txt', 'w') as f:
			for word in en:
				f.write(word + ' ')
				f.write(" ".join(str(v) for v in en[word]) + '\n')

		with open('ko.txt', 'w') as f:
			for word in ko:
				f.write(word + ' ')
				f.write(" ".join(str(v) for v in ko[word]) + '\n')


	# Generate predictions based on validation data
	def gen_predictions(self, val, out):
		predictions = []

		# make predictions
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

				similarity = cosine(enVector, koVector)
				similarity = abs(similarity[0][0])

				if similarity >= 0.8:
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
	args = parser.parse_args()

	model = BiCVM(wordVecSize=args.vec, num_sent=args.sen, num_iters=args.itr, 
				  neg_samples=args.neg, m=args.mar, tensors=args.ten, v=args.ver,
				  lr=args.lrt, idxLookup=args.idx)

	# print("---Loading Training Data---")
	# model.load_data('train.txt.gz')
	# print(len(model.theta))
	# print(model.theta[0])

	# print("---Training Model---")
	# model.train()
	# print(model.theta[0])

	# print("---Writing Predictions---")
	# model.gen_predictions("val.txt", "bicvm_pred.txt")

	print("---Dumping Word Vectors---")
	model.vec_to_txt()





if __name__ == '__main__':
	main()




