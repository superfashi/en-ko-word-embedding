# Imports
import torch
import random
import gzip

# Load the training data, assign each word a random vector that will be later
# optimized into a proper embedding
def load_data(wordVecSize=1):
	en = []
	ko = []
	enVec = []
	koVec = []
	count = 1
	with gzip.open('train.txt.gz', 'rt', encoding='utf_8') as f:
	  for line in f:
	  	if count % 100000 == 0:
	  		precent = (count / 2561937) * 100
	  		print(str(precent) + "% complete")
	  	data = line[:-1].split('\t')
	  	enWords = data[0].split()
	  	koWords = data[1].split()
	  	en.append(enWords)
	  	ko.append(koWords)
	  	enVec.append(torch.rand(len(enWords), wordVecSize))
	  	koVec.append(torch.rand(len(koWords), wordVecSize))
	  	count += 1
	return en, ko, enVec, koVec

# Define loss function
def loss(enVec, koVec, m=1, neg_samples=1, lamb=1):
	# calculates the regularization term and returns it
	def regu(theta):
		return (lamb / 2) * (torch.norm(theta)**2)

	# energy is the same as Ebi below
	def energy(a, b):
		return torch.norm(torch.sum(a,0) - torch.sum(b,0))**2
	# J(theta) = sum over all pairs (sum over all negative samples (Ehl(a, b, n_i) + regularization))
	# Ehl(a, b, n_i) = max(0, m + Ebi(a, b) -- Ebi(a, n))
	# Ebi(a, b) = || f(a) - g(b) ||^2
	# f = g = vector sum of all words in sentence
	obj = 0

	for j in range(len(enVec)):
		goodEnergy = energy(enVec[j], koVec[j]) # energy of the correct pair
		for i in range(neg_samples): # sample some bad pairs and use their energy
			negative_sample = random.randint(0, len(enVec))
			while negative_sample == j:
				negative_sample = random.randint(0, len(enVec))

			# update objective fctn
			obj += max(0, m + goodEnergy - energy(enVec[j], enVec[i])) + regu(enVec[j] + koVec[j])

	return obj

# Define main training loop
def train(en, ko, num_iters=1):
	optimizer = torch.optim.Adagrad(en + ko)

	for _ in range(num_iters):
		for i in range(len(en)):
			optimizer.zero_grad()
			l = loss(en, ko)
			l.backward()
			optimizer.step()

	return en, ko

# TODO: write trained embeddgins to output
# perhaps in the format: "word \t vector"
def write_to_output(en, ko, enVec, koVec):
	pass

# Train the model
def main():
	en = [torch.rand(i, 2, requires_grad = True) for i in range(2, 6)]
	ko = [torch.rand(i, 2, requires_grad = True) for i in range(2, 6)]
	
	print('Before Training')
	print(en[0])
	print(ko[0])

	train(en, ko, num_iters=100)

	print('After Training')
	print(en[0])
	print(ko[0])

if __name__ == '__main__':
	main()




