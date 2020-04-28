from transformers import *
import torch
import gzip

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

model.eval()

input_word = torch.tensor(tokenizer.encode("bats")).unsqueeze(0)

outputs = model(input_word)

hidden = outputs[0].squeeze(0).detach().numpy()

dim = hidden[0].shape[0]

print(dim)

# generate dictionaries
en = dict()
ko = dict()

# populate dictionaries
print('---Populating dictionaries---')
count = 0
with open('val.txt', 'r') as f:
	for line in f:
		if count >= 2000:
			break

		data = line[:-1].split('\t')
		enWords = data[0].split()
		koWords = data[1].split()

		for word in enWords:
			if word not in en:
				input_word = torch.tensor(tokenizer.encode(word)).unsqueeze(0)
				outputs = model(input_word)
				hidden = outputs[0].squeeze(0).detach().numpy()
				#print(word, outputs[0].shape)

				en[word] = hidden[0]
		#print("--------")

		for word in koWords:
			if word not in ko:
				input_word = torch.tensor(tokenizer.encode(word)).unsqueeze(0)
				outputs = model(input_word)
				hidden = outputs[0].squeeze(0).detach().numpy()
				#print(outputs[0].shape)

				ko[word] = hidden[0]

		count += 1
		if count % 1000 == 0:
			print(str(count) + ' finished')

print('---Write Vectors---')

# write vectors
with open('en.txt', 'w') as f:
	f.write(str(len(en)) + " " + str(dim) + '\n')
	for word in en:
		f.write(word + ' ')
		f.write(" ".join(str(v) for v in en[word]) + '\n')

with open('ko.txt', 'w') as f:
	f.write(str(len(ko)) + " " + str(dim) + '\n')
	for word in ko:
		f.write(word + ' ')
		f.write(" ".join(str(v) for v in ko[word]) + '\n')