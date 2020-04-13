''' Evaluates the given bilingual word embeddings model '''

# Create pairs of (prediction, actual)
def readValidationData(val_data, val_pred):
	gold = []
	pred = []
	with open(val_data, "r") as f:
		for line in f:
			splitLine = line[:-1].split('\t')
			if splitLine[2] != 'True' and splitLine[2] != 'False':
				raise ValueError('validation data is malformed')
			gold.append(splitLine[2])
	with open(val_pred, "r") as f:
		for line in f:
			pred.append(line[:-1])

	if len(gold) != len(pred):
		raise ValueError('Array length mismatch')

	return gold, pred


# Calculate precision recall and f1
def calcPRF(gold, pred):
	precision = 0
	recall = 0
	f1 = 0

	tp = 0
	fp = 0
	tn = 0
	fn = 0

	for i in range(len(gold)):
		if pred[i] == 'True':
			if gold[i] == 'True':
				tp += 1
			else:
				fp += 1
		else:
			if gold[i] == 'True':
				fn += 1
			else:
				tn += 1

	precision = 0 if tp + fp == 0 else tp / (tp + fp)
	recall = 0 if tp + fn == 0 else tp / (tp + fn)
	f1 = 0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

	return precision, recall, f1


def main():
	gold, pred = readValidationData('val.txt', 'val_pred.txt')
	p, r, f = calcPRF(gold, pred)

	print('---Results---')
	print('Precision: ' + str(round(p, 3)))
	print('Recall: ' + str(round(r, 3)))
	print('F1: ' + str(round(f, 3)))

if __name__ == '__main__':
	main()

