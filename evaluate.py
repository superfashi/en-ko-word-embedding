# Evaluates the given bilingual word embeddings model
import argparse
import pprint

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()
parser.add_argument('--outputfile', type=str, required=True) 
parser.add_argument('--model', type=str, required=True)

# Takes the output file as an argument and parses it into a list of pairs
# each pair[0] being the english word and pair[1] being the korean word
def readOutFile(outfile):
	enkoPairs = []
	with open(outfile, 'r') as f:
		for line in f:
			data = line.split('\t')
			enkoPairs.append((data[0], data[1]))
	return enkoPairs

# TODO: Load the model from a file so we can query it for word embeddings
def loadModel(model):
	pass

# Take the model and word pairs as input and outputs the average cosine similiarity
# between all pairs
def calcCosSim(enkoPairs, model):
	cosSims = []
	for pair in enkoPairs:
		# TODO query the model for embeddings
		# calculate the cosine similarity and add it to the list
	return sum(cosSims) / len(cosSims)


def main(args):
	enkoPairs = readOutFile(args.outputfile)
	model = loadModel(args.model)
	print(calcCosSim(enkoPairs, model))

if __name__ == '__main__':
	print("Evaluating Model")
	args = parser.parse_args()
    pp.pprint(args)
    main(args)