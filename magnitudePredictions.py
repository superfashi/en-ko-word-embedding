from pymagnitude import *

enVectors = Magnitude('enReal.magnitude')
koVectors = Magnitude('koReal.magnitude')

predictions = []
with open('val.txt', 'r') as f:
	for line in f:
		data = line[:-1].split("\t")
		envec = enVectors.query(data[0]) # get the english word's vector rep
		mostSimilar = [k for k, v in koVectors.most_similar(envec, topn = 5)]
		if data[1] in mostSimilar:
			predictions.append(True)
		else:
			predictions.append(False)


# write predictions to output
with open("bicvm_mag_pred", "w") as f:
	for line in predictions:
		f.write(str(line) + "\n")
