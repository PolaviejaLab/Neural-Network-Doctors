
import numpy as np




def performCalc(estimates,truths):

	#G computes the performance of a group: Accuracy, Youden's index, Sensitivity and Specificity.
	#G The data and ground truth are provided in the following format:
	#G estimates: [[c1d1, c1d2, c1d3, ... ], [c2d1, c2d2, c2d3, ... ], ... ], with c1d1 = doc 1 of case 1. -1 negative, 1 positive
	#G truth: [[case1], [case2], [case3], ... ], with [1 0] negative, [0 1] positive

	numData = truths.shape[0]
	if numData > 0:
		numPos = np.sum(truths[:,0]==0)
		numNeg = np.sum(truths[:,0]==1)
	else:
		numPos = 0.0
		numNeg = 0.0

	correct=0.0
	truePos=0.0
	trueNeg=0.0

	for i in xrange(numData):

		votes = np.sum(estimates[i][:])

		if votes > 0 and truths[i,0] == 0: #G Positive
			correct+=1
			truePos+=1

		if votes < 0 and truths[i,0] == 1: #G Negative
			correct+=1
			trueNeg+=1

		if votes == 0:
			correct+=0.5
			if truths[i,0] == 0:
				truePos+=0.5
			if truths[i,0] == 1:
				trueNeg+=0.5

	if numData == 0:
		accuracy = np.NAN
		sensitivity = np.NAN
		specificity = np.NAN
		youden = np.NaN

	elif numNeg == 0:
		accuracy = correct/numData
		sensitivity = truePos/numPos
		specificity = np.NAN
		youden = sensitivity

	elif numPos == 0:
		accuracy = correct/numData
		sensitivity = np.NAN
		specificity = trueNeg/numNeg
		youden = specificity

	else:
		accuracy = correct/numData
		sensitivity = truePos/numPos
		specificity = trueNeg/numNeg
		youden = sensitivity + specificity - 1

	return accuracy, youden, sensitivity, specificity




def performCalcOptimistic(estimates,truths):

	#G computes the performance of a group, with the rule that if there is a tie, the case is predicted as negative
	#G Performance is: Accuracy, Youden's index, Sensitivity and Specificity.
	#G The data and ground truth are provided in the following format:
	#G estimates: [[c1d1, c1d2, c1d3, ... ], [c2d1, c2d2, c2d3, ... ], ... ], with c1d1 = doc 1 of case 1. -1 negative, 1 positive
	#G truth: [[case1], [case2], [case3], ... ], with [1 0] negative, [0 1] positive

	numData = truths.shape[0]
	if numData > 0:
		numPos = np.sum(truths[:,0]==0)
		numNeg = np.sum(truths[:,0]==1)
	else:
		numPos = 0.0
		numNeg = 0.0

	correct=0.0
	truePos=0.0
	trueNeg=0.0

	for i in xrange(numData):

		votes = np.sum(estimates[i][:])

		if votes > 0 and truths[i,0] == 0: #G Positive
			correct+=1
			truePos+=1

		if votes <= 0 and truths[i,0] == 1: #G Negative
			correct+=1
			trueNeg+=1

	if numData == 0:
		accuracy = np.NAN
		sensitivity = np.NAN
		specificity = np.NAN
		youden = np.NaN

	elif numNeg == 0:
		accuracy = correct/numData
		sensitivity = truePos/numPos
		specificity = np.NAN
		youden = sensitivity

	elif numPos == 0:
		accuracy = correct/numData
		sensitivity = np.NAN
		specificity = trueNeg/numNeg
		youden = specificity

	else:
		accuracy = correct/numData
		sensitivity = truePos/numPos
		specificity = trueNeg/numNeg
		youden = sensitivity + specificity - 1

	return accuracy, youden, sensitivity, specificity




def performCalcWeights(estimates,truths,weights):

	#G computes the performance of a group, weighting the votes of eac subjetc
	#G Performance is: Accuracy, Youden's index, Sensitivity and Specificity.
	#G The data and ground truth are provided in the following format:
	#G estimates: [[c1d1, c1d2, c1d3, ... ], [c2d1, c2d2, c2d3, ... ], ... ], with c1d1 = doc 1 of case 1. -1 negative, 1 positive
	#G truth: [[case1], [case2], [case3], ... ], with [1 0] negative, [0 1] positive
	#G weights: [[w1c1, w1c2, w1c3, ... ], [w2c1, w2c2, w2c3, ... ], ... ], with w1c1 = weight of doc 1 of case 1
	#G weights not necessarily normalized. The function normalizes the weights

	numData = truths.shape[0]
	if numData > 0:
		numPos = np.sum(truths[:,0]==0)
		numNeg = np.sum(truths[:,0]==1)
	else:
		numPos = 0.0
		numNeg = 0.0

	correct=0.0
	truePos=0.0
	trueNeg=0.0

	for i in xrange(numData):
		#print estimates[0,:]
		normWeights = [float(j) for j in weights[i]]/np.sum(weights[i]) #G normalized weights
		#print normWeights
		votes = np.sum(np.multiply(estimates[i][:],normWeights))

		if votes > 0 and truths[i,0] == 0: #G Positive
			correct+=1
			truePos+=1

		if votes < 0 and truths[i,0] == 1: #G Negative
			correct+=1
			trueNeg+=1

		if votes == 0:
			correct+=0.5
			if truths[i,0] == 0:
				truePos+=0.5
			if truths[i,0] == 1:
				trueNeg+=0.5

	if numData == 0:
		accuracy = np.NAN
		sensitivity = np.NAN
		specificity = np.NAN
		youden = np.NaN

	elif numNeg == 0:
		accuracy = correct/numData
		sensitivity = truePos/numPos
		specificity = np.NAN
		youden = sensitivity

	elif numPos == 0:
		accuracy = correct/numData
		sensitivity = np.NAN
		specificity = trueNeg/numNeg
		youden = specificity

	else:
		accuracy = correct/numData
		sensitivity = truePos/numPos
		specificity = trueNeg/numNeg
		youden = sensitivity + specificity - 1

	return accuracy, youden, sensitivity, specificity




def performCalcHigher(estimates,truths,characteristic):

	#G computes the performance of a group, using the rule of choosing the opinion of the subject(s) with higher characteristic value
	#G Performance is: Accuracy, Youden's index, Sensitivity and Specificity.
	#G The data and ground truth are provided in the following format:
	#G estimates: [[c1d1, c1d2, c1d3, ... ], [c2d1, c2d2, c2d3, ... ], ... ], with c1d1 = doc 1 of case 1. -1 negative, 1 positive
	#G truth: [[case1], [case2], [case3], ... ], with [1 0] negative, [0 1] positive
	#G competence: [[c1c1, c1c2, c1c3, ... ], [c2c1, c2c2, c2c3, ... ], ... ], with c1c1 = competence of doc 1 of case 1

	numData = truths.shape[0]
	if numData > 0:
		numPos = np.sum(truths[:,0]==0)
		numNeg = np.sum(truths[:,0]==1)
	else:
		numPos = 0.0
		numNeg = 0.0

	correct=0.0
	truePos=0.0
	trueNeg=0.0

	for i in xrange(numData):

		compet = np.float64(characteristic[i])
		indSort = np.argsort(compet)
		indSort = indSort[::-1]
		indBest = np.where(compet == compet[indSort[0]])

		votes = 0.0
		for j in xrange(len(indBest[0])):
			votes += estimates[i][indBest[0][j]]

		if votes > 0 and truths[i,0] == 0: #G Positive
			correct+=1
			truePos+=1

		if votes < 0 and truths[i,0] == 1: #G Negative
			correct+=1
			trueNeg+=1

		if votes == 0:
			correct+=0.5
			if truths[i,0] == 0:
				truePos+=0.5
			if truths[i,0] == 1:
				trueNeg+=0.5

	if numData == 0:
		accuracy = np.NAN
		sensitivity = np.NAN
		specificity = np.NAN
		youden = np.NaN

	elif numNeg == 0:
		accuracy = correct/numData
		sensitivity = truePos/numPos
		specificity = np.NAN
		youden = sensitivity

	elif numPos == 0:
		accuracy = correct/numData
		sensitivity = np.NAN
		specificity = trueNeg/numNeg
		youden = specificity

	else:
		accuracy = correct/numData
		sensitivity = truePos/numPos
		specificity = trueNeg/numNeg
		youden = sensitivity + specificity - 1

	return accuracy, youden, sensitivity, specificity




def performance(data, nums):
	'''
	Computes performance of all the group-case data at once,
	as if they were produced by as many different diagnostician groups
	as number of group-case data.
	So group size is number of group-case instances (numGroups*numCases)
	For a computation by groups, see performSplit function
	'''

	train = data[0]
	trainT = data[1]
	valid = data[2]
	validT = data[3]
	test = data[4]
	testT = data[5]
	
	wPos = nums[4]
	wNeg = nums[5]

	estimatesTrain = [z[::6] for z in train[:]]
	estimatesValid = [z[::6] for z in valid[:]]

	truthsTrain = trainT
	truthsValid = validT

	accuraciesTrain = [z[2::6] for z in train[:]]
	accuraciesValid = [z[2::6] for z in valid[:]]

	youdensTrain = [z[3::6] for z in train[:]]
	youdensValid = [z[3::6] for z in valid[:]]

	sensitsTrain = [z[4::6] for z in train[:]]
	sensitsValid = [z[4::6] for z in valid[:]]

	specifsTrain = [z[5::6] for z in train[:]]
	specifsValid = [z[5::6] for z in valid[:]]

	adjYoudsTrain = [2*sum(i)-1 for i in zip([wPos * z for z in sensitsTrain],[wNeg * z for z in specifsTrain])] 
	adjYoudsValid = [2*sum(i)-1 for i in zip([wPos * z for z in sensitsValid],[wNeg * z for z in specifsValid])] 

	confidencesTrain = [z[1::6] for z in train[:]]
	confidencesValid = [z[1::6] for z in valid[:]]

	#G the structure of the following lists will be:
	#G [[acc youd sens spec](in training), [acc youd sens spec](in validation)]

	#G If you want instead to get [[acctrain accvalid],[youdtrain youdvalid],[etc],[etc]], you should do:
	#G perf_Maj = [[performCalc(estimatesTrain, truthsTrain)[i], performCalc(estimatesValid, truthsValid)[i]] for i in xrange(4)]

	#G performance of the majority voting
	perf_Maj = [performCalc(estimatesTrain, truthsTrain), performCalc(estimatesValid, truthsValid)]
	perf_Maj = [perf_Maj[0]+(2*(wPos*perf_Maj[0][2] + wNeg*perf_Maj[0][3])-1,), perf_Maj[1]+(2*(wPos*perf_Maj[1][2] + wNeg*perf_Maj[1][3])-1,)] # append adjusted Youden's

	#G performance of the 'optimistic' strategy (in case of tie, declare as negative)
	perf_Opt = [performCalcOptimistic(estimatesTrain, truthsTrain), performCalcOptimistic(estimatesValid, truthsValid)]
	perf_Opt = [perf_Opt[0]+(2*(wPos*perf_Opt[0][2] + wNeg*perf_Opt[0][3])-1,), perf_Opt[1]+(2*(wPos*perf_Opt[1][2] + wNeg*perf_Opt[1][3])-1,)] # append adjusted Youden's

	#G performance of the 'choose the most accurate' strategy
	perf_Acc = [performCalcHigher(estimatesTrain, truthsTrain, accuraciesTrain), performCalcHigher(estimatesValid, truthsValid, accuraciesValid)]
	perf_Acc = [perf_Acc[0]+(2*(wPos*perf_Acc[0][2] + wNeg*perf_Acc[0][3])-1,), perf_Acc[1]+(2*(wPos*perf_Acc[1][2] + wNeg*perf_Acc[1][3])-1,)] # append adjusted Youden's

	#G performance of the 'choose the best Youden's index' strategy
	perf_Youd = [performCalcHigher(estimatesTrain, truthsTrain, youdensTrain), performCalcHigher(estimatesValid, truthsValid, youdensValid)]
	perf_Youd = [perf_Youd[0]+(2*(wPos*perf_Youd[0][2] + wNeg*perf_Youd[0][3])-1,), perf_Youd[1]+(2*(wPos*perf_Youd[1][2] + wNeg*perf_Youd[1][3])-1,)] # append adjusted Youden's

	#G performance of the 'choose the best adjusted Youden's' strategy
	perf_Adj = [performCalcHigher(estimatesTrain, truthsTrain, adjYoudsTrain), performCalcHigher(estimatesValid, truthsValid, adjYoudsValid)]
	perf_Adj = [perf_Adj[0]+(2*(wPos*perf_Adj[0][2] + wNeg*perf_Adj[0][3])-1,), perf_Adj[1]+(2*(wPos*perf_Adj[1][2] + wNeg*perf_Adj[1][3])-1,)] # append adjusted Youden's

	#G performance of the 'choose the most confident' strategy
	perf_Conf = [performCalcHigher(estimatesTrain, truthsTrain, confidencesTrain), performCalcHigher(estimatesValid, truthsValid, confidencesValid)]
	perf_Conf = [perf_Conf[0]+(2*(wPos*perf_Conf[0][2] + wNeg*perf_Conf[0][3])-1,), perf_Conf[1]+(2*(wPos*perf_Conf[1][2] + wNeg*perf_Conf[1][3])-1,)] # append adjusted Youden's

	#G performance weighting subjects by their previous accuracy
	perf_AccW = [performCalcWeights(estimatesTrain, truthsTrain, accuraciesTrain), performCalcWeights(estimatesValid, truthsValid, accuraciesValid)]
	perf_AccW = [perf_AccW[0]+(2*(wPos*perf_AccW[0][2] + wNeg*perf_AccW[0][3])-1,), perf_AccW[1]+(2*(wPos*perf_AccW[1][2] + wNeg*perf_AccW[1][3])-1,)] # append adjusted Youden's

	#G performance weighting subjects by their previous Youden's index
	perf_YoudW = [performCalcWeights(estimatesTrain, truthsTrain, youdensTrain), performCalcWeights(estimatesValid, truthsValid, youdensValid)]
	perf_YoudW = [perf_YoudW[0]+(2*(wPos*perf_YoudW[0][2] + wNeg*perf_YoudW[0][3])-1,), perf_YoudW[1]+(2*(wPos*perf_YoudW[1][2] + wNeg*perf_YoudW[1][3])-1,)] # append adjusted Youden's

	#G performance weighting subjects by their previous adjusted Youden's
	perf_AdjW = [performCalcWeights(estimatesTrain, truthsTrain, adjYoudsTrain), performCalcWeights(estimatesValid, truthsValid, adjYoudsValid)]
	perf_AdjW = [perf_AdjW[0]+(2*(wPos*perf_AdjW[0][2] + wNeg*perf_AdjW[0][3])-1,), perf_AdjW[1]+(2*(wPos*perf_AdjW[1][2] + wNeg*perf_AdjW[1][3])-1,)] # append adjusted Youden's

	#G performance weighting subjects by their confidence
	perf_ConfW = [performCalcWeights(estimatesTrain, truthsTrain, confidencesTrain), performCalcWeights(estimatesValid, truthsValid, confidencesValid)]
	perf_ConfW = [perf_ConfW[0]+(2*(wPos*perf_ConfW[0][2] + wNeg*perf_ConfW[0][3])-1,), perf_ConfW[1]+(2*(wPos*perf_ConfW[1][2] + wNeg*perf_ConfW[1][3])-1,)] # append adjusted Youden's


        performanceDict = {
		'perf_Maj': perf_Maj,
		'perf_Opt': perf_Opt,
		'perf_Acc': perf_Acc,
		'perf_Youd': perf_Youd,
		'perf_Adj': perf_Adj,
		'perf_Conf': perf_Conf,
		'perf_AccW': perf_AccW,
		'perf_YoudW': perf_YoudW,
		'perf_AdjW': perf_AdjW,
		'perf_ConfW': perf_ConfW
		}

        return performanceDict




def performSplit(data, nums):
	'''
	Computes performance of every group
	'''

	train = data[0]
	trainT = data[1]
	valid = data[2]
	validT = data[3]
	test = data[4]
	testT = data[5]
	
	wPos = nums[4]
	wNeg = nums[5]

	estimatesTrain = [z[::6] for z in train[:]]
	estimatesValid = [z[::6] for z in valid[:]]

	truthsTrain = trainT
	truthsValid = validT

	accuraciesTrain = [z[2::6] for z in train[:]]
	accuraciesValid = [z[2::6] for z in valid[:]]

	youdensTrain = [z[3::6] for z in train[:]]
	youdensValid = [z[3::6] for z in valid[:]]

	sensitsTrain = [z[4::6] for z in train[:]]
	sensitsValid = [z[4::6] for z in valid[:]]

	specifsTrain = [z[5::6] for z in train[:]]
	specifsValid = [z[5::6] for z in valid[:]]

	adjYoudsTrain = [2*sum(i)-1 for i in zip([wPos * z for z in sensitsTrain],[wNeg * z for z in specifsTrain])] 
	adjYoudsValid = [2*sum(i)-1 for i in zip([wPos * z for z in sensitsValid],[wNeg * z for z in specifsValid])] 

	confidencesTrain = [z[1::6] for z in train[:]]
	confidencesValid = [z[1::6] for z in valid[:]]


	#numDoctors = nums[0]
	numTrain = nums[1]
	numValid = nums[2]
	numTest = nums[3]
	#numData = numTrain + numValid + numTest

	numDataTrain = train.shape[0]
	if numDataTrain != 0:
		numGroupsTrain = numDataTrain/numTrain
	else:
		numGroupsTrain = 0

	numDataValid = valid.shape[0]
	if numDataValid != 0:
		numGroupsValid = numDataValid/numValid
	else:
		numGroupsValid = 0

	numDataTest = test.shape[0]
	if numDataTest != 0:
		numGroupsTest = numDataTest/numTest
	else:
		numGroupsTest = 0


	#G the structure of the following lists will be:
	#G [[[group1: acc youd sens spec adj],[group2],...](in training), [[group1': acc youd sens spec adj],[group2'],...](in validation)]
	#G [training or valid][group number][acc, youd, sens, spec, adj]

	#G performance of the majority voting
	perf_Maj = [[performCalc(estimatesTrain[i*numTrain:(i+1)*numTrain], truthsTrain[i*numTrain:(i+1)*numTrain]) for i in xrange(numGroupsTrain)],[performCalc(estimatesValid[i*numValid:(i+1)*numValid], truthsValid[i*numValid:(i+1)*numValid]) for i in xrange(numGroupsValid)]]
	perf_Maj = [[perf_Maj[0][i]+(2*(wPos*perf_Maj[0][i][2] + wNeg*perf_Maj[0][i][3])-1,) for i in xrange(numGroupsTrain)], [perf_Maj[1][i]+(2*(wPos*perf_Maj[1][i][2] + wNeg*perf_Maj[1][i][3])-1,) for i in xrange(numGroupsValid)]] # append adjusted Youden's

	#G performance of the 'optimistic' strategy (in case of tie, declare as negative)
	perf_Opt = [[performCalcOptimistic(estimatesTrain[i*numTrain:(i+1)*numTrain], truthsTrain[i*numTrain:(i+1)*numTrain]) for i in xrange(numGroupsTrain)],[performCalcOptimistic(estimatesValid[i*numValid:(i+1)*numValid], truthsValid[i*numValid:(i+1)*numValid]) for i in xrange(numGroupsValid)]]
	perf_Opt = [[perf_Opt[0][i]+(2*(wPos*perf_Opt[0][i][2] + wNeg*perf_Opt[0][i][3])-1,) for i in xrange(numGroupsTrain)], [perf_Opt[1][i]+(2*(wPos*perf_Opt[1][i][2] + wNeg*perf_Opt[1][i][3])-1,) for i in xrange(numGroupsValid)]] # append adjusted Youden's

	#G performance of the 'choose the most accurate' strategy
	perf_Acc = [[performCalcHigher(estimatesTrain[i*numTrain:(i+1)*numTrain], truthsTrain[i*numTrain:(i+1)*numTrain], accuraciesTrain[i*numTrain:(i+1)*numTrain]) for i in xrange(numGroupsTrain)],[performCalcHigher(estimatesValid[i*numValid:(i+1)*numValid], truthsValid[i*numValid:(i+1)*numValid], accuraciesValid[i*numValid:(i+1)*numValid]) for i in xrange(numGroupsValid)]]
	perf_Acc = [[perf_Acc[0][i]+(2*(wPos*perf_Acc[0][i][2] + wNeg*perf_Acc[0][i][3])-1,) for i in xrange(numGroupsTrain)], [perf_Acc[1][i]+(2*(wPos*perf_Acc[1][i][2] + wNeg*perf_Acc[1][i][3])-1,) for i in xrange(numGroupsValid)]] # append adjusted Youden's

	#G performance of the 'choose the best Youden's index' strategy
	perf_Youd = [[performCalcHigher(estimatesTrain[i*numTrain:(i+1)*numTrain], truthsTrain[i*numTrain:(i+1)*numTrain], youdensTrain[i*numTrain:(i+1)*numTrain]) for i in xrange(numGroupsTrain)],[performCalcHigher(estimatesValid[i*numValid:(i+1)*numValid], truthsValid[i*numValid:(i+1)*numValid], youdensValid[i*numValid:(i+1)*numValid]) for i in xrange(numGroupsValid)]]
	perf_Youd = [[perf_Youd[0][i]+(2*(wPos*perf_Youd[0][i][2] + wNeg*perf_Youd[0][i][3])-1,) for i in xrange(numGroupsTrain)], [perf_Youd[1][i]+(2*(wPos*perf_Youd[1][i][2] + wNeg*perf_Youd[1][i][3])-1,) for i in xrange(numGroupsValid)]] # append adjusted Youden's

	#G performance of the 'choose the best adjusted Youden's' strategy
	perf_Adj = [[performCalcHigher(estimatesTrain[i*numTrain:(i+1)*numTrain], truthsTrain[i*numTrain:(i+1)*numTrain], adjYoudsTrain[i*numTrain:(i+1)*numTrain]) for i in xrange(numGroupsTrain)],[performCalcHigher(estimatesValid[i*numValid:(i+1)*numValid], truthsValid[i*numValid:(i+1)*numValid], adjYoudsValid[i*numValid:(i+1)*numValid]) for i in xrange(numGroupsValid)]]
	perf_Adj = [[perf_Adj[0][i]+(2*(wPos*perf_Adj[0][i][2] + wNeg*perf_Adj[0][i][3])-1,) for i in xrange(numGroupsTrain)], [perf_Adj[1][i]+(2*(wPos*perf_Adj[1][i][2] + wNeg*perf_Adj[1][i][3])-1,) for i in xrange(numGroupsValid)]] # append adjusted Youden's

	#G performance of the 'choose the most confident' strategy
	perf_Conf = [[performCalcHigher(estimatesTrain[i*numTrain:(i+1)*numTrain], truthsTrain[i*numTrain:(i+1)*numTrain], confidencesTrain[i*numTrain:(i+1)*numTrain]) for i in xrange(numGroupsTrain)],[performCalcHigher(estimatesValid[i*numValid:(i+1)*numValid], truthsValid[i*numValid:(i+1)*numValid], confidencesValid[i*numValid:(i+1)*numValid]) for i in xrange(numGroupsValid)]]
	perf_Conf = [[perf_Conf[0][i]+(2*(wPos*perf_Conf[0][i][2] + wNeg*perf_Conf[0][i][3])-1,) for i in xrange(numGroupsTrain)], [perf_Conf[1][i]+(2*(wPos*perf_Conf[1][i][2] + wNeg*perf_Conf[1][i][3])-1,) for i in xrange(numGroupsValid)]] # append adjusted Youden's

	#G performance weighting subjects by their previous accuracy
	perf_AccW = [[performCalcWeights(estimatesTrain[i*numTrain:(i+1)*numTrain], truthsTrain[i*numTrain:(i+1)*numTrain], accuraciesTrain[i*numTrain:(i+1)*numTrain]) for i in xrange(numGroupsTrain)],[performCalcWeights(estimatesValid[i*numValid:(i+1)*numValid], truthsValid[i*numValid:(i+1)*numValid], accuraciesValid[i*numValid:(i+1)*numValid]) for i in xrange(numGroupsValid)]]
	perf_AccW = [[perf_AccW[0][i]+(2*(wPos*perf_AccW[0][i][2] + wNeg*perf_AccW[0][i][3])-1,) for i in xrange(numGroupsTrain)], [perf_AccW[1][i]+(2*(wPos*perf_AccW[1][i][2] + wNeg*perf_AccW[1][i][3])-1,) for i in xrange(numGroupsValid)]] # append adjusted Youden's

	#G performance weighting subjects by their previous Youden's index
	perf_YoudW = [[performCalcWeights(estimatesTrain[i*numTrain:(i+1)*numTrain], truthsTrain[i*numTrain:(i+1)*numTrain], youdensTrain[i*numTrain:(i+1)*numTrain]) for i in xrange(numGroupsTrain)],[performCalcWeights(estimatesValid[i*numValid:(i+1)*numValid], truthsValid[i*numValid:(i+1)*numValid], youdensValid[i*numValid:(i+1)*numValid]) for i in xrange(numGroupsValid)]]
	perf_YoudW = [[perf_YoudW[0][i]+(2*(wPos*perf_YoudW[0][i][2] + wNeg*perf_YoudW[0][i][3])-1,) for i in xrange(numGroupsTrain)], [perf_YoudW[1][i]+(2*(wPos*perf_YoudW[1][i][2] + wNeg*perf_YoudW[1][i][3])-1,) for i in xrange(numGroupsValid)]] # append adjusted Youden's

	#G performance weighting subjects by their previous adjusted Youden's
	perf_AdjW = [[performCalcWeights(estimatesTrain[i*numTrain:(i+1)*numTrain], truthsTrain[i*numTrain:(i+1)*numTrain], adjYoudsTrain[i*numTrain:(i+1)*numTrain]) for i in xrange(numGroupsTrain)],[performCalcWeights(estimatesValid[i*numValid:(i+1)*numValid], truthsValid[i*numValid:(i+1)*numValid], adjYoudsValid[i*numValid:(i+1)*numValid]) for i in xrange(numGroupsValid)]]
	perf_AdjW = [[perf_AdjW[0][i]+(2*(wPos*perf_AdjW[0][i][2] + wNeg*perf_AdjW[0][i][3])-1,) for i in xrange(numGroupsTrain)], [perf_AdjW[1][i]+(2*(wPos*perf_AdjW[1][i][2] + wNeg*perf_AdjW[1][i][3])-1,) for i in xrange(numGroupsValid)]] # append adjusted Youden's

	#G performance weighting subjects by their confidence
	perf_ConfW = [[performCalcWeights(estimatesTrain[i*numTrain:(i+1)*numTrain], truthsTrain[i*numTrain:(i+1)*numTrain], confidencesTrain[i*numTrain:(i+1)*numTrain]) for i in xrange(numGroupsTrain)],[performCalcWeights(estimatesValid[i*numValid:(i+1)*numValid], truthsValid[i*numValid:(i+1)*numValid], confidencesValid[i*numValid:(i+1)*numValid]) for i in xrange(numGroupsValid)]]
	perf_ConfW = [[perf_ConfW[0][i]+(2*(wPos*perf_ConfW[0][i][2] + wNeg*perf_ConfW[0][i][3])-1,) for i in xrange(numGroupsTrain)], [perf_ConfW[1][i]+(2*(wPos*perf_ConfW[1][i][2] + wNeg*perf_ConfW[1][i][3])-1,) for i in xrange(numGroupsValid)]] # append adjusted Youden's


        performanceDict = {
		'perf_Maj': perf_Maj,
		'perf_Opt': perf_Opt,
		'perf_Acc': perf_Acc,
		'perf_Youd': perf_Youd,
		'perf_Adj': perf_Adj,
		'perf_Conf': perf_Conf,
		'perf_AccW': perf_AccW,
		'perf_YoudW': perf_YoudW,
		'perf_AdjW': perf_AdjW,
		'perf_ConfW': perf_ConfW
		}

        return performanceDict





