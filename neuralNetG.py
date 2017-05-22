'''
Here I'm going to train with all the groups, but with the possibility of using only a part on each epoch,
and then minibatch this part.
Also, I compute accuracy over the whole combinations both for train and validation
'''

import csv
import sys
import os
import cPickle as pickle
import numpy as np
import math
from matplotlib import pyplot as plt
import itertools
from imp import reload
import pandas as pd
import performFunctions as pfu
reload(pfu)
import plotters as plo
reload(plo)
import smoothers as smo
sys.path.append('../Gabriel_Project/Model')
sys.path.append('../Gabriel_Project/utils')
from loadData import loadData
#G from tf_utils import dense_to_one_hot
from docsStatistics import ROCStatistics
import tensorflow as tf




def netfunctionG5(N1, N2, plot_fig = 'yes', return_val = 'no'):
	#G N1=40 #size of first hidden layer
	#G N2=40 #size of second hidden layer

	mW1=0.00; sW1=0.5; mb1=0.00; sb1=0.00; #G hyperparameters of first layer
	mW2=0.00; sW2=0.5; mb2=0.00; sb2=0.00; #G hyperparameters of second layer
	mWo=0.00; sWo=0.5; mbo=0.00; sbo=0.00; #G hyperparameters of out layer

	groupSize = 5
	maxBatches = 250 #G maximum number of batches used on each epoch (if there's enough data)
			#G for groups of 2, 40 doc,54+54 cases, batches of 256: 164 is the maximum number
			#G for groups of 3, 40 ,54+54, 256: 2084
			#G for groups of 5, 40 ,54+54, 256: 138798
	propCombin = 0.001 #G maximum number of combinations used for analysis, to avoid combinatorial explosion.
			#G grs of 2: C(40,2)=780, *54=42120;
			#G grs of 3: C(40,3)=9880, *54=533520;
			#G grs of 5: C(40,5)=658008, *54=35532432;
	numEpochs=3000 #G number of epochs
	nSkip=5 #G frequency of plotting points
	learnRate = 0.00001
	randCases = 'no' #G 'yes': randomize cases; 'no', don't
	randDoctors = 'no' #G 'no'; 'same', same randomization for all cases; 'each', different randomization for each case
	#accMeasure = 2 #G Competence to define who is 'best'. 2: accuracy; 3: Youden's index #G Not used anymore
	#acm = accMeasure #G to make the name shorter ;-) #G Not used anymore

	if not os.path.exists('./data'):
        	os.makedirs('./data')
	if not os.path.exists('./data/partitionFile.csv'):
    		file('./data/partitionFile.csv', 'w').close()
	if not os.path.exists('./data/partitionFileNoHeaders.csv'):
    		file('./data/partitionFileNoHeaders.csv', 'w').close()

	ctime = os.stat('./data/partitionFile.csv').st_ctime
	ctime2 = os.stat('./data/partitionFileNoHeaders.csv').st_ctime

	#G randomize the original data from Krause's PNAS
	
	dFP = {
	'original': 'pnas.1601827113.sd01',
	'current': 'partitionFile',
	'001': 'partitionFile001',
	'4docs6cases': 'partitionFile4docs6cases',
	'goodYouden01': 'goodYouden01',
	'badValidation01': 'badValidation01',
	'badYoudenAndLoss01': 'badYoudenAndLoss01',
	'forLabMeeting001': 'forLabMeeting001',
	'forLabMeeting002': 'forLabMeeting002',
	'forPaper001': 'forPaper001',
	'horrible': 'horrible'
	}

	dataFilePre = dFP['current']  #G work from PNAS file ('original'), or from last randomization ('current'), and so on

	partitionFileCreator(dataFilePre, randCases)

	#G wait until new data files are stored
	while True:
		if os.stat('./data/partitionFile.csv').st_ctime > ctime and os.stat('./data/partitionFileNoHeaders.csv').st_ctime > ctime2:
			break
	#time.sleep(1.0)

	#G proportion of cases used for train, validation and test
	propTrain = 0.5
	propVal = 0.5
	propTest = 0

	# splits data into train, validation, and test cases, and computes some statistics on the training cases
	dataFile = 'partitionFile'
	dataSplit = loadData(propTrain, propVal, propTest, dataFile)
	Y_train = dataSplit[0]
	X_train = dataSplit[1]
	Y_valid = dataSplit[2]
	X_valid = dataSplit[3]
	Y_test = dataSplit[4]
	X_test = dataSplit[5]
	statistics = ROCStatistics(X_train, Y_train)
	#print np.sort(statistics['youden'])
	
	numPosTrain = np.sum([Y_train[i,1] for i in xrange(Y_train.shape[0])])
	numNegTrain = np.sum([Y_train[i,0] for i in xrange(Y_train.shape[0])])
	posToNegTrain = numPosTrain / numNegTrain #G number of positives to number of negatives ratio in training

	#importance = posToNegTrain; #G if we are interested only in accuracy
	importance = 3.0; #G if we want to give the same importance to a positive error than to a negative error (i.e. optimize Youden's index)
	#importance = 2.0; #G the times worse a false negative than a false positive is
	
	lam = importance / posToNegTrain; #G weight for positive errors in the loss function. lam in interval [0,Inf) (to use only with cross entropy)
	wPosTrain = importance / (importance+1); #G weight for sensitivity when determining who is 'best'
	wNegTrain = 1 - wPosTrain; #G weight for specificity when determining who is 'best'
	'''
	Not sure wether I'll use this or not
	numPosValid = np.sum([Y_valid[i,1] for i in xrange(Y_valid.shape[0])])
	numNegValid = np.sum([Y_valid[i,0] for i in xrange(Y_valid.shape[0])])
	posToNegValid = numPosValid / numNegValid #G number of positives to number of negatives ratio in validation

	wPosValid = importance / (importance+1); #G weight for sensitivity when determining who is 'best'
	wNegValid = 1 - wPosValid; #G weight for specificity when determining who is 'best'
	'''
	numDoctors = X_train.shape[1]
	numTrain = Y_train.shape[0]
	numValid = Y_valid.shape[0]
	numTest = Y_test.shape[0]
	nums = [numDoctors, numTrain, numValid, numTest, wPosTrain, wNegTrain]

	dataFileNH = 'partitionFileNoHeaders'

	crowd = netDataCreator(numDoctors, nums, statistics, dataFileNH, randDoctors = 'no') #G one single group (the crowd),
	crowdDict = pfu.performance(crowd, nums) 							#G that is why 1st argument is numDoctors


	netData = combNetDataCreator(groupSize, propCombin, nums, statistics, dataFileNH)
	#groupsDict = pfu.performance(netData)
	groupsDict = pfu.performSplit(netData, nums)

	train = netData[0]
	trainT = netData[1]
	valid = netData[2]
	validT = netData[3]
	test = netData[4]
	testT = netData[5]

	#G define what data features to use.
	#G remainder: data structure is [opi1 conf1 acc1 youd1 sens1 spec1  opi2 conf2 acc2 youd2 sens2 spec2 ...  opiN confN accN youdN sensN specN]
	
	sensTrain = train[:,4::6] #G sensitivity in training
	specTrain = train[:,5::6] #G specificity in training
	compTrain = [[2*(wPosTrain*sensTrain[i][j]+wNegTrain*specTrain[i][j])-1 for j in xrange(len(sensTrain[i]))] for i in xrange(len(sensTrain))] #G competence in training

	sensValid = valid[:,4::6] #G sensitivity in training (because we only use training performance as input)
	specValid = valid[:,5::6] #G specificity in training (")
	compValid = [[2*(wPosTrain*sensValid[i][j]+wNegTrain*specValid[i][j])-1 for j in xrange(len(sensValid[i]))] for i in xrange(len(sensValid))] #G competence in training (")

	sensTest = test[:,4::6] #G sensitivity in training (")
	specTest = test[:,5::6] #G specificity in training (")
	compTest = [[2*(wPosTrain*sensTest[i][j]+wNegTrain*specTest[i][j])-1 for j in xrange(len(sensTest[i]))] for i in xrange(len(sensTest))] #G competence in training (")
	'''
	dataComp = [train, trainT, compTrain, valid, validT, compValid, test, testT, compTest]
	pickle.dump( dataComp , open( 'dataComp.pkl', 'wb' ) )
	'''
	nconf = 4.0 #G confidence (originally in a 1 to 4 scale), if changed here, change accordingly in analyzers.py (same name for the variable)

	a1 = np.multiply(train[:,0::6],compTrain) #G opinion * competence
	b1 = train[:,1::6]/nconf  
	#a1 = np.multiply(train[:,0::6],1) #G opinion, competence not considered
	#b1 = 0*train[:,1::6] #G confidence not considered
	xTrain = np.append(a1,b1,axis=1)
	#G c1 = compTrain #G competence
	#G xTrain = np.append(xTrain,c1,axis=1) #G appends competence for further calculation of performance of best doctor

	a2 = np.multiply(valid[:,0::6],compValid) #G opinion * competence IN THE TRAINING
	b2 = valid[:,1::6]/nconf
	#a2 = np.multiply(valid[:,0::6],1)
	#b2 = 0*valid[:,1::6]
	xValid = np.append(a2,b2,axis=1)
	#G c2 = compValid
	#G xValid = np.append(xValid,c2,axis=1)
	
	if len(test)>0:
		a3 = np.multiply(test[:,0::6],compTest) #G opinion * competence IN THE TRAINING
		b3 = test[:,1::6]/nconf
		xTest = np.append(a3,b3,axis=1)
		#G c3 = compTest
		#G xTest = np.append(xTest,c3,axis=1)
	
	yTrain = trainT
	yValid = validT
	yTest = testT


	#G Din=2*xTrain.shape[1]/3 #G multiplied by 2/3 because one column (competence) will not be a feature
	Din=xTrain.shape[1]
	Dout=2 #G two categories

	#define placeholders for input data and target values
	x=tf.placeholder(tf.float32,shape=[None,Din])
	y_=tf.placeholder(tf.float32, shape=[None,Dout]) #G truths

	#G Number of neurons on first (N1) and second (N2) layer defined in the function calling
	#G Initialization parameters defined above (mean and std of weights and biases)

	if N1 > 1 and N2 > 1:
		with tf.name_scope('hidden1'):
			W1=tf.Variable(tf.truncated_normal([Din,N1],mean=mW1,stddev=sW1))
			b1=tf.Variable(tf.truncated_normal([N1],mean=mb1,stddev=sb1)) #G init. w. bias to avoid "dead neurons"?
			hidden1=tf.nn.relu6(tf.matmul(x, W1) + b1)

		with tf.name_scope('hidden2'):
			W2=tf.Variable(tf.truncated_normal([N1,N2],mean=mW2,stddev=sW2))
			b2=tf.Variable(tf.truncated_normal([N2],mean=mb2,stddev=sb2)) #G init. w. bias to avoid "dead neurons"?
			hidden2=tf.nn.relu6(tf.matmul(hidden1, W2) + b2)

		W3=tf.Variable(tf.truncated_normal([N2,Dout],mean=mWo,stddev=sWo))
		b3=tf.Variable(tf.truncated_normal([Dout],mean=mbo,stddev=sbo)) #G init. with bias to avoid "dead neurons"?
		#define the computation
		y = tf.matmul(hidden2, W3) + b3 #G y: evidence of the case belonging to a category. Then softmax to obtain probabilities

	if N1 > 1 and N2 <= 1:
		with tf.name_scope('hidden1'):
			W1=tf.Variable(tf.truncated_normal([Din,N1],mean=mW1,stddev=sW1))
			b1=tf.Variable(tf.truncated_normal([N1],mean=mb1,stddev=sb1))
			hidden1=tf.nn.relu6(tf.matmul(x, W1) + b1)

		W2=tf.Variable(tf.truncated_normal([N1,Dout],mean=mWo,stddev=sWo))
		b2=tf.Variable(tf.truncated_normal([Dout],mean=mbo,stddev=sbo))
		#define the computation
		y = tf.matmul(hidden1, W2) + b2 #G y: evidence of the case belonging to a category. Then softmax to obtain probabilities

	if N1 <= 1 and N2 > 1:
		with tf.name_scope('hidden2'):
			W1=tf.Variable(tf.truncated_normal([Din,N2],mean=mW2,stddev=sW2))
			b1=tf.Variable(tf.truncated_normal([N2],mean=mb2,stddev=sb2))
			hidden2=tf.nn.relu6(tf.matmul(x, W1) + b1)

		W2=tf.Variable(tf.truncated_normal([N2,Dout],mean=mWo,stddev=sWo))
		b2=tf.Variable(tf.truncated_normal([Dout],mean=mbo,stddev=sbo))
		#define the computation
		y = tf.matmul(hidden2, W2) + b2 #G y: evidence of the case belonging to a category. Then softmax to obtain probabilities

	if N1 <= 1 and N2 <= 1:
		W1=tf.Variable(tf.truncated_normal([Din,Dout],mean=mWo,stddev=sWo))
		b1=tf.Variable(tf.truncated_normal([Dout],mean=mbo,stddev=sbo))
		#define the computation
		y = tf.matmul(x, W1) + b1 #G y: evidence of the case belonging to a category. Then softmax to obtain probabilities

	#the loss function
	#loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_)) #G (logits, truths)
	#loss_function = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
	#G loss function, but weighting positive errors relative to negative errors (1 is the same importance for both)
	#loss_function = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y, y_,0.25)) #G (logits, truths)
	#loss_function = tf.reduce_mean(wNegTrain * tf.square(y - y_) + wPosTrain * tf.square(y - y_)) #G (logits, truths)
	
	casNegPred = tf.squeeze(tf.gather(tf.nn.softmax(y),tf.where(tf.equal(tf.argmax(y_,1),0))))
	casNegTruth = tf.squeeze(tf.gather(y_,tf.where(tf.equal(tf.argmax(y_,1),0))))
	casPosPred = tf.squeeze(tf.gather(tf.nn.softmax(y),tf.where(tf.equal(tf.argmax(y_,1),1))))
	casPosTruth = tf.squeeze(tf.gather(y_,tf.where(tf.equal(tf.argmax(y_,1),1))))
	#loss_function = wNegTrain * tf.reduce_mean(tf.abs(tf.squeeze(casNegPred)[:,0] - tf.squeeze(casNegTruth)[:,0])) + wPosTrain * tf.reduce_mean(tf.abs(tf.squeeze(casPosPred)[:,0] - tf.squeeze(casPosTruth)[:,0])) #G (logits, truths)
	loss_function = 1 - wNegTrain * tf.reduce_mean(casNegPred[:,0]) - wPosTrain * tf.reduce_mean(casPosPred[:,1]) #G (logits, truths)	
	#optimizer
	#train_step=tf.train.GradientDescentOptimizer(learnRate).minimize(loss_function)
	
	# test functions
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) #G logical vector with correct (1) or wrong (0)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        positives = tf.cast(tf.equal(tf.argmax(y_, 1), 1), tf.float32) #G 1 positive, 0 negative
	negatives = tf.cast(tf.equal(tf.argmax(y_, 1), 0), tf.float32) #G 1 negative, 0 positive
	prediction = tf.cast(tf.argmax(y, 1), tf.float32) #G [1 0]: negative, [0 1]: positive, ergo 0 is negative and 1 positive

	#ssenss = tf.reduce_mean(tf.cast(tf.gather(correct_prediction,tf.where(tf.equal(positives,1))),tf.float32))
	#ssenss = tf.reduce_mean(tf.gather(positives,tf.where(tf.equal(positives,1)))-tf.gather(prediction,tf.where(tf.equal(positives,1))))

	#ssens = tf.gather(prediction,tf.where(tf.equal(positives,1)))
	#ssenss = tf.reduce_mean(1.0-ssens)
	#sspecc = tf.reduce_mean(tf.gather(prediction,tf.where(tf.equal(negatives,1))))
	
	#loss_function = wNegTrain*sspecc + wPosTrain*ssenss
	
	train_step=tf.train.AdamOptimizer(learnRate).minimize(loss_function)

	saver = tf.train.Saver()

	#start interactive session
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	#the training loop
	#G Defined above numEpochs=20 #G number of epochs
	#G Defined above nSkip=10 #G frequency of plotting points
	netLoss = np.zeros([2,numEpochs/nSkip]) #G loss functions of training and validation
	netPerf = np.zeros([2,5,numEpochs/nSkip]) #G performance (acc, youd, sens, spec, adjacc) of net in training and validation

	if yTrain.shape[0]>=256*4:
		batchSize=256 #G minibatch size
	elif yTrain.shape[0]>=256*2:
		batchSize=128
	else:
		batchSize=yTrain.shape[0]

	if maxBatches > int(np.floor(yTrain.shape[0]/batchSize)):
		numBatches = int(np.floor(yTrain.shape[0]/batchSize))
	else:
		numBatches = maxBatches
	print 'batches per epoch', numBatches
	z=0
	prInter=200 #G plotting interval
	for i in range(numEpochs):
		if (i+1)%prInter==0:
			print "{0:.3e}".format(i+1),'epochs' 
		batch = makeBatch(xTrain,yTrain,batchSize*numBatches); #G xTrain: [[opi1*comp1 opi2*comp2 ... conf1 conf2 ... comp1 comp2 ...]["]...]
		         						#G ytrain: [[1 0][1 0][0 1] ...] = [neg neg pos ...]
		for j in range(numBatches):
			miniBatch = [batch[k][j*batchSize:(j+1)*batchSize] for k in range(len(batch))]
			xB = miniBatch[0]
			xBatch = xB[:,:Din]
			train_step.run(feed_dict={x: xBatch, y_: miniBatch[1]})

		if i%nSkip==0:
			'''
			netLoss[0,z]=sess.run(loss_function, feed_dict={x: xTrain[:,:Din],y_:yTrain}) #G y_ = truths
			netLoss[1,z]=sess.run(loss_function, feed_dict={x: xValid[:,:Din],y_:yValid})
			'''
			#G competence of the net with the training cases:
			
			netLoss[0,z], netPerf[0,0,z], realPos, realNeg, pred = sess.run([loss_function, accuracy, positives, negatives, prediction], feed_dict={x: xTrain[:,:Din], y_: yTrain})
			'''
			if i == 100:
				print wei1
			'''
			netTruePos = np.sum(a and b for a, b in zip(realPos, pred))
			netTrueNeg = np.sum(a and b for a, b in zip(realNeg, 1-pred))

			netPerf[0,2,z] = netTruePos / np.sum(realPos == 1) #G sensitivity			
			netPerf[0,3,z] = netTrueNeg / np.sum(realNeg == 1) #G specificity
			netPerf[0,1,z] = netPerf[0,2,z] + netPerf[0,3,z] - 1 #G Youden's index
			netPerf[0,4,z] = wPosTrain * netPerf[0,2,z] + wNegTrain * netPerf[0,3,z] #G adjusted accuracy

			#G competence of the net with the validation cases:

			netLoss[1,z], netPerf[1,0,z], realPosV, realNegV, predV = sess.run([loss_function, accuracy, positives, negatives, prediction], feed_dict={x: xValid[:,:Din], y_: yValid})

			netTruePosV = np.sum(a and b for a, b in zip(realPosV, predV))
			netTrueNegV = np.sum(a and b for a, b in zip(realNegV, 1-predV))

			netPerf[1,2,z] = netTruePosV / np.sum(realPosV == 1) #G sensitivity
			netPerf[1,3,z] = netTrueNegV / np.sum(realNegV == 1) #G specificity
			netPerf[1,1,z] = netPerf[1,2,z] + netPerf[1,3,z] - 1 #G Youden's index
			netPerf[1,4,z] = wPosTrain * netPerf[1,2,z] + wNegTrain * netPerf[1,3,z] #G adjusted accuracy
			
			networkDict = {
				'netLoss': netLoss[:,:z],
				'netPerf': netPerf[:,:z]
				}

			scoresDict = networkDict
			scoresDict.update(groupsDict)

			z+=1

		if (i > 0 and (i+1)%prInter==0) or (numEpochs < prInter and i == numEpochs-1):
			
			pickle.dump( scoresDict , open( 'toPlot.pkl', 'wb' ) )

			we1, bi1, we2, bi2, we3, bi3 = sess.run([W1, b1, W2, b2, W3, b3], feed_dict={x: xTrain[:,:Din], y_: yTrain})
			weightsAndBias = [we1, bi1, we2, bi2, we3, bi3] 
			pickle.dump( weightsAndBias , open( 'weightsAndBias' + str(i+1) + '.pkl', 'wb' ) ) #G At some point this should be moved so we take weights at various epochs and select the bests
			netPred = [pred, predV]
			#pickle.dump( netPred , open( 'netPred' + str(i+1) + '.pkl', 'wb' ) ) #G At some point this should be moved so we take predictions at various epochs and select the bests
		
		if plot_fig == 'yes' and ( (i > 0 and (i+1)%prInter==0) or (numEpochs < prInter and i == numEpochs-1) ):
			plo.oneIterPlotter(scoresDict)
			#save_path = saver.save(sess, "./data/model.ckpt")
			#saver.restore(sess, "/tmp/model.ckpt")
			if i < numEpochs-1:
				plt.close()
	
	if return_val == 'yes':
		return scoresDict, crowdDict



def netfunctionLoopG5(N1,N2,Nloop):

	# N1 size of first hidden layer
	# N2 size of second hidden layer
	# Nloop number of realizations of the training

	#G performance is: accuracy, Youden's index., sensitivity, specificity adn adjusted accuracy
	#G [training o validation][iteration][acc, youd, sens, spec, adj]
	netPerf = np.zeros([2,Nloop,5]) #G performance in training and validation
	perf_Maj = np.zeros([2,Nloop,5]) # performance of majority voting in training and validation
	perf_Opt = np.zeros([2,Nloop,5]) # performance of choosing the optimistic strategy in training and validation
	perf_Acc = np.zeros([2,Nloop,5]) # performance of most competent (best accuracy) in training and validation
	perf_Youd = np.zeros([2,Nloop,5]) # performance of most competent (best Youden's index) in training and validation
	perf_Adj = np.zeros([2,Nloop,5]) # performance of most competent (best adjusted accuracy) in training and validation
	perf_Conf = np.zeros([2,Nloop,5]) # performance of chosing the most confident in training and validation
	perf_AccW = np.zeros([2,Nloop,5]) # performance weighting by accuracy in training and validation
	perf_YoudW = np.zeros([2,Nloop,5]) # performance weighting by Youden's index in training and validation
	perf_AdjW = np.zeros([2,Nloop,5]) # performance weighting by adjusted accuracy in training and validation
	perf_ConfW = np.zeros([2,Nloop,5]) # performance weighting by confidence in training and validation
	woc = np.zeros([2,Nloop,5]) # performance of woc in training and validation

	for q in range(0,Nloop):
		print 'iteration', q+1

		scoresDict, crowdDict = netfunctionG5(N1, N2, plot_fig='no', return_val='yes')
		pickle.dump( scoresDict , open( './data/iter' + str(q+79) + '.pkl', 'wb' ) ) # writes data to document
		#G load data with: scoresDict = pickle.load(open( './data/iter' + str(q+1) + '.pkl', 'rb' ) ) # loads data from document

		netLoss = scoresDict['netLoss']
		smooLoss = smo.movingWindow(netLoss[1],(len(netLoss[1])+1)/4)
		pos = np.argmin(smooLoss)
		netPerf[:,q,:] = [scoresDict['netPerf'][0,:,pos], scoresDict['netPerf'][1,:,pos]]
		
		pM = scoresDict['perf_Maj']
		perf_Maj[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		pM = scoresDict['perf_Opt']
		perf_Opt[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		pM = scoresDict['perf_Acc']
		perf_Acc[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		pM = scoresDict['perf_Youd']
		perf_Youd[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		pM = scoresDict['perf_Adj']
		perf_Adj[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		pM = scoresDict['perf_Conf']
		perf_Conf[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		pM = scoresDict['perf_AccW']
		perf_AccW[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		pM = scoresDict['perf_YoudW']
		perf_YoudW[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		pM = scoresDict['perf_AdjW']
		perf_AdjW[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		pM = scoresDict['perf_ConfW']
		perf_ConfW[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		
		woc[:,q,:] = [crowdDict['perf_Maj'][0],crowdDict['perf_Maj'][1]]
	
	loopDict= {
		'netPerf': netPerf,
		'perf_Maj': perf_Maj,
		'perf_Opt': perf_Opt,
		'perf_Acc': perf_Acc,
		'perf_Youd':perf_Youd,
		'perf_Adj':perf_Adj,
		'perf_Conf': perf_Conf,
		'perf_AccW': perf_AccW,
		'perf_YoudW': perf_YoudW,
		'perf_AdjW': perf_AdjW,
		'perf_ConfW': perf_ConfW,
		'woc': woc
	}

	plo.loopPlotter(loopDict)




def partitionFileCreator(dataFilePre, randCases): #G randCases 'yes', randomizes cases; prev. version: randDoctors 'yes', randomizes doctors
	'''Creates two files named partitionFile and partitionFileNoHeaders, with the order of the cases randomized, so each time
	the data is loaded the training, validation and test (if there are) cases are shuffled
	The document is stored in the 'data' subfolder.
	'''

	dfp = pd.read_csv('./data/' + dataFilePre + '.csv')

	labels = ['diagnostician', 'case', 'melanoma', 'decision', 'confidence']
	''' The structure of the document is:
	      diagnostician     case   melanoma    decision    confidence
		    1             1        1           1            2
		    1             2        1           0            3
		    1             3        0           0            3
		    2             1        1           1            4
		    2             2        1           0            1
		    2             3        0           1            1
	0: negative
	1: positive
	confidence rated in a 1 to 4 scale
	'''

	cases = dfp['case'].unique() # finds how many different images were diagnosed
    	numCases = len(cases)
	if randCases == 'yes':
		permCases = np.random.permutation(range(numCases)) #G randomizes cases
	else:
		permCases = np.array(range(numCases)) #G cases not randomized

	doctors = dfp['diagnostician'].unique() # finds how many different doctors
    	numDoctors = len(doctors)
	#if randDoctors == 'yes':
	#	permDoctors = np.random.permutation(range(numDoctors)) #G randomizes doctors
	#else:
	#	permDoctors = np.array(range(numDoctors)) #G doctors not randomized


	doctColumnPre = dfp['diagnostician'] # 1st column: Doctor number
	caseColumnPre = dfp['case'] # 2nd column: Case number
	truthColumnPre = dfp['melanoma'] # 3rd column: Truths
	decisColumnPre = dfp['decision'] # 4th column: Decisions
	confColumnPre = dfp['confidence'] # 5th column: Confidences

	doctColumn = doctColumnPre
	caseColumn = caseColumnPre
	truthColumn = np.zeros(len(truthColumnPre))
	decisColumn = np.zeros(len(decisColumnPre))
	confColumn = np.zeros(len(confColumnPre))

	for i in range(numDoctors):
		#caseColumn[i*numCases:(i+1)*numCases] = [caseColumnPre[i*numCases+j for j in permCases]
		truthColumn[i*numCases:(i+1)*numCases] = [truthColumnPre[i*numCases+j] for j in permCases]
		decisColumn[i*numCases:(i+1)*numCases] = [decisColumnPre[i*numCases+j] for j in permCases]
		confColumn[i*numCases:(i+1)*numCases] = [confColumnPre[i*numCases+j] for j in permCases]

	dataDict = {labels[0]: doctColumn.astype(int),
	labels[1]: caseColumn.astype(int),
	labels[2]: truthColumn.astype(int),
	labels[3]: decisColumn.astype(int),
	labels[4]: confColumn.astype(int)
	}

	df = pd.DataFrame(dataDict, columns = labels)
	df.to_csv('./data/partitionFile.csv', index = False)

	dfNH = pd.DataFrame(dataDict, columns = labels)
	dfNH.to_csv('./data/partitionFileNoHeaders.csv', header = False, index = False)




def netDataCreator(groupSize, nums, statistics, dataFile, randDoctors): # la de no headers partitionFileNoHeaders

	numDoctors = nums[0]
	numTrain = nums[1]
	numValid = nums[2]
	numTest = nums[3]
	numCases = numTrain + numValid + numTest

	#reading in the data from a csv file
	csvfile=open('./data/' + dataFile + '.csv','rb')
	dfile=csv.reader(csvfile)

	Ndata = numDoctors * numCases #number of case-opinion pairs
	data=np.zeros([Ndata,5])
	j=0
	for row in dfile: #G moves across rows
		i=0;
		for x in row: #G moves through the row
			data[j,i]=int(row[i])
			i+=1
		j+=1


	#make the matrix of data using groups which will be used for training the net
	numGroups = int(np.floor(float(numDoctors)/groupSize)) #G ceil: one last smaller group, floor: no final smaller group

	netDataTrain = np.zeros([numGroups*numTrain,6*groupSize])
	truthDummyTrain = np.zeros(numGroups*numTrain)
	netDataValid = np.zeros([numGroups*numValid,6*groupSize])
	truthDummyValid = np.zeros(numGroups*numValid)
	netDataTest = np.zeros([numGroups*numTest,6*groupSize])
	truthDummyTest = np.zeros(numGroups*numTest)

	#G randDoctors: 'no'; 'same', same randomization for all cases; 'each', different randomization for each case
	if randDoctors == 'same':
		permDoctors = np.random.permutation(range(numDoctors)) #G randomizes doctors
	elif randDoctors == 'no':
		permDoctors = np.array(range(numDoctors)) #G doctors not randomized

	#G remainder: data[ index, case, truth, opinion, confidence ]
	for i in range(numTrain):

		truthDummyTrain[i*numGroups:(i+1)*numGroups] = data[i,2]

		if randDoctors == 'each':
			permDoctors = np.random.permutation(range(numDoctors)) #G randomizes doctors

		acc = statistics['indAcc'][permDoctors]/100; #G accuracy
		youd = statistics['youden'][permDoctors]; #G Youden's index		
		sens = statistics['sensit']; #G sensitivity
		spec = statistics['specif']; #G specificity

		for j in range(numGroups):

			Data1 = np.array([data[k*numCases+i,:] for k in permDoctors[j*groupSize:(j+1)*groupSize]])
			#convert from 0,1 to (-1,1)
			Data1[:,3]=2*(Data1[:,3]-0.5)
			Data1=Data1[:,3:] #G store only opinion and confidence
			Data1 = np.concatenate((Data1,np.array([acc[permDoctors[j*groupSize:(j+1)*groupSize]]]).T),axis=1) #G adds accuracy
			Data1 = np.concatenate((Data1,np.array([youd[permDoctors[j*groupSize:(j+1)*groupSize]]]).T),axis=1) #G adds Youdens
			Data1 = np.concatenate((Data1,np.array([sens[permDoctors[j*groupSize:(j+1)*groupSize]]]).T),axis=1) #G adds sensitivity
			Data1 = np.concatenate((Data1,np.array([spec[permDoctors[j*groupSize:(j+1)*groupSize]]]).T),axis=1) #G adds specificity

			netDataTrain[i*numGroups+j,:] = [k for l in Data1 for k in l]

	for i in range(numValid):

		truthDummyValid[i*numGroups:(i+1)*numGroups] = data[numTrain+i,2]

		if randDoctors == 'each':
			permDoctors = np.random.permutation(range(numDoctors)) #G randomizes doctors

		acc = statistics['indAcc'][permDoctors]/100; #G accuracy
		youd = statistics['youden'][permDoctors]; #G Youden's index		
		sens = statistics['sensit']; #G sensitivity
		spec = statistics['specif']; #G specificity

		for j in range(numGroups):

			Data1 = np.array([data[k*numCases+numTrain+i,:] for k in permDoctors[j*groupSize:(j+1)*groupSize]])
			#convert from 0,1 to (-1,1)
			Data1[:,3]=2*(Data1[:,3]-0.5)
			Data1=Data1[:,3:] #G store only opinion and confidence
			Data1 = np.concatenate((Data1,np.array([acc[permDoctors[j*groupSize:(j+1)*groupSize]]]).T),axis=1) #G adds accuracy
			Data1 = np.concatenate((Data1,np.array([youd[permDoctors[j*groupSize:(j+1)*groupSize]]]).T),axis=1) #G adds Youdens
			Data1 = np.concatenate((Data1,np.array([sens[permDoctors[j*groupSize:(j+1)*groupSize]]]).T),axis=1) #G adds sensitivity
			Data1 = np.concatenate((Data1,np.array([spec[permDoctors[j*groupSize:(j+1)*groupSize]]]).T),axis=1) #G adds specificity

			netDataValid[i*numGroups+j,:] = [k for l in Data1 for k in l]

	for i in range(numTest):

		truthDummyTest[i*numGroups:(i+1)*numGroups] = data[numTrain+numValid+i,2]

		if randDoctors == 'each':
			permDoctors = np.random.permutation(range(numDoctors)) #G randomizes doctors

		acc = statistics['indAcc'][permDoctors]/100; #G accuracy
		youd = statistics['youden'][permDoctors]; #G Youden's index
		sens = statistics['sensit']; #G sensitivity
		spec = statistics['specif']; #G specificity

		for j in range(numGroups):

			Data1 = np.array([data[k*numCases+numTrain+numValid+i,:] for k in permDoctors[j*groupSize:(j+1)*groupSize]])
			#convert from 0,1 to (-1,1)
			Data1[:,3]=2*(Data1[:,3]-0.5)
			Data1=Data1[:,3:] #G store only opinion and confidence
			Data1 = np.concatenate((Data1,np.array([acc[permDoctors[j*groupSize:(j+1)*groupSize]]]).T),axis=1) #G adds accuracy
			Data1 = np.concatenate((Data1,np.array([youd[permDoctors[j*groupSize:(j+1)*groupSize]]]).T),axis=1) #G adds Youdens
			Data1 = np.concatenate((Data1,np.array([sens[permDoctors[j*groupSize:(j+1)*groupSize]]]).T),axis=1) #G adds sensitivity
			Data1 = np.concatenate((Data1,np.array([spec[permDoctors[j*groupSize:(j+1)*groupSize]]]).T),axis=1) #G adds specificity

			netDataTest[i*numGroups+j,:] = [k for l in Data1 for k in l]

	truthTrain=np.zeros([numGroups*numTrain,2])
	truthValid=np.zeros([numGroups*numValid,2])
	truthTest=np.zeros([numGroups*numTest,2])
	
	truthTrain[:,0]=1-truthDummyTrain; truthTrain[:,1]=truthDummyTrain; truthTrain.astype('int32')
	truthValid[:,0]=1-truthDummyValid; truthValid[:,1]=truthDummyValid; truthValid.astype('int32')
	truthTest[:,0]=1-truthDummyTest; truthTest[:,1]=truthDummyTest; truthTest.astype('int32')
	
	#permindex=np.random.permutation(numGroups*numTrain) #G random permutation of the case-opinion pairs
	train=netDataTrain#[permindex,:] It's gonna be permuted in the makeBatch function
	trainT=truthTrain#[permindex,:]

	valid=netDataValid
	validT=truthValid

	test=netDataTest
	testT=truthTest


	return train, trainT, valid, validT, test, testT




def combNetDataCreator(groupSize, propCombin, nums, statistics, dataFile):
#G data file has no headers
#G creates the data for all the combinations given a group size

	numDoctors = nums[0]
	numTrain = nums[1]
	numValid = nums[2]
	numTest = nums[3]
	numCases = numTrain + numValid + numTest

	#reading in the data from a csv file
	csvfile=open('./data/' + dataFile + '.csv','rb')
	dfile=csv.reader(csvfile)

	Ndata = numDoctors * numCases #number of case-opinion pairs
	data=np.zeros([Ndata,5])
	j=0
	for row in dfile: #G moves across rows
		i=0;
		for x in row: #G moves through the row
			data[j,i]=int(row[i])
			i+=1
		j+=1


	#make the matrix of data using groups which will be used for training the net
	allCombDoctors = list(itertools.combinations(range(numDoctors),groupSize))
	numCombin = len(allCombDoctors)
	randCombin = np.random.permutation(numCombin)
	numGroups = int(np.floor(numCombin*propCombin))
	print 'data size', numGroups*numTrain
	combDoctors = [allCombDoctors[i] for i in randCombin[:numGroups]]
	

	netDataTrain = np.zeros([numGroups*numTrain,6*groupSize])
	netDataValid = np.zeros([numGroups*numValid,6*groupSize])
	netDataTest = np.zeros([numGroups*numTest,6*groupSize])

	truthDummyTrain = np.matlib.repmat(data[:numTrain,2], numGroups,1).reshape(numGroups*numTrain)
	truthDummyValid = np.matlib.repmat(data[numTrain:numTrain+numValid,2], numGroups,1).reshape(numGroups*numValid)
	truthDummyTest = np.matlib.repmat(data[numTrain+numValid:numCases,2], numGroups,1).reshape(numGroups*numTest)

	#G remainder: data[ index, case, truth, opinion, confidence ]
	for i in range(numGroups):

		acc = statistics['indAcc']/100; #G accuracy
		youd = statistics['youden']; #G Youden's index
		sens = statistics['sensit']; #G sensitivity
		spec = statistics['specif']; #G specificity

		for j in range(numTrain):

			#foo = [lista[combDoctors[combinationPosition][0]],lista[combDoctors[samePosition][1]]]

			Data1 = np.array([data[k*numCases+j,:] for k in combDoctors[i]])
			#convert from 0,1 to (-1,1)
			Data1[:,3]=2*(Data1[:,3]-0.5)
			Data1=Data1[:,3:] #G store only opinion and confidence
			Data1 = np.concatenate((Data1,np.array([acc[combDoctors[i][k]] for k in range(groupSize)]).reshape(groupSize,1)),axis=1) #G adds accuracy
			Data1 = np.concatenate((Data1,np.array([youd[combDoctors[i][k]] for k in range(groupSize)]).reshape(groupSize,1)),axis=1) #G adds Youdens
			Data1 = np.concatenate((Data1,np.array([sens[combDoctors[i][k]] for k in range(groupSize)]).reshape(groupSize,1)),axis=1) #G adds sensitivity
			Data1 = np.concatenate((Data1,np.array([spec[combDoctors[i][k]] for k in range(groupSize)]).reshape(groupSize,1)),axis=1) #G adds specificity

			netDataTrain[i*numTrain+j,:] = [k for l in Data1 for k in l]

		for j in range(numValid):

			Data1 = np.array([data[k*numCases+numTrain+j,:] for k in combDoctors[i]])
			#convert from 0,1 to (-1,1)
			Data1[:,3]=2*(Data1[:,3]-0.5)
			Data1=Data1[:,3:] #G store only opinion and confidence
			Data1 = np.concatenate((Data1,np.array([acc[combDoctors[i][k]] for k in range(groupSize)]).reshape(groupSize,1)),axis=1) #G adds accuracy
			Data1 = np.concatenate((Data1,np.array([youd[combDoctors[i][k]] for k in range(groupSize)]).reshape(groupSize,1)),axis=1) #G adds Youdens
			Data1 = np.concatenate((Data1,np.array([sens[combDoctors[i][k]] for k in range(groupSize)]).reshape(groupSize,1)),axis=1) #G adds sensitivity
			Data1 = np.concatenate((Data1,np.array([spec[combDoctors[i][k]] for k in range(groupSize)]).reshape(groupSize,1)),axis=1) #G adds specificity

			netDataValid[i*numValid+j,:] = [k for l in Data1 for k in l]

		for j in range(numTest):

			Data1 = np.array([data[k*numCases+numTrain+numValid+j,:] for k in combDoctors[i]])
			#convert from 0,1 to (-1,1)
			Data1[:,3]=2*(Data1[:,3]-0.5)
			Data1=Data1[:,3:] #G store only opinion and confidence
			Data1 = np.concatenate((Data1,np.array([acc[combDoctors[i][k]] for k in range(groupSize)]).reshape(groupSize,1)),axis=1) #G adds accuracy
			Data1 = np.concatenate((Data1,np.array([youd[combDoctors[i][k]] for k in range(groupSize)]).reshape(groupSize,1)),axis=1) #G adds Youdens
			Data1 = np.concatenate((Data1,np.array([sens[combDoctors[i][k]] for k in range(groupSize)]).reshape(groupSize,1)),axis=1) #G adds sensitivity
			Data1 = np.concatenate((Data1,np.array([spec[combDoctors[i][k]] for k in range(groupSize)]).reshape(groupSize,1)),axis=1) #G adds specificity

			netDataTest[i*numTest+j,:] = [k for l in Data1 for k in l]

	truthTrain=np.zeros([numGroups*numTrain,2])
	truthValid=np.zeros([numGroups*numValid,2])
	truthTest=np.zeros([numGroups*numTest,2])

	truthTrain[:,0]=1-truthDummyTrain; truthTrain[:,1]=truthDummyTrain; truthTrain.astype('int32')
	truthValid[:,0]=1-truthDummyValid; truthValid[:,1]=truthDummyValid; truthValid.astype('int32')
	truthTest[:,0]=1-truthDummyTest; truthTest[:,1]=truthDummyTest; truthTest.astype('int32')

	#permindex=np.random.permutation(numGroups*numTrain) #G random permutation of the case-opinion pairs
	train=netDataTrain#[permindex,:] It's gonna be permuted in the makeBatch function
	trainT=truthTrain#[permindex,:]

	valid=netDataValid
	validT=truthValid

	test=netDataTest
	testT=truthTest


	return train, trainT, valid, validT, test, testT




def makeBatch(data,truth,n):
	#makes a minibatch for network training
	#G n: batch size
	N=data.shape[0]
	ind=np.random.permutation(N)

	x=data[ind[:n],:]
	y=truth[ind[:n]]

	return x,y




def makeAllBatches(data,truth,n):
	#groups data in batches for network training
	#G n: batch size
	N=data.shape[0]
	ind=np.random.permutation(N)

	x=data[ind,:]
	y=truth[ind]

	return x,y





