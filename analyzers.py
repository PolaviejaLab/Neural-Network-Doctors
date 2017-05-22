
import os
import cPickle as pickle
import numpy as np
import seaborn as sns
import sys
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('GtkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from imp import reload
import pandas as pd
sys.path.append('../Gabriel_Project/Model')
sys.path.append('../Gabriel_Project/utils')
from loadData import loadData
#G from tf_utils import dense_to_one_hot
from docsStatistics import ROCStatistics
import performFunctions as pfu
reload(pfu)
import neuralNetG as nng
reload(nng)
#from pandas import DataFrame




def rawDataLoader(groupSize,propCombin,propTrain,propVal,createDoc = 'no',returnValue = 'no'):

	#groupSize = 2
	#propCombin = 0.1

	#G proportion of cases used for train, validation and test
	#propTrain = 1.0
	#propVal = 0.0
	propTest = 1 - propTrain - propVal

	if createDoc == 'yes':

		randCases = 'yes' #G 'yes': randomize cases; 'no', don't
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

		nng.partitionFileCreator(dataFilePre, randCases)

		#G wait until new data files are stored
		while True:
			if os.stat('./data/partitionFile.csv').st_ctime > ctime and os.stat('./data/partitionFileNoHeaders.csv').st_ctime > ctime2:
				break
		#time.sleep(1.0)

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
	importance = 1.0; #G if we want to give the same importance to a positive error than to a negative error (i.e. optimize Youden's index)
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

	#crowd = nng.netDataCreator(numDoctors, nums, statistics, dataFileNH, randDoctors = 'no') #G one single group (the crowd),
	#crowdDict = pfu.performance(crowd, nums) 							#G that is why 1st argument is numDoctors

	#G remainder: netData structure is [opi1 conf1 acc1 youd1 sens1 spec1  opi2 conf2 acc2 youd2 sens2 spec2 ...  opiN confN accN youdN sensN specN]
	netData = nng.combNetDataCreator(groupSize, propCombin, nums, statistics, dataFileNH)
	#groupsDict = pfu.performance(netData)
	#groupsDict = pfu.performSplit(netData, nums)

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

	
	data = [train, trainT, compTrain, valid, validT, compValid, test, testT, compTest]

	allData = [data, dataSplit]
	
	if returnValue == 'yes':
		return allData
	


	
def rawDataAnalyzer():

	groupSize = 2
	#G proportion of total combinations used
	propCombin = 1.0
	#G proportion of cases used for train, validation and test
	propTrain = 0.5
	propVal = 0.5
	
	data = rawDataLoader(groupSize,propCombin,propTrain,propVal,returnValue = 'yes')
	#data=pickle.load( open( 'dataComp.pkl', 'rb' ) )

	train = data[0][0]
	trainT = data[0][1]
	compTrain = data[0][2]
	valid = data[0][3]
	validT = data[0][4]
	compValid = data[0][5]
	test = data[0][6]
	testT = data[0][7]
	compTest = data[0][8]

	opins = train[:,0::6]
	confs = train[:,1::6]
	opiConf = np.append(opins,confs,axis=1) # [opi1 opi2 conf1 conf2] 

	popo = [c[2:4] for c in opiConf if c[0]==1 and c[1]==1] # Store the confidences in the cases when both said positive
	nepo = [c[2:4] for c in opiConf if c[0]==-1 and c[1]==1] # Store the confidences in the cases when doc1 said negative and doc2 said positive
	pone = [c[2:4] for c in opiConf if c[0]==1 and c[1]==-1] # Store the confidences in the cases when doc1 said positive and doc2 said negative
	nene = [c[2:4] for c in opiConf if c[0]==-1 and c[1]==-1] # Store the confidences in the cases when both said negative


	pp11 = len([c for c in popo if c[0]==1 and c[1]==1])
	pp12 = len([c for c in popo if (c[0]==1 and c[1]==2) or (c[0]==2 and c[1]==1)])
	pp13 = len([c for c in popo if (c[0]==1 and c[1]==3) or (c[0]==3 and c[1]==1)])
	pp14 = len([c for c in popo if (c[0]==1 and c[1]==4) or (c[0]==4 and c[1]==1)])
	pp22 = len([c for c in popo if c[0]==2 and c[1]==2])
	pp23 = len([c for c in popo if (c[0]==2 and c[1]==3) or (c[0]==3 and c[1]==2)])
	pp24 = len([c for c in popo if (c[0]==2 and c[1]==4) or (c[0]==4 and c[1]==2)])
	pp33 = len([c for c in popo if c[0]==3 and c[1]==3])
	pp34 = len([c for c in popo if (c[0]==3 and c[1]==4) or (c[0]==4 and c[1]==3)])
	pp44 = len([c for c in popo if c[0]==4 and c[1]==4])

	pospos = [pp11, pp12, pp13, pp14, pp22, pp23, pp24, pp33, pp34, pp44]
	#pospos = [c/2.0 for c in pospos] #DIVIDED BY TWO, CAUSE WE ARE USING PROPTRAIN = 1, BUT WANT TO AVERAGE OVER ALL POSSIBLE TRAIN PARTITIONS

	
	nn11 = len([c for c in nene if c[0]==1 and c[1]==1])
	nn12 = len([c for c in nene if (c[0]==1 and c[1]==2) or (c[0]==2 and c[1]==1)])
	nn13 = len([c for c in nene if (c[0]==1 and c[1]==3) or (c[0]==3 and c[1]==1)])
	nn14 = len([c for c in nene if (c[0]==1 and c[1]==4) or (c[0]==4 and c[1]==1)])
	nn22 = len([c for c in nene if c[0]==2 and c[1]==2])
	nn23 = len([c for c in nene if (c[0]==2 and c[1]==3) or (c[0]==3 and c[1]==2)])
	nn24 = len([c for c in nene if (c[0]==2 and c[1]==4) or (c[0]==4 and c[1]==2)])
	nn33 = len([c for c in nene if c[0]==3 and c[1]==3])
	nn34 = len([c for c in nene if (c[0]==3 and c[1]==4) or (c[0]==4 and c[1]==3)])
	nn44 = len([c for c in nene if c[0]==4 and c[1]==4])

	negneg = [nn11, nn12, nn13, nn14, nn22, nn23, nn24, nn33, nn34, nn44]
	#negneg = [c/2.0 for c in negneg] #DIVIDED BY TWO, CAUSE WE ARE USING PROPTRAIN = 1, BUT WANT TO AVERAGE OVER ALL POSSIBLE TRAIN PARTITIONS

	
	np11 = len([c for c in nepo if c[0]==1 and c[1]==1]) + len([c for c in pone if c[0]==1 and c[1]==1])
	np12 = len([c for c in nepo if c[0]==1 and c[1]==2]) + len([c for c in pone if c[0]==2 and c[1]==1])
	np13 = len([c for c in nepo if c[0]==1 and c[1]==3]) + len([c for c in pone if c[0]==3 and c[1]==1])
	np14 = len([c for c in nepo if c[0]==1 and c[1]==4]) + len([c for c in pone if c[0]==4 and c[1]==1])
	np21 = len([c for c in nepo if c[0]==2 and c[1]==1]) + len([c for c in pone if c[0]==1 and c[1]==2])
	np22 = len([c for c in nepo if c[0]==2 and c[1]==2]) + len([c for c in pone if c[0]==2 and c[1]==2])
	np23 = len([c for c in nepo if c[0]==2 and c[1]==3]) + len([c for c in pone if c[0]==3 and c[1]==2])
	np24 = len([c for c in nepo if c[0]==2 and c[1]==4]) + len([c for c in pone if c[0]==4 and c[1]==2])
	np31 = len([c for c in nepo if c[0]==3 and c[1]==1]) + len([c for c in pone if c[0]==1 and c[1]==3])
	np32 = len([c for c in nepo if c[0]==3 and c[1]==2]) + len([c for c in pone if c[0]==2 and c[1]==3])
	np33 = len([c for c in nepo if c[0]==3 and c[1]==3]) + len([c for c in pone if c[0]==3 and c[1]==3])
	np34 = len([c for c in nepo if c[0]==3 and c[1]==4]) + len([c for c in pone if c[0]==4 and c[1]==3])
	np41 = len([c for c in nepo if c[0]==4 and c[1]==1]) + len([c for c in pone if c[0]==1 and c[1]==4])
	np42 = len([c for c in nepo if c[0]==4 and c[1]==2]) + len([c for c in pone if c[0]==2 and c[1]==4])
	np43 = len([c for c in nepo if c[0]==4 and c[1]==3]) + len([c for c in pone if c[0]==3 and c[1]==4])
	np44 = len([c for c in nepo if c[0]==4 and c[1]==4]) + len([c for c in pone if c[0]==4 and c[1]==4])

	negpos = [np11, np12, np13, np14, np12, np22, np23, np24, np31, np32, np33, np34, np41, np42, np43, np44]
	#negpos = [c/2.0 for c in negpos] #DIVIDED BY TWO, CAUSE WE ARE USING PROPTRAIN = 1, BUT WANT TO AVERAGE OVER ALL POSSIBLE TRAIN PARTITIONS


	plt.close()
	fig = plt.figure(figsize=(14,11))
	plt.clf()
	'''
	plt.switch_backend('TkAgg')
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	'''
	sns.set_style('whitegrid',{'axes.grid' : False})

	gs = gridspec.GridSpec(2, 2)
	gs.update(left=0.09, right=0.97, bottom=0.08, top=0.94, wspace=0.25, hspace=0.3)

	titles = ['positive vs positive', 'negative vs negative', 'negative vs positive'] #vs with a point?
	toplot = [pospos, negneg, negpos]
	labels1 = ['1-1','1-2','1-3','1-4','2-2','2-3','2-4','3-3','3-4','4-4']
	labels2 = labels1
	labels3 = ['1-1','1-2','1-3','1-4','2-1','2-2','2-3','2-4','3-1','3-2','3-3','3-4','4-1','4-2','4-3','4-4']
	tolabels = [labels1, labels2, labels3]
	tosubpl = [[[0,1],[0,1]],[[0,1],[1,2]],[[1,2],[0,2]]]

	for i in xrange(3):

		tsp = tosubpl[i]
		axi = plt.subplot(gs[tsp[0][0]:tsp[0][1],tsp[1][0]:tsp[1][1]])
		tpl = toplot[i]
		spi = axi.bar(xrange(1,len(tpl)+1), tpl, align='center')
		axi.set_title(titles[i],fontsize=22)
		axi.set_xticks(range(1,len(tpl)+1))
		axi.set_xticklabels(tolabels[i],fontsize=16)
		axi.tick_params(axis = 'y', labelsize = 16)
		axi.set_xlabel('confidence-confidence',fontsize=20)
		axi.set_ylabel('number of groups',fontsize=20)
		axi.set_xlim([0.25,len(tpl)+0.75])

		
	plt.draw()
	#plt.show()
	plt.pause(1)
	plt.savefig('histsConfVsConf.tiff',dpi=75)



	opiConfComp = np.append(opiConf,compTrain,axis=1) # [opi1 opi2 conf1 conf2 comp1 comp2] 

	cLims = [[[0.00,0.50],[0.00,0.50]], [[0.50,0.75],[0.00,0.50]], [[0.00,0.50],[0.50,0.75]], [[0.75,1.00],[0.00,0.50]], [[0.00,0.50],[1.00,0.75]],
		[[0.50,0.75],[0.50,0.75]], [[0.75,1.00],[0.50,0.75]], [[0.50,0.75],[0.75,1.00]], [[0.75,1.00],[0.75,1.00]]] # partition limits on competence
	
	for j in xrange(len(cLims)):

		popo = [c[2:4] for c in opiConfComp if c[0]==1 and c[1]==1 and cLims[j][0][0]<=c[4]<=cLims[j][0][1] and cLims[j][1][0]<=c[5]<=cLims[j][1][1]] # confidences when both said positive
		nepo = [c[2:4] for c in opiConfComp if c[0]==-1 and c[1]==1 and cLims[j][0][0]<=c[4]<=cLims[j][0][1] and cLims[j][1][0]<=c[5]<=cLims[j][1][1]] # confidences when doc1 said negative and doc2 said positive
		pone = [c[2:4] for c in opiConfComp if c[0]==1 and c[1]==-1 and cLims[j][0][0]<=c[5]<=cLims[j][0][1] and cLims[j][1][0]<=c[4]<=cLims[j][1][1]] # confidences when doc1 said positive and doc2 said negative
		nene = [c[2:4] for c in opiConfComp if c[0]==-1 and c[1]==-1 and cLims[j][0][0]<=c[4]<=cLims[j][0][1] and cLims[j][1][0]<=c[5]<=cLims[j][1][1]] # confidences when both said negative


		pp11 = len([c for c in popo if c[0]==1 and c[1]==1])
		pp12 = len([c for c in popo if (c[0]==1 and c[1]==2) or (c[0]==2 and c[1]==1)])
		pp13 = len([c for c in popo if (c[0]==1 and c[1]==3) or (c[0]==3 and c[1]==1)])
		pp14 = len([c for c in popo if (c[0]==1 and c[1]==4) or (c[0]==4 and c[1]==1)])
		pp22 = len([c for c in popo if c[0]==2 and c[1]==2])
		pp23 = len([c for c in popo if (c[0]==2 and c[1]==3) or (c[0]==3 and c[1]==2)])
		pp24 = len([c for c in popo if (c[0]==2 and c[1]==4) or (c[0]==4 and c[1]==2)])
		pp33 = len([c for c in popo if c[0]==3 and c[1]==3])
		pp34 = len([c for c in popo if (c[0]==3 and c[1]==4) or (c[0]==4 and c[1]==3)])
		pp44 = len([c for c in popo if c[0]==4 and c[1]==4])

		pospos = [pp11, pp12, pp13, pp14, pp22, pp23, pp24, pp33, pp34, pp44]
		#pospos = [c/2.0 for c in pospos] #DIVIDED BY TWO, CAUSE WE ARE USING PROPTRAIN = 1, BUT WANT TO AVERAGE OVER ALL POSSIBLE TRAIN PARTITIONS

	
		nn11 = len([c for c in nene if c[0]==1 and c[1]==1])
		nn12 = len([c for c in nene if (c[0]==1 and c[1]==2) or (c[0]==2 and c[1]==1)])
		nn13 = len([c for c in nene if (c[0]==1 and c[1]==3) or (c[0]==3 and c[1]==1)])
		nn14 = len([c for c in nene if (c[0]==1 and c[1]==4) or (c[0]==4 and c[1]==1)])
		nn22 = len([c for c in nene if c[0]==2 and c[1]==2])
		nn23 = len([c for c in nene if (c[0]==2 and c[1]==3) or (c[0]==3 and c[1]==2)])
		nn24 = len([c for c in nene if (c[0]==2 and c[1]==4) or (c[0]==4 and c[1]==2)])
		nn33 = len([c for c in nene if c[0]==3 and c[1]==3])
		nn34 = len([c for c in nene if (c[0]==3 and c[1]==4) or (c[0]==4 and c[1]==3)])
		nn44 = len([c for c in nene if c[0]==4 and c[1]==4])

		negneg = [nn11, nn12, nn13, nn14, nn22, nn23, nn24, nn33, nn34, nn44]
		#negneg = [c/2.0 for c in negneg] #DIVIDED BY TWO, CAUSE WE ARE USING PROPTRAIN = 1, BUT WANT TO AVERAGE OVER ALL POSSIBLE TRAIN PARTITIONS

	
		np11 = len([c for c in nepo if c[0]==1 and c[1]==1]) + len([c for c in pone if c[0]==1 and c[1]==1])
		np12 = len([c for c in nepo if c[0]==1 and c[1]==2]) + len([c for c in pone if c[0]==2 and c[1]==1])
		np13 = len([c for c in nepo if c[0]==1 and c[1]==3]) + len([c for c in pone if c[0]==3 and c[1]==1])
		np14 = len([c for c in nepo if c[0]==1 and c[1]==4]) + len([c for c in pone if c[0]==4 and c[1]==1])
		np21 = len([c for c in nepo if c[0]==2 and c[1]==1]) + len([c for c in pone if c[0]==1 and c[1]==2])
		np22 = len([c for c in nepo if c[0]==2 and c[1]==2]) + len([c for c in pone if c[0]==2 and c[1]==2])
		np23 = len([c for c in nepo if c[0]==2 and c[1]==3]) + len([c for c in pone if c[0]==3 and c[1]==2])
		np24 = len([c for c in nepo if c[0]==2 and c[1]==4]) + len([c for c in pone if c[0]==4 and c[1]==2])
		np31 = len([c for c in nepo if c[0]==3 and c[1]==1]) + len([c for c in pone if c[0]==1 and c[1]==3])
		np32 = len([c for c in nepo if c[0]==3 and c[1]==2]) + len([c for c in pone if c[0]==2 and c[1]==3])
		np33 = len([c for c in nepo if c[0]==3 and c[1]==3]) + len([c for c in pone if c[0]==3 and c[1]==3])
		np34 = len([c for c in nepo if c[0]==3 and c[1]==4]) + len([c for c in pone if c[0]==4 and c[1]==3])
		np41 = len([c for c in nepo if c[0]==4 and c[1]==1]) + len([c for c in pone if c[0]==1 and c[1]==4])
		np42 = len([c for c in nepo if c[0]==4 and c[1]==2]) + len([c for c in pone if c[0]==2 and c[1]==4])
		np43 = len([c for c in nepo if c[0]==4 and c[1]==3]) + len([c for c in pone if c[0]==3 and c[1]==4])
		np44 = len([c for c in nepo if c[0]==4 and c[1]==4]) + len([c for c in pone if c[0]==4 and c[1]==4])

		negpos = [np11, np12, np13, np14, np12, np22, np23, np24, np31, np32, np33, np34, np41, np42, np43, np44]
		#negpos = [c/2.0 for c in negpos] #DIVIDED BY TWO, CAUSE WE ARE USING PROPTRAIN = 1, BUT WANT TO AVERAGE OVER ALL POSSIBLE TRAIN PARTITIONS


		plt.close()
		fig = plt.figure(figsize=(14,11))
		plt.clf()
		'''
		plt.switch_backend('TkAgg')
		mng = plt.get_current_fig_manager()
		mng.resize(*mng.window.maxsize())
		'''
		sns.set_style('whitegrid',{'axes.grid' : False})

		gs = gridspec.GridSpec(2, 2)
		gs.update(left=0.09, right=0.97, bottom=0.08, top=0.94, wspace=0.25, hspace=0.3)

		titles = ['positive vs positive', 'negative vs negative', 'negative vs positive'] #vs with a point?
		toplot = [pospos, negneg, negpos]
		labels1 = ['1-1','1-2','1-3','1-4','2-2','2-3','2-4','3-3','3-4','4-4']
		labels2 = labels1
		labels3 = ['1-1','1-2','1-3','1-4','2-1','2-2','2-3','2-4','3-1','3-2','3-3','3-4','4-1','4-2','4-3','4-4']
		tolabels = [labels1, labels2, labels3]
		tosubpl = [[[0,1],[0,1]],[[0,1],[1,2]],[[1,2],[0,2]]]

		for i in xrange(3):

			tsp = tosubpl[i]
			axi = plt.subplot(gs[tsp[0][0]:tsp[0][1],tsp[1][0]:tsp[1][1]])
			tpl = toplot[i]
			spi = axi.bar(xrange(1,len(tpl)+1), tpl, align='center')
			axi.set_title(titles[i],fontsize=22)
			axi.set_xticks(range(1,len(tpl)+1))
			axi.set_xticklabels(tolabels[i],fontsize=16)
			axi.tick_params(axis = 'y', labelsize = 16)
			axi.set_xlabel('confidence-confidence',fontsize=20)
			axi.set_ylabel('number of groups',fontsize=20)
			axi.set_xlim([0.25,len(tpl)+0.75])

		
		plt.draw()
		#plt.show()
		plt.pause(0.1)
		plt.savefig('histsConfVsConf_' + str(cLims[j][0][1]) + '_' + str(cLims[j][1][1]) + '.tiff',dpi=75)




def rawDataTruthsAnalyzer():

	groupSize = 2
	#G proportion of total combinations used
	propCombin = 1.0
	#G proportion of cases used for train, validation and test
	propTrain = 0.5
	propVal = 0.5
	
	data = rawDataLoader(groupSize,propCombin,propTrain,propVal,returnValue = 'yes')
	#data=pickle.load( open( 'dataComp.pkl', 'rb' ) )

	train = data[0][0]
	trainT = data[0][1]
	compTrain = data[0][2]
	valid = data[0][3]
	validT = data[0][4]
	compValid = data[0][5]
	test = data[0][6]
	testT = data[0][7]
	compTest = data[0][8]

	opins = train[:,0::6]
	confs = train[:,1::6]
	opiConf = np.append(opins,confs,axis=1) # [opi1 opi2 conf1 conf2] 
	opiConfTruths = np.append(opiConf,trainT, axis=1) # [opi1 opi2 conf1 conf2 truthNeg truthPos] 

	popoN = [c[2:4] for c in opiConfTruths if c[0]==1 and c[1]==1 and c[4]==1] # Store the confidences in the cases when both said positive, and truth was negative
	nepoN = [c[2:4] for c in opiConfTruths if c[0]==-1 and c[1]==1 and c[4]==1] # Store the confidences in the cases when doc1 said negative and doc2 said positive, and truth was negative
	poneN = [c[2:4] for c in opiConfTruths if c[0]==1 and c[1]==-1 and c[4]==1] # Store the confidences in the cases when doc1 said positive and doc2 said negative, and truth was negative
	neneN = [c[2:4] for c in opiConfTruths if c[0]==-1 and c[1]==-1 and c[4]==1] # Store the confidences in the cases when both said negative, and truth was negative

	popoP = [c[2:4] for c in opiConfTruths if c[0]==1 and c[1]==1 and c[5]==1] # Store the confidences in the cases when both said positive, and truth was negative
	nepoP = [c[2:4] for c in opiConfTruths if c[0]==-1 and c[1]==1 and c[5]==1] # Store the confidences in the cases when doc1 said negative and doc2 said positive, and truth was negative
	poneP = [c[2:4] for c in opiConfTruths if c[0]==1 and c[1]==-1 and c[5]==1] # Store the confidences in the cases when doc1 said positive and doc2 said negative, and truth was negative
	neneP = [c[2:4] for c in opiConfTruths if c[0]==-1 and c[1]==-1 and c[5]==1] # Store the confidences in the cases when both said negative, and truth was negative


	pp11N = len([c for c in popoN if c[0]==1 and c[1]==1])
	pp12N = len([c for c in popoN if (c[0]==1 and c[1]==2) or (c[0]==2 and c[1]==1)])
	pp13N = len([c for c in popoN if (c[0]==1 and c[1]==3) or (c[0]==3 and c[1]==1)])
	pp14N = len([c for c in popoN if (c[0]==1 and c[1]==4) or (c[0]==4 and c[1]==1)])
	pp22N = len([c for c in popoN if c[0]==2 and c[1]==2])
	pp23N = len([c for c in popoN if (c[0]==2 and c[1]==3) or (c[0]==3 and c[1]==2)])
	pp24N = len([c for c in popoN if (c[0]==2 and c[1]==4) or (c[0]==4 and c[1]==2)])
	pp33N = len([c for c in popoN if c[0]==3 and c[1]==3])
	pp34N = len([c for c in popoN if (c[0]==3 and c[1]==4) or (c[0]==4 and c[1]==3)])
	pp44N = len([c for c in popoN if c[0]==4 and c[1]==4])

	pp11P = len([c for c in popoP if c[0]==1 and c[1]==1])
	pp12P = len([c for c in popoP if (c[0]==1 and c[1]==2) or (c[0]==2 and c[1]==1)])
	pp13P = len([c for c in popoP if (c[0]==1 and c[1]==3) or (c[0]==3 and c[1]==1)])
	pp14P = len([c for c in popoP if (c[0]==1 and c[1]==4) or (c[0]==4 and c[1]==1)])
	pp22P = len([c for c in popoP if c[0]==2 and c[1]==2])
	pp23P = len([c for c in popoP if (c[0]==2 and c[1]==3) or (c[0]==3 and c[1]==2)])
	pp24P = len([c for c in popoP if (c[0]==2 and c[1]==4) or (c[0]==4 and c[1]==2)])
	pp33P = len([c for c in popoP if c[0]==3 and c[1]==3])
	pp34P = len([c for c in popoP if (c[0]==3 and c[1]==4) or (c[0]==4 and c[1]==3)])
	pp44P = len([c for c in popoP if c[0]==4 and c[1]==4])

	posposN = [pp11N, pp12N, pp13N, pp14N, pp22N, pp23N, pp24N, pp33N, pp34N, pp44N]
	posposP = [pp11P, pp12P, pp13P, pp14P, pp22P, pp23P, pp24P, pp33P, pp34P, pp44P]
	pospos = [posposN, posposP]
	#pospos = [[c/2.0 for c in pospos[i]] for i in xrange(len(pospos))] #DIVIDED BY TWO, CAUSE WE ARE USING PROPTRAIN = 1, BUT WANT TO AVERAGE OVER ALL POSSIBLE TRAIN PARTITIONS

	
	nn11N = len([c for c in neneN if c[0]==1 and c[1]==1])
	nn12N = len([c for c in neneN if (c[0]==1 and c[1]==2) or (c[0]==2 and c[1]==1)])
	nn13N = len([c for c in neneN if (c[0]==1 and c[1]==3) or (c[0]==3 and c[1]==1)])
	nn14N = len([c for c in neneN if (c[0]==1 and c[1]==4) or (c[0]==4 and c[1]==1)])
	nn22N = len([c for c in neneN if c[0]==2 and c[1]==2])
	nn23N = len([c for c in neneN if (c[0]==2 and c[1]==3) or (c[0]==3 and c[1]==2)])
	nn24N = len([c for c in neneN if (c[0]==2 and c[1]==4) or (c[0]==4 and c[1]==2)])
	nn33N = len([c for c in neneN if c[0]==3 and c[1]==3])
	nn34N = len([c for c in neneN if (c[0]==3 and c[1]==4) or (c[0]==4 and c[1]==3)])
	nn44N = len([c for c in neneN if c[0]==4 and c[1]==4])

	nn11P = len([c for c in neneP if c[0]==1 and c[1]==1])
	nn12P = len([c for c in neneP if (c[0]==1 and c[1]==2) or (c[0]==2 and c[1]==1)])
	nn13P = len([c for c in neneP if (c[0]==1 and c[1]==3) or (c[0]==3 and c[1]==1)])
	nn14P = len([c for c in neneP if (c[0]==1 and c[1]==4) or (c[0]==4 and c[1]==1)])
	nn22P = len([c for c in neneP if c[0]==2 and c[1]==2])
	nn23P = len([c for c in neneP if (c[0]==2 and c[1]==3) or (c[0]==3 and c[1]==2)])
	nn24P = len([c for c in neneP if (c[0]==2 and c[1]==4) or (c[0]==4 and c[1]==2)])
	nn33P = len([c for c in neneP if c[0]==3 and c[1]==3])
	nn34P = len([c for c in neneP if (c[0]==3 and c[1]==4) or (c[0]==4 and c[1]==3)])
	nn44P = len([c for c in neneP if c[0]==4 and c[1]==4])

	negnegN = [nn11N, nn12N, nn13N, nn14N, nn22N, nn23N, nn24N, nn33N, nn34N, nn44N]
	negnegP = [nn11P, nn12P, nn13P, nn14P, nn22P, nn23P, nn24P, nn33P, nn34P, nn44P]
	negneg = [negnegN, negnegP]
	#negneg = [[c/2.0 for c in negneg[i]] for i in xrange(len(negneg))] #DIVIDED BY TWO, CAUSE WE ARE USING PROPTRAIN = 1, BUT WANT TO AVERAGE OVER ALL POSSIBLE TRAIN PARTITIONS

	
	np11N = len([c for c in nepoN if c[0]==1 and c[1]==1]) + len([c for c in poneN if c[0]==1 and c[1]==1])
	np12N = len([c for c in nepoN if c[0]==1 and c[1]==2]) + len([c for c in poneN if c[0]==2 and c[1]==1])
	np13N = len([c for c in nepoN if c[0]==1 and c[1]==3]) + len([c for c in poneN if c[0]==3 and c[1]==1])
	np14N = len([c for c in nepoN if c[0]==1 and c[1]==4]) + len([c for c in poneN if c[0]==4 and c[1]==1])
	np21N = len([c for c in nepoN if c[0]==2 and c[1]==1]) + len([c for c in poneN if c[0]==1 and c[1]==2])
	np22N = len([c for c in nepoN if c[0]==2 and c[1]==2]) + len([c for c in poneN if c[0]==2 and c[1]==2])
	np23N = len([c for c in nepoN if c[0]==2 and c[1]==3]) + len([c for c in poneN if c[0]==3 and c[1]==2])
	np24N = len([c for c in nepoN if c[0]==2 and c[1]==4]) + len([c for c in poneN if c[0]==4 and c[1]==2])
	np31N = len([c for c in nepoN if c[0]==3 and c[1]==1]) + len([c for c in poneN if c[0]==1 and c[1]==3])
	np32N = len([c for c in nepoN if c[0]==3 and c[1]==2]) + len([c for c in poneN if c[0]==2 and c[1]==3])
	np33N = len([c for c in nepoN if c[0]==3 and c[1]==3]) + len([c for c in poneN if c[0]==3 and c[1]==3])
	np34N = len([c for c in nepoN if c[0]==3 and c[1]==4]) + len([c for c in poneN if c[0]==4 and c[1]==3])
	np41N = len([c for c in nepoN if c[0]==4 and c[1]==1]) + len([c for c in poneN if c[0]==1 and c[1]==4])
	np42N = len([c for c in nepoN if c[0]==4 and c[1]==2]) + len([c for c in poneN if c[0]==2 and c[1]==4])
	np43N = len([c for c in nepoN if c[0]==4 and c[1]==3]) + len([c for c in poneN if c[0]==3 and c[1]==4])
	np44N = len([c for c in nepoN if c[0]==4 and c[1]==4]) + len([c for c in poneN if c[0]==4 and c[1]==4])

	np11P = len([c for c in nepoP if c[0]==1 and c[1]==1]) + len([c for c in poneP if c[0]==1 and c[1]==1])
	np12P = len([c for c in nepoP if c[0]==1 and c[1]==2]) + len([c for c in poneP if c[0]==2 and c[1]==1])
	np13P = len([c for c in nepoP if c[0]==1 and c[1]==3]) + len([c for c in poneP if c[0]==3 and c[1]==1])
	np14P = len([c for c in nepoP if c[0]==1 and c[1]==4]) + len([c for c in poneP if c[0]==4 and c[1]==1])
	np21P = len([c for c in nepoP if c[0]==2 and c[1]==1]) + len([c for c in poneP if c[0]==1 and c[1]==2])
	np22P = len([c for c in nepoP if c[0]==2 and c[1]==2]) + len([c for c in poneP if c[0]==2 and c[1]==2])
	np23P = len([c for c in nepoP if c[0]==2 and c[1]==3]) + len([c for c in poneP if c[0]==3 and c[1]==2])
	np24P = len([c for c in nepoP if c[0]==2 and c[1]==4]) + len([c for c in poneP if c[0]==4 and c[1]==2])
	np31P = len([c for c in nepoP if c[0]==3 and c[1]==1]) + len([c for c in poneP if c[0]==1 and c[1]==3])
	np32P = len([c for c in nepoP if c[0]==3 and c[1]==2]) + len([c for c in poneP if c[0]==2 and c[1]==3])
	np33P = len([c for c in nepoP if c[0]==3 and c[1]==3]) + len([c for c in poneP if c[0]==3 and c[1]==3])
	np34P = len([c for c in nepoP if c[0]==3 and c[1]==4]) + len([c for c in poneP if c[0]==4 and c[1]==3])
	np41P = len([c for c in nepoP if c[0]==4 and c[1]==1]) + len([c for c in poneP if c[0]==1 and c[1]==4])
	np42P = len([c for c in nepoP if c[0]==4 and c[1]==2]) + len([c for c in poneP if c[0]==2 and c[1]==4])
	np43P = len([c for c in nepoP if c[0]==4 and c[1]==3]) + len([c for c in poneP if c[0]==3 and c[1]==4])
	np44P = len([c for c in nepoP if c[0]==4 and c[1]==4]) + len([c for c in poneP if c[0]==4 and c[1]==4])

	negposN = [np11N, np12N, np13N, np14N, np12N, np22N, np23N, np24N, np31N, np32N, np33N, np34N, np41N, np42N, np43N, np44N]
	negposP = [np11P, np12P, np13P, np14P, np12P, np22P, np23P, np24P, np31P, np32P, np33P, np34P, np41P, np42P, np43P, np44P]
	negpos = [negposN, negposP]
	#negneg = [[c/2.0 for c in negneg[i]] for i in xrange(len(negneg))] #DIVIDED BY TWO, CAUSE WE ARE USING PROPTRAIN = 1, BUT WANT TO AVERAGE OVER ALL POSSIBLE TRAIN PARTITIONS



	plt.close()
	fig = plt.figure(figsize=(14,11))
	plt.clf()
	'''
	plt.switch_backend('TkAgg')
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	'''
	sns.set_style('whitegrid',{'axes.grid' : False})

	gs = gridspec.GridSpec(2, 2)
	gs.update(left=0.09, right=0.97, bottom=0.08, top=0.94, wspace=0.25, hspace=0.3)

	titles = ['positive vs positive', 'negative vs negative', 'negative vs positive'] #vs with a point?
	toplot = [pospos, negneg, negpos]
	labels1 = ['1-1','1-2','1-3','1-4','2-2','2-3','2-4','3-3','3-4','4-4']
	labels2 = labels1
	labels3 = ['1-1','1-2','1-3','1-4','2-1','2-2','2-3','2-4','3-1','3-2','3-3','3-4','4-1','4-2','4-3','4-4']
	tolabels = [labels1, labels2, labels3]
	tosubpl = [[[0,1],[0,1]],[[0,1],[1,2]],[[1,2],[0,2]]]

	for i in xrange(3):

		tsp = tosubpl[i]
		axi = plt.subplot(gs[tsp[0][0]:tsp[0][1],tsp[1][0]:tsp[1][1]])
		tpl = toplot[i]
		lenpl1 = len(tpl[0])
		lenpl2 = len(tpl[1])
		lenpl = np.maximum(lenpl1,lenpl2)
		sp1 = axi.bar(np.arange(1,lenpl1+1)-0.2, tpl[0], width=0.4, color='b', align='center', label='truth negative')
		sp2 = axi.bar(np.arange(1,lenpl1+1)+0.2, tpl[1], width=0.4, color='r', align='center', label='truth positive')
		axi.set_title(titles[i],fontsize=22)
		axi.set_xticks(range(1,lenpl+1))
		axi.set_xticklabels(tolabels[i],fontsize=16)
		axi.tick_params(axis = 'y', labelsize = 16)
		axi.set_xlabel('confidence-confidence',fontsize=20)
		axi.set_ylabel('number of groups',fontsize=20)
		axi.set_xlim([0.25,lenpl+0.75])
		axi.legend(loc = 2, ncol = 1, frameon=False, fontsize=18)

		
	plt.draw()
	#plt.show()
	plt.pause(1)
	plt.savefig('histsConfsTruths.tiff',dpi=75)



	opiConfComp = np.append(opiConf,compTrain,axis=1) # [opi1 opi2 conf1 conf2 comp1 comp2] 
	opiConfCompTruths = np.append(opiConfComp,trainT,axis=1) # [opi1 opi2 conf1 conf2 comp1 comp2 truthNeg truthPos] 

	cLims = [[[0.00,0.50],[0.00,0.50]], [[0.50,0.75],[0.00,0.50]], [[0.00,0.50],[0.50,0.75]], [[0.75,1.00],[0.00,0.50]], [[0.00,0.50],[1.00,0.75]],
		[[0.50,0.75],[0.50,0.75]], [[0.75,1.00],[0.50,0.75]], [[0.50,0.75],[0.75,1.00]], [[0.75,1.00],[0.75,1.00]]] # partition limits on competence
	
	for j in xrange(len(cLims)):

		popoN = [c[2:4] for c in opiConfCompTruths if c[0]==1 and c[1]==1 and cLims[j][0][0]<=c[4]<=cLims[j][0][1] and cLims[j][1][0]<=c[5]<=cLims[j][1][1] and c[6]==1] # Store the confidences in the cases when both said positive, and truth was negative
		nepoN = [c[2:4] for c in opiConfCompTruths if c[0]==-1 and c[1]==1 and cLims[j][0][0]<=c[4]<=cLims[j][0][1] and cLims[j][1][0]<=c[5]<=cLims[j][1][1] and c[6]==1] # Store the confidences in the cases when doc1 said negative and doc2 said positive, and truth was negative
		poneN = [c[2:4] for c in opiConfCompTruths if c[0]==1 and c[1]==-1 and cLims[j][0][0]<=c[5]<=cLims[j][0][1] and cLims[j][1][0]<=c[4]<=cLims[j][1][1] and c[6]==1] # Store the confidences in the cases when doc1 said positive and doc2 said negative, and truth was negative
		neneN = [c[2:4] for c in opiConfCompTruths if c[0]==-1 and c[1]==-1 and cLims[j][0][0]<=c[4]<=cLims[j][0][1] and cLims[j][1][0]<=c[5]<=cLims[j][1][1] and c[6]==1] # Store the confidences in the cases when both said negative, and truth was negative

		popoP = [c[2:4] for c in opiConfCompTruths if c[0]==1 and c[1]==1 and cLims[j][0][0]<=c[4]<=cLims[j][0][1] and cLims[j][1][0]<=c[5]<=cLims[j][1][1] and c[7]==1] # Store the confidences in the cases when both said positive, and truth was negative
		nepoP = [c[2:4] for c in opiConfCompTruths if c[0]==-1 and c[1]==1 and cLims[j][0][0]<=c[4]<=cLims[j][0][1] and cLims[j][1][0]<=c[5]<=cLims[j][1][1] and c[7]==1] # Store the confidences in the cases when doc1 said negative and doc2 said positive, and truth was negative
		poneP = [c[2:4] for c in opiConfCompTruths if c[0]==1 and c[1]==-1 and cLims[j][0][0]<=c[5]<=cLims[j][0][1] and cLims[j][1][0]<=c[4]<=cLims[j][1][1] and c[7]==1] # Store the confidences in the cases when doc1 said positive and doc2 said negative, and truth was negative
		neneP = [c[2:4] for c in opiConfCompTruths if c[0]==-1 and c[1]==-1 and cLims[j][0][0]<=c[4]<=cLims[j][0][1] and cLims[j][1][0]<=c[5]<=cLims[j][1][1] and c[7]==1] # Store the confidences in the cases when both said negative, and truth was negative


		pp11N = len([c for c in popoN if c[0]==1 and c[1]==1])
		pp12N = len([c for c in popoN if (c[0]==1 and c[1]==2) or (c[0]==2 and c[1]==1)])
		pp13N = len([c for c in popoN if (c[0]==1 and c[1]==3) or (c[0]==3 and c[1]==1)])
		pp14N = len([c for c in popoN if (c[0]==1 and c[1]==4) or (c[0]==4 and c[1]==1)])
		pp22N = len([c for c in popoN if c[0]==2 and c[1]==2])
		pp23N = len([c for c in popoN if (c[0]==2 and c[1]==3) or (c[0]==3 and c[1]==2)])
		pp24N = len([c for c in popoN if (c[0]==2 and c[1]==4) or (c[0]==4 and c[1]==2)])
		pp33N = len([c for c in popoN if c[0]==3 and c[1]==3])
		pp34N = len([c for c in popoN if (c[0]==3 and c[1]==4) or (c[0]==4 and c[1]==3)])
		pp44N = len([c for c in popoN if c[0]==4 and c[1]==4])

		pp11P = len([c for c in popoP if c[0]==1 and c[1]==1])
		pp12P = len([c for c in popoP if (c[0]==1 and c[1]==2) or (c[0]==2 and c[1]==1)])
		pp13P = len([c for c in popoP if (c[0]==1 and c[1]==3) or (c[0]==3 and c[1]==1)])
		pp14P = len([c for c in popoP if (c[0]==1 and c[1]==4) or (c[0]==4 and c[1]==1)])
		pp22P = len([c for c in popoP if c[0]==2 and c[1]==2])
		pp23P = len([c for c in popoP if (c[0]==2 and c[1]==3) or (c[0]==3 and c[1]==2)])
		pp24P = len([c for c in popoP if (c[0]==2 and c[1]==4) or (c[0]==4 and c[1]==2)])
		pp33P = len([c for c in popoP if c[0]==3 and c[1]==3])
		pp34P = len([c for c in popoP if (c[0]==3 and c[1]==4) or (c[0]==4 and c[1]==3)])
		pp44P = len([c for c in popoP if c[0]==4 and c[1]==4])

		posposN = [pp11N, pp12N, pp13N, pp14N, pp22N, pp23N, pp24N, pp33N, pp34N, pp44N]
		posposP = [pp11P, pp12P, pp13P, pp14P, pp22P, pp23P, pp24P, pp33P, pp34P, pp44P]
		pospos = [posposN, posposP]
		#pospos = [[c/2.0 for c in pospos[i]] for i in xrange(len(pospos))] #DIVIDED BY TWO, CAUSE WE ARE USING PROPTRAIN = 1, BUT WANT TO AVERAGE OVER ALL POSSIBLE TRAIN PARTITIONS

	
		nn11N = len([c for c in neneN if c[0]==1 and c[1]==1])
		nn12N = len([c for c in neneN if (c[0]==1 and c[1]==2) or (c[0]==2 and c[1]==1)])
		nn13N = len([c for c in neneN if (c[0]==1 and c[1]==3) or (c[0]==3 and c[1]==1)])
		nn14N = len([c for c in neneN if (c[0]==1 and c[1]==4) or (c[0]==4 and c[1]==1)])
		nn22N = len([c for c in neneN if c[0]==2 and c[1]==2])
		nn23N = len([c for c in neneN if (c[0]==2 and c[1]==3) or (c[0]==3 and c[1]==2)])
		nn24N = len([c for c in neneN if (c[0]==2 and c[1]==4) or (c[0]==4 and c[1]==2)])
		nn33N = len([c for c in neneN if c[0]==3 and c[1]==3])
		nn34N = len([c for c in neneN if (c[0]==3 and c[1]==4) or (c[0]==4 and c[1]==3)])
		nn44N = len([c for c in neneN if c[0]==4 and c[1]==4])

		nn11P = len([c for c in neneP if c[0]==1 and c[1]==1])
		nn12P = len([c for c in neneP if (c[0]==1 and c[1]==2) or (c[0]==2 and c[1]==1)])
		nn13P = len([c for c in neneP if (c[0]==1 and c[1]==3) or (c[0]==3 and c[1]==1)])
		nn14P = len([c for c in neneP if (c[0]==1 and c[1]==4) or (c[0]==4 and c[1]==1)])
		nn22P = len([c for c in neneP if c[0]==2 and c[1]==2])
		nn23P = len([c for c in neneP if (c[0]==2 and c[1]==3) or (c[0]==3 and c[1]==2)])
		nn24P = len([c for c in neneP if (c[0]==2 and c[1]==4) or (c[0]==4 and c[1]==2)])
		nn33P = len([c for c in neneP if c[0]==3 and c[1]==3])
		nn34P = len([c for c in neneP if (c[0]==3 and c[1]==4) or (c[0]==4 and c[1]==3)])
		nn44P = len([c for c in neneP if c[0]==4 and c[1]==4])

		negnegN = [nn11N, nn12N, nn13N, nn14N, nn22N, nn23N, nn24N, nn33N, nn34N, nn44N]
		negnegP = [nn11P, nn12P, nn13P, nn14P, nn22P, nn23P, nn24P, nn33P, nn34P, nn44P]
		negneg = [negnegN, negnegP]
		#negneg = [[c/2.0 for c in negneg[i]] for i in xrange(len(negneg))] #DIVIDED BY TWO, CAUSE WE ARE USING PROPTRAIN = 1, BUT WANT TO AVERAGE OVER ALL POSSIBLE TRAIN PARTITIONS

	
		np11N = len([c for c in nepoN if c[0]==1 and c[1]==1]) + len([c for c in poneN if c[0]==1 and c[1]==1])
		np12N = len([c for c in nepoN if c[0]==1 and c[1]==2]) + len([c for c in poneN if c[0]==2 and c[1]==1])
		np13N = len([c for c in nepoN if c[0]==1 and c[1]==3]) + len([c for c in poneN if c[0]==3 and c[1]==1])
		np14N = len([c for c in nepoN if c[0]==1 and c[1]==4]) + len([c for c in poneN if c[0]==4 and c[1]==1])
		np21N = len([c for c in nepoN if c[0]==2 and c[1]==1]) + len([c for c in poneN if c[0]==1 and c[1]==2])
		np22N = len([c for c in nepoN if c[0]==2 and c[1]==2]) + len([c for c in poneN if c[0]==2 and c[1]==2])
		np23N = len([c for c in nepoN if c[0]==2 and c[1]==3]) + len([c for c in poneN if c[0]==3 and c[1]==2])
		np24N = len([c for c in nepoN if c[0]==2 and c[1]==4]) + len([c for c in poneN if c[0]==4 and c[1]==2])
		np31N = len([c for c in nepoN if c[0]==3 and c[1]==1]) + len([c for c in poneN if c[0]==1 and c[1]==3])
		np32N = len([c for c in nepoN if c[0]==3 and c[1]==2]) + len([c for c in poneN if c[0]==2 and c[1]==3])
		np33N = len([c for c in nepoN if c[0]==3 and c[1]==3]) + len([c for c in poneN if c[0]==3 and c[1]==3])
		np34N = len([c for c in nepoN if c[0]==3 and c[1]==4]) + len([c for c in poneN if c[0]==4 and c[1]==3])
		np41N = len([c for c in nepoN if c[0]==4 and c[1]==1]) + len([c for c in poneN if c[0]==1 and c[1]==4])
		np42N = len([c for c in nepoN if c[0]==4 and c[1]==2]) + len([c for c in poneN if c[0]==2 and c[1]==4])
		np43N = len([c for c in nepoN if c[0]==4 and c[1]==3]) + len([c for c in poneN if c[0]==3 and c[1]==4])
		np44N = len([c for c in nepoN if c[0]==4 and c[1]==4]) + len([c for c in poneN if c[0]==4 and c[1]==4])

		np11P = len([c for c in nepoP if c[0]==1 and c[1]==1]) + len([c for c in poneP if c[0]==1 and c[1]==1])
		np12P = len([c for c in nepoP if c[0]==1 and c[1]==2]) + len([c for c in poneP if c[0]==2 and c[1]==1])
		np13P = len([c for c in nepoP if c[0]==1 and c[1]==3]) + len([c for c in poneP if c[0]==3 and c[1]==1])
		np14P = len([c for c in nepoP if c[0]==1 and c[1]==4]) + len([c for c in poneP if c[0]==4 and c[1]==1])
		np21P = len([c for c in nepoP if c[0]==2 and c[1]==1]) + len([c for c in poneP if c[0]==1 and c[1]==2])
		np22P = len([c for c in nepoP if c[0]==2 and c[1]==2]) + len([c for c in poneP if c[0]==2 and c[1]==2])
		np23P = len([c for c in nepoP if c[0]==2 and c[1]==3]) + len([c for c in poneP if c[0]==3 and c[1]==2])
		np24P = len([c for c in nepoP if c[0]==2 and c[1]==4]) + len([c for c in poneP if c[0]==4 and c[1]==2])
		np31P = len([c for c in nepoP if c[0]==3 and c[1]==1]) + len([c for c in poneP if c[0]==1 and c[1]==3])
		np32P = len([c for c in nepoP if c[0]==3 and c[1]==2]) + len([c for c in poneP if c[0]==2 and c[1]==3])
		np33P = len([c for c in nepoP if c[0]==3 and c[1]==3]) + len([c for c in poneP if c[0]==3 and c[1]==3])
		np34P = len([c for c in nepoP if c[0]==3 and c[1]==4]) + len([c for c in poneP if c[0]==4 and c[1]==3])
		np41P = len([c for c in nepoP if c[0]==4 and c[1]==1]) + len([c for c in poneP if c[0]==1 and c[1]==4])
		np42P = len([c for c in nepoP if c[0]==4 and c[1]==2]) + len([c for c in poneP if c[0]==2 and c[1]==4])
		np43P = len([c for c in nepoP if c[0]==4 and c[1]==3]) + len([c for c in poneP if c[0]==3 and c[1]==4])
		np44P = len([c for c in nepoP if c[0]==4 and c[1]==4]) + len([c for c in poneP if c[0]==4 and c[1]==4])

		negposN = [np11N, np12N, np13N, np14N, np12N, np22N, np23N, np24N, np31N, np32N, np33N, np34N, np41N, np42N, np43N, np44N]
		negposP = [np11P, np12P, np13P, np14P, np12P, np22P, np23P, np24P, np31P, np32P, np33P, np34P, np41P, np42P, np43P, np44P]
		negpos = [negposN, negposP]
		#negneg = [[c/2.0 for c in negneg[i]] for i in xrange(len(negneg))] #DIVIDED BY TWO, CAUSE WE ARE USING PROPTRAIN = 1, BUT WANT TO AVERAGE OVER ALL POSSIBLE TRAIN PARTITIONS



		plt.close()
		fig = plt.figure(figsize=(14,11))
		plt.clf()
		'''
		plt.switch_backend('TkAgg')
		mng = plt.get_current_fig_manager()
		mng.resize(*mng.window.maxsize())
		'''
		sns.set_style('whitegrid',{'axes.grid' : False})

		gs = gridspec.GridSpec(2, 2)
		gs.update(left=0.09, right=0.97, bottom=0.08, top=0.94, wspace=0.25, hspace=0.3)

		titles = ['positive vs positive', 'negative vs negative', 'negative vs positive'] #vs with a point?
		toplot = [pospos, negneg, negpos]
		labels1 = ['1-1','1-2','1-3','1-4','2-2','2-3','2-4','3-3','3-4','4-4']
		labels2 = labels1
		labels3 = ['1-1','1-2','1-3','1-4','2-1','2-2','2-3','2-4','3-1','3-2','3-3','3-4','4-1','4-2','4-3','4-4']
		tolabels = [labels1, labels2, labels3]
		tosubpl = [[[0,1],[0,1]],[[0,1],[1,2]],[[1,2],[0,2]]]

		for i in xrange(3):

			tsp = tosubpl[i]
			axi = plt.subplot(gs[tsp[0][0]:tsp[0][1],tsp[1][0]:tsp[1][1]])
			tpl = toplot[i]
			lenpl1 = len(tpl[0])
			lenpl2 = len(tpl[1])
			lenpl = np.maximum(lenpl1,lenpl2)
			sp1 = axi.bar(np.arange(1,lenpl1+1)-0.2, tpl[0], width=0.4, color='b', align='center', label='truth negative')
			sp2 = axi.bar(np.arange(1,lenpl1+1)+0.2, tpl[1], width=0.4, color='r', align='center', label='truth positive')
			axi.set_title(titles[i],fontsize=22)
			axi.set_xticks(range(1,lenpl+1))
			axi.set_xticklabels(tolabels[i],fontsize=16)
			axi.tick_params(axis = 'y', labelsize = 16)
			axi.set_xlabel('confidence-confidence',fontsize=20)
			axi.set_ylabel('number of groups',fontsize=20)
			axi.set_xlim([0.25,lenpl+0.75])
			axi.legend(loc = 2, ncol = 1, frameon=False, fontsize=18)

		
		plt.draw()
		#plt.show()
		plt.pause(0.1)
		plt.savefig('histsConfsTruths_' + str(cLims[j][0][1]) + '_' + str(cLims[j][1][1]) + '.tiff',dpi=75)





def checkNetConfs():

	checkDict = checkNet()
	pposTrain = checkDict['pposTrain']
	'''
	netPred=pickle.load( open( 'netPred3000.pkl', 'rb' ) )
	pposTrain = netPred[0]
	'''
	groupSize = 2
	#G proportion of total combinations used
	propCombin = 1.0
	#G proportion of cases used for train, validation and test
	propTrain = 0.5
	propVal = 0.5
	
	data = rawDataLoader(groupSize,propCombin,propTrain,propVal,returnValue = 'yes')
	#data=pickle.load( open( 'dataComp.pkl', 'rb' ) )

	train = data[0][0]
	trainT = data[0][1]
	compTrain = data[0][2]
	valid = data[0][3]
	validT = data[0][4]
	compValid = data[0][5]
	test = data[0][6]
	testT = data[0][7]
	compTest = data[0][8]

	opins = train[:,0::6]
	confs = train[:,1::6]
	opiConf = np.append(opins,confs,axis=1) # [opi1 opi2 conf1 conf2] 
	opiConfTruths = np.append(opiConf,trainT, axis=1) # [opi1 opi2 conf1 conf2 truthNeg truthPos] 
	decis = np.round(pposTrain)
	dataAndNet = [np.append(opiConfTruths[i],decis[i]) for i in xrange(len(decis))] # [opi1 opi2 conf1 conf2 truthNeg truthPos net_decision] 
	
	inds = [2,3,6] # confs and net probs

	popoN = [[c[i] for i in inds] for c in dataAndNet if c[0]==1 and c[1]==1 and c[4]==1] # Store the confidences and net decision in the cases when both said positive, and truth was negative
	nepoN = [[c[i] for i in inds] for c in dataAndNet if c[0]==-1 and c[1]==1 and c[4]==1] # Store the confidences and net decision in the cases when doc1 said negative and doc2 said positive, and truth was negative
	poneN = [[c[i] for i in inds] for c in dataAndNet if c[0]==1 and c[1]==-1 and c[4]==1] # Store the confidences and net decision in the cases when doc1 said positive and doc2 said negative, and truth was negative
	neneN = [[c[i] for i in inds] for c in dataAndNet if c[0]==-1 and c[1]==-1 and c[4]==1] # Store the confidences and net decision in the cases when both said negative, and truth was negative

	popoP = [[c[i] for i in inds] for c in dataAndNet if c[0]==1 and c[1]==1 and c[5]==1] # Store the confidences and net decision in the cases when both said positive, and truth was negative
	nepoP = [[c[i] for i in inds] for c in dataAndNet if c[0]==-1 and c[1]==1 and c[5]==1] # Store the confidences and net decision in the cases when doc1 said negative and doc2 said positive, and truth was negative
	poneP = [[c[i] for i in inds] for c in dataAndNet if c[0]==1 and c[1]==-1 and c[5]==1] # Store the confidences and net decision in the cases when doc1 said positive and doc2 said negative, and truth was negative
	neneP = [[c[i] for i in inds] for c in dataAndNet if c[0]==-1 and c[1]==-1 and c[5]==1] # Store the confidences and net decision in the cases when both said negative, and truth was negative


	cond1 = [[1,1],[1,2],[1,3],[1,4],[2,2],[2,3],[2,4],[3,3],[3,4],[4,4]]

	# the following lines: 1st letter, doc 1; 2nd letter, doc2; 3rd letter (if total is 4), network; last capital letter, truth 
 
	ppN = [[c for c in popoN if c[0:2]==cond1[i] or c[0:2]==[cond1[i][1],cond1[i][0]]] for i in xrange(len(cond1))]; # confidences and net conditional on confidences configuration
	ppnN = [[c for c in ppN[i] if c[2]==0] for i in xrange(len(ppN))] # The previous but selecting only the cases where the net says negative	
	pppN = [[c for c in ppN[i] if c[2]==1] for i in xrange(len(ppN))] # The previous but selecting only the cases where the net says positive 	

	ppP = [[c for c in popoP if c[0:2]==cond1[i] or c[0:2]==[cond1[i][1],cond1[i][0]]] for i in xrange(len(cond1))]; # confidences and net conditional on confidences configuration
	ppnP = [[c for c in ppP[i] if c[2]==0] for i in xrange(len(ppP))] # The previous but selecting only the cases where the net says negative	
	pppP = [[c for c in ppP[i] if c[2]==1] for i in xrange(len(ppP))] # The previous but selecting only the cases where the net says positive 	

	nnN = [[c for c in neneN if c[0:2]==cond1[i] or c[0:2]==[cond1[i][1],cond1[i][0]]] for i in xrange(len(cond1))]; # confidences and net conditional on confidences configuration
	nnnN = [[c for c in nnN[i] if c[2]==0] for i in xrange(len(nnN))] # The previous but selecting only the cases where the net says negative	
	nnpN = [[c for c in nnN[i] if c[2]==1] for i in xrange(len(nnN))] # The previous but selecting only the cases where the net says positive 	

	nnP = [[c for c in neneP if c[0:2]==cond1[i] or c[0:2]==[cond1[i][1],cond1[i][0]]] for i in xrange(len(cond1))]; # confidences and net conditional on confidences configuration
	nnnP = [[c for c in nnP[i] if c[2]==0] for i in xrange(len(nnP))] # The previous but selecting only the cases where the net says negative	
	nnpP = [[c for c in nnP[i] if c[2]==1] for i in xrange(len(nnP))] # The previous but selecting only the cases where the net says positive 
	

	cond2 = [[1,1],[1,2],[1,3],[1,4],[2,1],[2,2],[2,3],[2,4],[3,1],[3,2],[3,3],[3,4],[4,1],[4,2],[4,3],[4,4]]
	
	npN = [[c for c in nepoN if c[0:2]==cond2[i]] + [c for c in poneN if c[0:2]==[cond2[i][1],cond2[i][0]]] for i in xrange(len(cond2))]; # confidences and net conditional on confidences configuration
	npnN = [[c for c in npN[i] if c[2]==0] for i in xrange(len(npN))] # The previous but selecting only the cases where the net says negative	
	nppN = [[c for c in npN[i] if c[2]==1] for i in xrange(len(npN))] # The previous but selecting only the cases where the net says positive 	

	npP = [[c for c in nepoP if c[0:2]==cond2[i]] + [c for c in poneP if c[0:2]==[cond2[i][1],cond2[i][0]]] for i in xrange(len(cond2))]; # confidences and net conditional on confidences configuration
	npnP = [[c for c in npP[i] if c[2]==0] for i in xrange(len(npP))] # The previous but selecting only the cases where the net says negative	
	nppP = [[c for c in npP[i] if c[2]==1] for i in xrange(len(npP))] # The previous but selecting only the cases where the net says positive 	

        
	popo_neNe = [len(c) for c in ppnN]
	popo_poNe = [len(c) for c in pppN]
	popo_nePo = [len(c) for c in ppnP]
	popo_poPo = [len(c) for c in pppP]

	nene_neNe = [len(c) for c in nnnN]
	nene_poNe = [len(c) for c in nnpN]
	nene_nePo = [len(c) for c in nnnP]
	nene_poPo = [len(c) for c in nnpP]

	nepo_neNe = [len(c) for c in npnN]
	nepo_poNe = [len(c) for c in nppN]
	nepo_nePo = [len(c) for c in npnP]
	nepo_poPo = [len(c) for c in nppP]




	plt.close()
	fig = plt.figure(figsize=(14,11))
	plt.clf()
	'''
	plt.switch_backend('TkAgg')
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	'''
	sns.set_style('whitegrid',{'axes.grid' : False})

	gs = gridspec.GridSpec(2, 2)
	gs.update(left=0.09, right=0.97, bottom=0.08, top=0.94, wspace=0.25, hspace=0.3)

	titles = ['positive vs positive, truth negative', 'positive vs positive, truth positive', 'negative vs negative, truth negative', 'negative vs negative, truth positive'] #vs with a point?
	toplot = [[popo_neNe,popo_poNe], [popo_nePo,popo_poPo], [nene_neNe,nene_poNe], [nene_nePo,nene_poPo]]
	labels = ['1-1','1-2','1-3','1-4','2-2','2-3','2-4','3-3','3-4','4-4']
	tosubpl = [[[0,1],[0,1]],[[0,1],[1,2]],[[1,2],[0,1]],[[1,2],[1,2]]]

	for i in xrange(4):

		tsp = tosubpl[i]
		axi = plt.subplot(gs[tsp[0][0]:tsp[0][1],tsp[1][0]:tsp[1][1]])
		tpl = toplot[i]
		lenpl1 = len(tpl[0])
		lenpl2 = len(tpl[1])
		lenpl = np.maximum(lenpl1,lenpl2)
		sp1 = axi.bar(np.arange(1,lenpl1+1)-0.2, tpl[0], width=0.4, color='b', align='center', label='net negative')
		sp2 = axi.bar(np.arange(1,lenpl1+1)+0.2, tpl[1], width=0.4, color='r', align='center', label='net positive')
		axi.set_title(titles[i],fontsize=22)
		axi.set_xticks(range(1,lenpl+1))
		axi.set_xticklabels(labels,fontsize=16)
		axi.tick_params(axis = 'y', labelsize = 16)
		axi.set_xlabel('confidence-confidence',fontsize=20)
		axi.set_ylabel('number of groups',fontsize=20)
		axi.set_xlim([0.25,lenpl+0.75])
		axi.legend(loc = 0, ncol = 1, frameon=False, fontsize=18)

		
	plt.draw()
	#plt.show()
	plt.pause(1)
	plt.savefig('confsNetAgree.tiff',dpi=75)

	
	
	plt.close()
	fig = plt.figure(figsize=(14,11))
	plt.clf()
	'''
	plt.switch_backend('TkAgg')
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	'''
	sns.set_style('whitegrid',{'axes.grid' : False})

	gs = gridspec.GridSpec(2, 2)
	gs.update(left=0.09, right=0.97, bottom=0.08, top=0.94, wspace=0.25, hspace=0.3)

	titles = ['negative vs positive, truth negative', 'negative vs positive, truth positive'] #vs with a point?
	toplot = [[nepo_neNe,nepo_poNe],[nepo_nePo,nepo_poPo]]
	labels = ['1-1','1-2','1-3','1-4','2-1','2-2','2-3','2-4','3-1','3-2','3-3','3-4','4-1','4-2','4-3','4-4']
	tosubpl = [[[0,1],[0,2]],[[1,2],[0,2]]]

	for i in xrange(2):

		tsp = tosubpl[i]
		axi = plt.subplot(gs[tsp[0][0]:tsp[0][1],tsp[1][0]:tsp[1][1]])
		tpl = toplot[i]
		lenpl1 = len(tpl[0])
		lenpl2 = len(tpl[1])
		lenpl = np.maximum(lenpl1,lenpl2)
		sp1 = axi.bar(np.arange(1,lenpl1+1)-0.2, tpl[0], width=0.4, color='b', align='center', label='net negative')
		sp2 = axi.bar(np.arange(1,lenpl1+1)+0.2, tpl[1], width=0.4, color='r', align='center', label='net positive')
		axi.set_title(titles[i],fontsize=22)
		axi.set_xticks(range(1,lenpl+1))
		axi.set_xticklabels(labels,fontsize=16)
		axi.tick_params(axis = 'y', labelsize = 16)
		axi.set_xlabel('confidence-confidence',fontsize=20)
		axi.set_ylabel('number of groups',fontsize=20)
		axi.set_xlim([0.25,lenpl+0.75])
		axi.legend(loc = 2, ncol = 1, frameon=False, fontsize=18)

		
	plt.draw()
	#plt.show()
	plt.pause(1)
	plt.savefig('confsNetDisag.tiff',dpi=75)




def accVsConf(propCombin,propTrain,propVal,bin=0.1):

	groupSize = 1
	data = rawDataLoader(groupSize,propCombin,propTrain,propVal, returnValue = 'yes')
	#data=pickle.load( open( 'dataComp.pkl', 'rb' ) )

	train = data[0][0]
	trainT = data[0][1]
	compTrain = data[0][2]
	valid = data[0][3]
	validT = data[0][4]
	compValid = data[0][5]
	test = data[0][6]
	testT = data[0][7]
	compTest = data[0][8]

	opins = train[:,0]
	confs = train[:,1]
	opiConf = [[opins[i],confs[i]] for i in xrange(len(opins))] # [opi conf]
	opiConfComp = np.append(opiConf,compTrain,axis=1) # [opi conf comp] 
	dataAndTruths = np.append(opiConfComp,trainT,axis=1) # [opi conf comp truthNeg truthPos] 

	posN = [c[1:3] for c in dataAndTruths if c[0]==1 and c[3]==1] # Store the confidences and accuracies (youden) in the cases when doc said positive and truth was negative
	negN = [c[1:3] for c in dataAndTruths if c[0]==-1 and c[3]==1] # Store the confidences and accuracies (youden) in the cases when doc said negative and truth was negative
	posP = [c[1:3] for c in dataAndTruths if c[0]==1 and c[4]==1] # Store the confidences and accuracies (youden) in the cases when doc said positive and truth was positive
	negP = [c[1:3] for c in dataAndTruths if c[0]==-1 and c[4]==1] # Store the confidences and accuracies (youden) in the cases when doc said negative and truth was positive
	
	acc = np.arange(0,1+bin,bin) #accuracy (youden in the training)
	acc[-1] = acc[-1]+0.0001 #for the 'less than' to include = 1.0 in the last bin
	conf = np.arange(1,5) #confidence
	
	binPosN = [[[ a for a in posN if a[0]==c and acc[b]<=a[1]<acc[b+1] ] for c in conf] for b in xrange(len(acc)-1)]
	binNegN = [[[ a for a in negN if a[0]==c and acc[b]<=a[1]<acc[b+1] ] for c in conf] for b in xrange(len(acc)-1)]
	binPosP = [[[ a for a in posP if a[0]==c and acc[b]<=a[1]<acc[b+1] ] for c in conf] for b in xrange(len(acc)-1)]
	binNegP = [[[ a for a in negP if a[0]==c and acc[b]<=a[1]<acc[b+1] ] for c in conf] for b in xrange(len(acc)-1)]

	lenPosN = [[ len(binPosN[i][j]) for j in xrange(len(binPosN[0])) ] for i in xrange(len(binPosN))]
	lenNegN = [[ len(binNegN[i][j]) for j in xrange(len(binNegN[0])) ] for i in xrange(len(binNegN))]
	lenPosP = [[ len(binPosP[i][j]) for j in xrange(len(binPosP[0])) ] for i in xrange(len(binPosP))]
	lenNegP = [[ len(binNegP[i][j]) for j in xrange(len(binNegP[0])) ] for i in xrange(len(binNegP))]
	
	# norm_division_G function is defined in the end of this document
	correctTruthNeg = [[norm_division_G(lenNegN[i][j],lenPosN[i][j]) for j in xrange(len(lenNegN[0]))] for i in xrange(len(lenNegN))] # Specificity
	correctTruthPos = [[norm_division_G(lenPosP[i][j],lenNegP[i][j]) for j in xrange(len(lenPosP[0]))] for i in xrange(len(lenPosP))] # Sensitivity

	correctDocNeg = [[norm_division_G(lenNegN[i][j],lenNegP[i][j]) for j in xrange(len(lenNegN[0]))] for i in xrange(len(lenNegN))] # Proportion of correct when doc said negative
	correctDocPos = [[norm_division_G(lenPosP[i][j],lenPosN[i][j]) for j in xrange(len(lenPosP[0]))] for i in xrange(len(lenPosP))] # Proportion of correct when doc said positiive

	# print [[lenNegN[i][j]+lenPosN[i][j] for j in xrange(len(lenNegN[0]))] for i in xrange(len(lenNegN))] 
	# print [[lenPosP[i][j]+lenNegP[i][j] for j in xrange(len(lenPosP[0]))] for i in xrange(len(lenPosP))]

	perfMatrices = {
		'acc': acc,
		'conf': conf,
		'correctTruthNeg': correctTruthNeg,
		'correctTruthPos': correctTruthPos,
		'correctDocNeg': correctDocNeg,
		'correctDocPos': correctDocPos
		}

	return perfMatrices




def accVsConfMethod(bin = 0.1, createDoc = 'no'):

	#G createDoc: use current partition document, or create new
	groupSize = 2
	#G proportion of total combinations used
	propCombin = 1.0
	#G proportion of cases used for train, validation and test
	propTrain = 0.5
	propVal = 0.5
	
	#G data is [[data group size: train net inputs, train truths, train adjusted accs, valid net imputs, ...],
	#G           [data single doctor: train truths, train doc decision, valid truths, ...]]
	data = rawDataLoader(groupSize,propCombin,propTrain,propVal,createDoc,returnValue = 'yes')
	#data=pickle.load( open( 'dataComp.pkl', 'rb' ) )

	#G bin = binning of accuracy in the performance matrices
	perfMatrices = accVsConf(propCombin,propTrain,propVal,bin)
	#G perfMatrices = [[1st bin of acc: conf=1, c=2, c=3, c=4], [2nd bin of acc: conf=1, c=2, c=3, c=4], ...]
	correctDocNeg = perfMatrices['correctDocNeg']  # Proportion of correct when doc said negative
	correctDocPos = perfMatrices['correctDocPos']  # Proportion of correct when doc said positive

	train = data[0][0]
	trainT = data[0][1]
	valid = data[0][3]
	validT = data[0][4]

	compTrain = data[0][2]
	compValid = data[0][5]

	confsTrain = [z[1::6] for z in train[:]]
	confsValid = [z[1::6] for z in valid[:]]
	
	estimatesTrain = [z[::6] for z in train[:]]
	estimatesValid = [z[::6] for z in valid[:]]

	truthsTrain = trainT
	truthsValid = validT

	acc = np.arange(0,1+bin,bin) #accuracy (youden in the training)
	acc[-1] = acc[-1]+0.0001 #for the 'less than' to include = 1.0 in the last bin
	conf = np.arange(1,5) #confidence

	binConfTrain = [[int(a-1) for a in confsTrain[i]] for i in xrange(len(confsTrain))]
	binConfValid = [[int(a-1) for a in confsValid[i]] for i in xrange(len(confsValid))]

	binCompTrain = [[int(np.ceil(a/bin)-1) for a in compTrain[i]] for i in xrange(len(compTrain))]
	[[a+1 for a in binCompTrain[i] if a==-1] for i in xrange(len(binCompTrain))] #To include acc=0 in the first bin (instead of it going in the 0th bin!)	
	binCompValid = [[int(np.ceil(a/bin)-1) for a in compValid[i]] for i in xrange(len(compValid))]
	[[a+1 for a in binCompValid[i] if a==-1] for i in xrange(len(binCompValid))] #To include acc=0 in the first bin (instead of it going in the 0th bin!)

	probCorrectValid = [[correctDocNeg[binCompValid[i][j]][binConfValid[i][j]]  if int(estimatesValid[i][j])==-1 else correctDocPos[binCompValid[i][j]][binConfValid[i][j]] for j in xrange(groupSize)] for i in xrange(len(binCompValid))]

	numTrain = len(data[1][0])
	numValid = len(data[1][0])

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

	matMet = [pfu.performCalcHigher(estimatesValid[i*numValid:(i+1)*numValid], truthsValid[i*numValid:(i+1)*numValid], probCorrectValid[i*numValid:(i+1)*numValid]) for i in xrange(numGroupsValid)]
	best = [pfu.performCalcHigher(estimatesValid[i*numValid:(i+1)*numValid], truthsValid[i*numValid:(i+1)*numValid], compValid[i*numValid:(i+1)*numValid]) for i in xrange(numGroupsValid)]
	conf = [pfu.performCalcHigher(estimatesValid[i*numValid:(i+1)*numValid], truthsValid[i*numValid:(i+1)*numValid], confsValid[i*numValid:(i+1)*numValid]) for i in xrange(numGroupsValid)]
	opti = [pfu.performCalcOptimistic(estimatesValid[i*numValid:(i+1)*numValid], truthsValid[i*numValid:(i+1)*numValid]) for i in xrange(numGroupsValid)]
	
	print 'matrix method',[np.mean( [matMet[i][j] for i in xrange(len(matMet))] ) for j in xrange(len(matMet[0]))]
	print 'best',[np.mean( [best[i][j] for i in xrange(len(best))] ) for j in xrange(len(best[0]))]
        print 'confidence',[np.mean( [conf[i][j] for i in xrange(len(conf))] ) for j in xrange(len(conf[0]))]
	print 'optimistic',[np.mean( [opti[i][j] for i in xrange(len(opti))] ) for j in xrange(len(opti[0]))]




def accVsConfPlotter(propCombin=1.0, bin = 0.05, propTrain = 1.0, propVal = 0.0):	

	#G propCombin: proportion of total combinations used
	#G propTrain, propValid: proportion of cases used for train, validation and test
	#G bin: binning of accuracy in the performance matrices	

	perfMatrices = accVsConf(propCombin,propTrain,propVal,bin)

	acc = perfMatrices['acc']
	conf = perfMatrices['conf']
	correctTruthNeg = perfMatrices['correctTruthNeg']
	correctTruthPos = perfMatrices['correctTruthPos']
	correctDocNeg = perfMatrices['correctDocNeg']
	correctDocPos = perfMatrices['correctDocPos']

	plt.close()
	fig = plt.figure(figsize=(14,11))
	plt.clf()
	'''
	plt.switch_backend('TkAgg')
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	'''
	sns.set_style('whitegrid',{'axes.grid' : False})

	gs = gridspec.GridSpec(2, 2)
	gs.update(left=0.09, right=0.97, bottom=0.08, top=0.94, wspace=0.25, hspace=0.3)

	titles = ['truth negative', 'truth positive', 'doctor negative', 'doctor positive'] #vs with a point?
	toplot = [correctTruthNeg, correctTruthPos, correctDocNeg, correctDocPos]
	xticks = np.arange(1.5,5.5)
	xticklabels = ('1','2','3','4')
	tosubpl = [[[0,1],[0,1]],[[0,1],[1,2]],[[1,2],[0,1]],[[1,2],[1,2]]]

	for i in xrange(4):
		tsp = tosubpl[i]
		axi = plt.subplot(gs[tsp[0][0]:tsp[0][1],tsp[1][0]:tsp[1][1]])
		tpl = toplot[i]

		noNans = [acc[r+1] for r in xrange(len(tpl)) if not all(np.isnan(tpl[r])) ]
		accLims = [min(noNans)-bin, max(noNans)];

		binCbar = 0.1 #will be tha spacing in the colorbar, but I need to define it here to do the following:
		minPlot = binCbar*np.floor(np.nanmin(np.array(tpl))/binCbar)
		maxPlot = binCbar*np.ceil(np.nanmax(np.array(tpl))/binCbar)
		
		spi = axi.pcolormesh(np.append(conf,conf[-1]+1),acc,np.ma.masked_invalid(np.asarray(tpl)), cmap='jet', vmin=minPlot, vmax=maxPlot)
		axi.set_ylim(accLims)
		axi.set_title(titles[i],fontsize=22)
		axi.set_xticks(xticks)
		axi.set_xticklabels(xticklabels)
		axi.tick_params(labelsize=16)
		axi.set_xlabel('confidence',fontsize=20)
		axi.set_ylabel('competence',fontsize=20)
		
		cbari = plt.colorbar(spi)
		cbari.set_label(label = 'probability of right', size = 20)
		cbari.set_ticks(np.arange(-1.0,1+binCbar,binCbar))
		cbari.ax.tick_params(labelsize = 14)

	
	plt.draw()
	#plt.show()
	plt.pause(1)
	plt.savefig('accVsConf1.tiff',dpi=75)














def accVsConf2(propCombin=0.1,bin=0.1,propTrain=0.5,propVal=0.5):

	groupSize = 2
	data = rawDataLoader(groupSize,propCombin,propTrain,propVal, returnValue = 'yes')
	#data=pickle.load( open( 'dataComp.pkl', 'rb' ) )

	train = data[0][0]
	trainT = data[0][1]
	compTrain = data[0][2]
	valid = data[0][3]
	validT = data[0][4]
	compValid = data[0][5]
	test = data[0][6]
	testT = data[0][7]
	compTest = data[0][8]

	acc = np.arange(0,1+bin,bin) #accuracy differences (youden in the training)
	acc[-1] = acc[-1]+0.0001 #for the 'less than' to include = 1.0 in the last bin
	conf = np.arange(0,4) #confidence differences
	
	#G MAYBE I SHOULD DO JUST THREE BINS: LOWER, EQUAL AND HIGHER

	if propVal == 0:

		opins = train[:,0::6]
		confs = train[:,1::6]
		opiConf = np.append(opins,confs,axis=1) # [opi1 opi2 conf1 conf2] 
		opiConfComp = np.append(opiConf,compTrain, axis=1) # [opi1 opi2 conf1 conf2 comp1 comp2] 
		dataAndTruths = np.append(opiConfComp,trainT, axis=1) # [opi1 opi2 conf1 conf2 comp1 comp2 truthNeg truthPos] 

		binData = [[[ a for a in dataAndTruths if np.absolute(a[2]-a[3])==c and acc[b]<=np.absolute(a[4]-a[5])<acc[b+1] ] for c in conf] for b in xrange(len(acc)-1)]
	
		estimatesTrain = [[[a[0:2] for a in b] for b in c] for c in binData]
		confidencesTrain = [[[a[2:4] for a in b] for b in c] for c in binData]
		competencesTrain = [[[a[4:6] for a in b] for b in c] for c in binData]
		truthsTrain = [[[a[6:8] for a in b] for b in c] for c in binData]
	
		correctBest = [[pfu.performCalcHigher(estimatesTrain[i][j],np.array(truthsTrain[i][j]),competencesTrain[i][j])[1] for j in xrange(len(conf))] for i in xrange(len(acc)-1)]
		correctConf = [[pfu.performCalcHigher(estimatesTrain[i][j],np.array(truthsTrain[i][j]),confidencesTrain[i][j])[1] for j in xrange(len(conf))] for i in xrange(len(acc)-1)]

	else:

		opins = valid[:,0::6]
		confs = valid[:,1::6]
		opiConf = np.append(opins,confs,axis=1) # [opi1 opi2 conf1 conf2] 
		opiConfComp = np.append(opiConf,compValid, axis=1) # [opi1 opi2 conf1 conf2 comp1 comp2] 
		dataAndTruths = np.append(opiConfComp,validT, axis=1) # [opi1 opi2 conf1 conf2 comp1 comp2 truthNeg truthPos] 

		binData = [[[ a for a in dataAndTruths if np.absolute(a[2]-a[3])==c and acc[b]<=np.absolute(a[4]-a[5])<acc[b+1] ] for c in conf] for b in xrange(len(acc)-1)]
	
		estimatesValid = [[[a[0:2] for a in b] for b in c] for c in binData]
		confidencesValid = [[[a[2:4] for a in b] for b in c] for c in binData]
		competencesValid = [[[a[4:6] for a in b] for b in c] for c in binData]
		truthsValid = [[[a[6:8] for a in b] for b in c] for c in binData]

		correctBest = [[pfu.performCalcHigher(estimatesValid[i][j],np.array(truthsValid[i][j]),competencesValid[i][j])[1] for j in xrange(len(conf))] for i in xrange(len(acc)-1)]
		correctConf = [[pfu.performCalcHigher(estimatesValid[i][j],np.array(truthsValid[i][j]),confidencesValid[i][j])[1] for j in xrange(len(conf))] for i in xrange(len(acc)-1)]


	perfMatrices = {
		'acc': acc,
		'conf': conf,
		'correctBest': correctBest,
		'correctConf': correctConf
		}

	return perfMatrices
	



def accVsConf2Best(propCombin=0.1,bin=0.1,propTrain=0.5,propVal=0.5):

	groupSize = 2
	data = rawDataLoader(groupSize,propCombin,propTrain,propVal, returnValue = 'yes')
	#data=pickle.load( open( 'dataComp.pkl', 'rb' ) )

	train = data[0][0]
	trainT = data[0][1]
	compTrain = data[0][2]
	valid = data[0][3]
	validT = data[0][4]
	compValid = data[0][5]
	test = data[0][6]
	testT = data[0][7]
	compTest = data[0][8]

	acc = np.arange(0,1+bin,bin) #accuracy differences (youden in the training)
	acc[-1] = acc[-1]+0.0001 #for the 'less than' to include = 1.0 in the last bin
	conf = np.arange(0,4) #confidence differences
	
	#G MAYBE I SHOULD DO JUST THREE BINS: LOWER, EQUAL AND HIGHER

	if propVal == 0:

		opins = train[:,0::6]
		confs = train[:,1::6]
		opiConf = np.append(opins,confs,axis=1) # [opi1 opi2 conf1 conf2] 
		opiConfComp = np.append(opiConf,compTrain, axis=1) # [opi1 opi2 conf1 conf2 comp1 comp2] 
		dataAndTruths = np.append(opiConfComp,trainT, axis=1) # [opi1 opi2 conf1 conf2 comp1 comp2 truthNeg truthPos] 

		binData12 = [[[ a for a in dataAndTruths if a[4]>=a[5] and a[3]-a[2]==c and acc[b]<=a[4]-a[5]<acc[b+1] ] for c in conf] for b in xrange(len(acc)-1)]
		binData21 = [[[ a for a in dataAndTruths if a[4]<a[5] and a[2]-a[3]==c and acc[b]<=a[5]-a[4]<acc[b+1] ] for c in conf] for b in xrange(len(acc)-1)]
		
		binData = [[ binData12[i][j]+binData21[i][j] for j in xrange(len(conf))] for i in xrange(len(acc)-1)]
		print [[len(a) for a in b] for b in binData]
		estimatesTrain = [[[a[0:2] for a in b] for b in c] for c in binData]
		confidencesTrain = [[[a[2:4] for a in b] for b in c] for c in binData]
		competencesTrain = [[[a[4:6] for a in b] for b in c] for c in binData]
		truthsTrain = [[[a[6:8] for a in b] for b in c] for c in binData]
	
		correctBest = [[pfu.performCalcHigher(estimatesTrain[i][j],np.array(truthsTrain[i][j]),competencesTrain[i][j])[1] for j in xrange(len(conf))] for i in xrange(len(acc)-1)]
		correctConf = [[pfu.performCalcHigher(estimatesTrain[i][j],np.array(truthsTrain[i][j]),confidencesTrain[i][j])[1] for j in xrange(len(conf))] for i in xrange(len(acc)-1)]

	else:

		opins = valid[:,0::6]
		confs = valid[:,1::6]
		opiConf = np.append(opins,confs,axis=1) # [opi1 opi2 conf1 conf2] 
		opiConfComp = np.append(opiConf,compValid, axis=1) # [opi1 opi2 conf1 conf2 comp1 comp2] 
		dataAndTruths = np.append(opiConfComp,validT, axis=1) # [opi1 opi2 conf1 conf2 comp1 comp2 truthNeg truthPos] 

		binData12 = [[[ a for a in dataAndTruths if a[4]>=a[5] and a[3]-a[2]==c and acc[b]<=a[4]-a[5]<acc[b+1] ] for c in conf] for b in xrange(len(acc)-1)]
		binData21 = [[[ a for a in dataAndTruths if a[4]<a[5] and a[2]-a[3]==c and acc[b]<=a[5]-a[4]<acc[b+1] ] for c in conf] for b in xrange(len(acc)-1)]
		
		binData = [[ binData12[i][j]+binData21[i][j] for j in xrange(len(conf))] for i in xrange(len(acc)-1)]
		print [[len(a) for a in b] for b in binData]
		estimatesValid = [[[a[0:2] for a in b] for b in c] for c in binData]
		confidencesValid = [[[a[2:4] for a in b] for b in c] for c in binData]
		competencesValid = [[[a[4:6] for a in b] for b in c] for c in binData]
		truthsValid = [[[a[6:8] for a in b] for b in c] for c in binData]
	
		correctBest = [[pfu.performCalcHigher(estimatesValid[i][j],np.array(truthsValid[i][j]),competencesValid[i][j])[1] for j in xrange(len(conf))] for i in xrange(len(acc)-1)]
		correctConf = [[pfu.performCalcHigher(estimatesValid[i][j],np.array(truthsValid[i][j]),confidencesValid[i][j])[1] for j in xrange(len(conf))] for i in xrange(len(acc)-1)]


	perfMatrices = {
		'acc': acc,
		'conf': conf,
		'correctBest': correctBest,
		'correctConf': correctConf
		}

	return perfMatrices
	



def accVsConf2Plotter(propCombin = 1.0,bin = 0.05):	

	#G propCombin: proportion of total combinations used
	#G propTrain, propValid: proportion of cases used for train, validation and test
	#G bin: binning of accuracy in the performance matrices	


	#G Method of |comp1-comp2| vs |conf1-conf2|, only training cases 
	perfMatrices1 = accVsConf2(propCombin,bin,propTrain=1.0,propVal=0.0)

	acc1 = perfMatrices1['acc']
	conf1 = perfMatrices1['conf']
	correctBest1 = perfMatrices1['correctBest']
	correctConf1 = perfMatrices1['correctConf']

	diffBestConf1 = [ [correctBest1[i][j]-correctConf1[i][j] for j in xrange(len(conf1))] for i in xrange(len(acc1)-1)]	

	
	#G Method of |comp1-comp2| vs |conf1-conf2|, training cases to compute competence, matrix over validation cases 
	perfMatrices2 = accVsConf2(propCombin,bin,propTrain=0.5,propVal=0.5)

	acc2 = perfMatrices2['acc']
	conf2 = perfMatrices2['conf']
	correctBest2 = perfMatrices2['correctBest']
	correctConf2 = perfMatrices2['correctConf']

	diffBestConf2 = [ [correctBest1[i][j]-correctConf2[i][j] for j in xrange(len(conf2))] for i in xrange(len(acc2)-1)]

	
	#G Method of comp1-comp2 vs conf1-conf2, only training cases, with comp1 and conf1 the numbers of the most competent of the two 
	perfMatrices3 = accVsConf2Best(propCombin,bin,propTrain=1.0,propVal=0.0)

	acc3 = perfMatrices3['acc']
	conf3 = perfMatrices3['conf']
	correctBest3 = perfMatrices3['correctBest']
	correctConf3 = perfMatrices3['correctConf']

	diffBestConf3 = [ [correctBest3[i][j]-correctConf3[i][j] for j in xrange(len(conf3))] for i in xrange(len(acc3)-1)]
	
	
	#G Method of comp1-comp2 vs conf1-conf2, train to compute competence and valid for matrix, with comp1 and conf1 the numbers of the most competent of the two 
	perfMatrices4 = accVsConf2Best(propCombin,bin,propTrain=0.5,propVal=0.5)

	acc4 = perfMatrices4['acc']
	conf4 = perfMatrices4['conf']
	correctBest4 = perfMatrices4['correctBest']
	correctConf4 = perfMatrices4['correctConf']

	diffBestConf4 = [ [correctBest4[i][j]-correctConf4[i][j] for j in xrange(len(conf4))] for i in xrange(len(acc4)-1)]

	
	toplot = [diffBestConf1, diffBestConf2, diffBestConf3, diffBestConf4]
	acc = [acc1, acc2, acc3, acc4]
	confs = [conf1, conf2, conf3, conf4]


	plt.close()
	fig = plt.figure(figsize=(14,11))
	plt.clf()
	'''
	plt.switch_backend('TkAgg')
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	'''
	sns.set_style('whitegrid',{'axes.grid' : False})

	gs = gridspec.GridSpec(2, 2)
	gs.update(left=0.09, right=0.97, bottom=0.08, top=0.94, wspace=0.25, hspace=0.3)


	titles = ['all cases, absolute differences', 'cases split, absolute differences', 'all cases, best - worst', 'cases split, best - worst'] #vs with a point?
	xticks = np.arange(0.5,4.5)
	xticklabels = ('0','1','2','3')
	xlabels = [r'|$\Delta$confidence|',r'|$\Delta$confidence|',r'-$\Delta$confidence',r'-$\Delta$confidence']
	ylabels = [r'|$\Delta$competence|',r'|$\Delta$competence|',r'$\Delta$competence',r'$\Delta$competence']
	tosubpl = [[[0,1],[0,1]],[[0,1],[1,2]],[[1,2],[0,1]],[[1,2],[1,2]]]

	for i in xrange(4):
		tsp = tosubpl[i]
		axi = plt.subplot(gs[tsp[0][0]:tsp[0][1],tsp[1][0]:tsp[1][1]])
		tpl = toplot[i]
		acci = acc[i]
		conf = confs[i]

		noNans = [acci[r+1] for r in xrange(len(tpl)) if not all(np.isnan(tpl[r])) ]
		accLims = [min(noNans)-bin, max(noNans)];

		binCbar = 0.05 #will be tha spacing in the colorbar, but I need to define it here to do the following:
		minPlot = max(-0.30,binCbar*np.floor(np.nanmin(np.array(tpl))/binCbar))
		maxPlot = min(0.35,binCbar*np.ceil(np.nanmax(np.array(tpl))/binCbar))
		
		spi = axi.pcolormesh(np.append(conf,conf[-1]+1),acci,np.ma.masked_invalid(np.asarray(tpl)), cmap='jet', vmin=minPlot, vmax=maxPlot)
		axi.set_ylim(accLims)
		axi.set_title(titles[i],fontsize=22)
		axi.set_xticks(xticks)
		axi.set_xticklabels(xticklabels)
		axi.tick_params(labelsize=16)
		axi.set_xlabel(xlabels[i],fontsize=20)
		axi.set_ylabel(ylabels[i],fontsize=20)
		
		cbari = plt.colorbar(spi)
		cbari.set_label(label = r'$\mathregular{p_{best} - p_{conf}}$', size = 20)
		cbari.set_ticks(np.arange(-1.0,1+binCbar,binCbar))
		cbari.ax.tick_params(labelsize = 14)

	
	plt.draw()
	#plt.show()
	plt.pause(1)
	plt.savefig('accVsConf2.tiff',dpi=75)







def checkNet():

	groupSize = 2
	#G proportion of total combinations used
	propCombin = 1.0
	#G proportion of cases used for train, validation and test
	propTrain = 0.5
	propVal = 0.5	
	data = rawDataLoader(groupSize,propCombin,propTrain,propVal,returnValue = 'yes')
	#data=pickle.load( open( 'dataComp.pkl', 'rb' ) )

	train = data[0][0]

	train = data[0][0]
	trainT = data[0][1]
	compTrain = data[0][2]
	valid = data[0][3]
	validT = data[0][4]
	compValid = data[0][5]
	test = data[0][6]
	testT = data[0][7]
	compTest = data[0][8]

	nconf = 4.0 #G confidence (originally in a 1 to 4 scale), if changed here, change accordingly in analyzers.py (same name for the variable)

	a1 = np.multiply(train[:,0::6],compTrain) #G opinion * competence
	b1 = train[:,1::6]/nconf  
	#a1 = np.multiply(train[:,0::6],1) #G opinion, competence not considered
	#b1 = 0*train[:,1::6] #G confidence not considered
	xTrain = np.append(a1,b1,axis=1)
	#G c1 = compTrain #G competence
	#G xTrain = np.append(xTrain,c1,axis=1) #G appends competence for further calculation of performance of best doctor

	if len(valid)>0:
		a2 = np.multiply(valid[:,0::6],compValid) #G opinion * competence IN THE TRAINING
		b2 = valid[:,1::6]/nconf
		#a2 = np.multiply(valid[:,0::6],1)
		#b2 = 0*valid[:,1::6]
		xValid = np.append(a2,b2,axis=1)
		#G c2 = compValid
		#G xValid = np.append(xValid,c2,axis=1)
	else: xValid = []

	if len(test)>0:
		a3 = np.multiply(test[:,0::6],compTest) #G opinion * competence IN THE TRAINING
		b3 = test[:,1::6]/nconf
		xTest = np.append(a3,b3,axis=1)
		#G c3 = compTest
		#G xTest = np.append(xTest,c3,axis=1)
	else: xTest = []

	yTrain = trainT
	yValid = validT
	yTest = testT


	wab=pickle.load( open( 'weightsAndBias3000.pkl', 'rb' ) ) # load a set of weights and bias. Ex: [wei1, b1, wei2, b2, wei3, b3]
								#[weights]=[[100],[100],[100],[100]],  [bias]=[100]
	Din = len(wab[0]) # dimension of input
	N1 = len(wab[1]) # dim of first layer
	N2 = len(wab[3]) # dim of second layer
	Dout = len(wab[5]) # number of classes


	#[( x*W1+b1 )W2 + b2 ]W3 + b3 #sketch of computation

	# [xi] = [len(acc1),len(acc2),len(layeri)]	
	x2 = [ relu6G( [np.sum( np.multiply(a ,[wab[0][d][n] for d in xrange(Din)] )) for n in xrange(N1)] + wab[1] ) for a in xTrain ] # x1*W2+b2
	x3 = [ relu6G( [np.sum( [x2[a][d]*wab[2][d][n] for d in xrange(N1)] ) for n in xrange(N2)] + wab[3] ) for a in xrange(len(xTrain))] # x2*W2+b2
	xo = [ [np.sum( [x3[a][d]*wab[4][d][n] for d in xrange(N2)] ) for n in xrange(Dout)] + wab[5] for a in xrange(len(xTrain))] # x3*W3+b3
	# logits:
	p = [np.exp(xo[a])/np.sum(np.exp(xo[a])) for a in xrange(len(xTrain))]
	# prob of positive:
	pposTrain = [p[a][1] for a in xrange(len(xTrain))]

	numPosTrain = np.sum([a[1]==1 for a in yTrain])
	numNegTrain = np.sum([a[0]==1 for a in yTrain])

	correctPosTrain = np.sum([yTrain[i][1]==1 and pposTrain[i]>0.5 for i in xrange(len(pposTrain))])
	correctNegTrain = np.sum([yTrain[i][0]==1 and pposTrain[i]<=0.5 for i in xrange(len(pposTrain))])

	sensitTrain = correctPosTrain/float(numPosTrain)
	specifTrain = correctNegTrain/float(numNegTrain)

	youdenTrain = sensitTrain + specifTrain - 1

	checkDict = {
		'p' : p,
		'pposTrain' : pposTrain,
		'sensitTrain' : sensitTrain,
		'specifTrain' : specifTrain,
		'youdenTrain' : youdenTrain
		}

	return checkDict
		
	
	

def netAnalyzer(c1,c2):

	# c1: confidence of first doctor
	# c2: confidence of second doctor

	wab=pickle.load( open( 'weightsAndBias3000.pkl', 'rb' ) ) # load a set of weights and bias. Ex: [wei1, b1, wei2, b2, wei3, b3]
								#[weights]=[[100],[100],[100],[100]],  [bias]=[100]
	Din = len(wab[0]) # dimension of input
	N1 = len(wab[1]) # dim of first layer
	N2 = len(wab[3]) # dim of second layer
	Dout = len(wab[5]) # number of classes

 
	bin = 0.01
	nconf = 4.0 # in netfunctionG5 confidences are divided by 4
	conf1 = c1/nconf
	conf2 = c2/nconf
	acc1 = np.arange(0,1+bin,bin) #accuracy of first doctor
	acc2 = np.arange(0,1+bin,bin) #accuracy of second doctor

	#[( x*W1+b1 )W2 + b2 ]W3 + b3 #sketch of computation

	# [xi] = [len(acc1),len(acc2),len(layeri)]	
	x2 = [[ relu6G( [np.sum( np.multiply([-a1,a2,conf1,conf2] ,[wab[0][d][n] for d in xrange(Din)] )) for n in xrange(N1)] + wab[1] ) for a2 in acc2] for a1 in acc1] # x1*W2+b2
	x3 = [[ relu6G( [np.sum( [x2[a1][a2][d]*wab[2][d][n] for d in xrange(N1)] ) for n in xrange(N2)] + wab[3] ) for a2 in xrange(len(acc2))] for a1 in xrange(len(acc1))] # x2*W2+b2
	xo = [[ [np.sum( [x3[a1][a2][d]*wab[4][d][n] for d in xrange(N2)] ) for n in xrange(Dout)] + wab[5] for a2 in xrange(len(acc2))] for a1 in xrange(len(acc1))] # x3*W3+b3
	
	# logits:
	p = [[np.exp(xo[a1][a2])/np.sum(np.exp(xo[a1][a2])) for a2 in xrange(len(acc2))] for a1 in xrange(len(acc1))]
	# prob of positive:
	ppos = [[p[a1][a2][1] for a2 in xrange(len(acc2))] for a1 in xrange(len(acc1))]

	pickle.dump( ppos , open( 'ppos' + str(c1) + str(c2) + '.pkl', 'wb' ) )
	
	#print x2[20][20]


def netAnzPlotter4(c1):

	plt.close()
	fig = plt.figure(figsize=(14,11))
	plt.clf()
	'''
	plt.switch_backend('TkAgg')
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	'''

	gs = gridspec.GridSpec(2, 2)
	gs.update(left=0.09, right=0.97, bottom=0.08, top=0.94, wspace=0.25, hspace=0.3)

	tosub = [[0,0],[0,1],[1,0],[1,1]]

	for i in xrange(4):

		c2 = i+1
		data = pickle.load(open( 'ppos' + str(c1) + str(c2) + '.pkl', 'rb' ) )
		len1 = len(data[0])
		len2 = len(data[1])

		axi = plt.subplot(gs[tosub[i][0], tosub[i][1]])
		#tocbar = {'label' : 'Probability of positive'}
		sepXticks = int(np.floor((len2-1)/4.0))
		sepYticks = int(np.floor((len1-1)/4.0))
		spi = axi.pcolormesh(np.asarray(data), cmap='jet', vmin = 0.0, vmax = 1.0)
	
		axi.set_xticks(range(0,len2,sepXticks))
		axi.set_xticklabels(['0.00','0.25','0.50','0.75','1.00'],fontsize=16)
		axi.set_yticks(range(0,len1,sepYticks))
		axi.set_yticklabels(['0.00','0.25','0.50','0.75','1.00'],fontsize=16)
		axi.set_xlabel('historic of the positive',fontsize=20)
		axi.set_ylabel('historic of the negative',fontsize=20)
		axi.set_xlim(left=0, right=len2)
		axi.set_ylim(bottom=0, top=len1)
		totit = 'Confidences: negative ' + str(int(c1)) + ', positive ' + str(int(c2))
		axi.set_title(totit,fontsize=22)
		#axi.tick_params{}
	
		cbari = plt.colorbar(spi)
		cbari.set_label(label = 'probability of positive', size = 20)
		binCbar = 0.25
		cbari.set_ticks(np.arange(0,1+binCbar,binCbar))
		cbari.ax.tick_params(labelsize = 14)


	plt.draw()
	#plt.show()
	plt.pause(2)
	plt.savefig('probs4_' + str(c1) + '.tiff',dpi=100)
	#plt.close()




def netAnzPlotter(c1,c2):

	plt.close()
	fig = plt.figure(figsize=(14,11))
	plt.clf()
	'''
	plt.switch_backend('TkAgg')
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	'''

	gs = gridspec.GridSpec(1, 1)
	gs.update(left=0.09, right=0.97, bottom=0.08, top=0.94, wspace=0.25, hspace=0.3)

	data = pickle.load(open( 'ppos' + str(c1) + str(c2) + '.pkl', 'rb' ) )
	len1 = len(data[0])
	len2 = len(data[1])

	ax1 = plt.subplot(gs[0,0])
	#tocbar = {'label' : 'Probability of positive'}
	sepXticks = int(np.floor((len2-1)/4.0))
	sepYticks = int(np.floor((len1-1)/4.0))
	sp1 = ax1.pcolormesh(np.asarray(data), cmap='jet', vmin = 0.0, vmax = 1.0)

	ax1.set_xticks(range(0,len2,sepXticks))
	ax1.set_xticklabels(['0.00','0.25','0.50','0.75','1.00'],fontsize=16)
	ax1.set_yticks(range(0,len1,sepYticks))
	ax1.set_yticklabels(['0.00','0.25','0.50','0.75','1.00'],fontsize=16)
	ax1.set_xlabel('historic of the positive',fontsize=24)
	ax1.set_ylabel('historic of the negative',fontsize=25)
	ax1.set_xlim(left=0, right=len2)
	ax1.set_ylim(bottom=0, top=len1)
	totit = 'Confidences: negative ' + str(int(c1)) + ', positive ' + str(int(c2))
	ax1.set_title(totit,fontsize=25)
	#ax1.tick_params{}

	cbar1 = plt.colorbar(sp1)
	cbar1.set_label(label = 'probability of positive', size = 24)
	binCbar = 0.25
	cbar1.set_ticks(np.arange(0,1+binCbar,binCbar))
	cbar1.ax.tick_params(labelsize = 14)


	plt.draw()
	#plt.show()
	plt.pause(2)
	plt.savefig('probs_' + str(c1) + str(c2) + '.tiff',dpi=100)
	#plt.close()




def relu6G(x):
	minRel = np.maximum(0, x) #minimum of the relu is zero
	rel = np.minimum(minRel, 6) #relu truncated at 6
	return rel




def norm_division_G(x,y):
   	try:
      		l = float(x)/float(x+y)
	except ZeroDivisionError:
      		l = float(np.nan)
   	return l
	














