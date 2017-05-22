
import sys
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('GtkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats as ss
import smoothers as smo
import cPickle as pickle
import seaborn as sns




def oneIterPlotter(scoresDict):

	#scoresDict=pickle.load( open( 'toPlot.pkl', 'rb' ) )
        netLoss = scoresDict['netLoss']
        netPerf = scoresDict['netPerf']
	perf_Acc = scoresDict['perf_Acc']
        perf_Youd = scoresDict['perf_Youd']
	perf_Adj = scoresDict['perf_Adj']
        perf_Maj = scoresDict['perf_Maj']
	perf_Opt = scoresDict['perf_Opt']
	perf_Conf = scoresDict['perf_Conf']

	lenLoss = len(netLoss[0])

	plt.close()
	fig = plt.figure
	plt.switch_backend('TkAgg')
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())

	# Loss functions
	plt.subplot(221)
	plt.plot(netLoss[0,:],'b-', linewidth=3.0, label='training')
	plt.plot(netLoss[1,:], 'r-', linewidth=3.0, label='validation')
	plt.legend()
	axes = plt.gca()
	xlims = axes.get_xlim()
	plt.xlim(xlims[0],1.025*xlims[1])
	#plt.ylim(0.08,0.12)
	plt.xlabel('num. epochs / skip', fontsize = 18)
	plt.ylabel('Loss', fontsize = 20)


	# Performance
	plt.subplot(222)
	plt.plot(netPerf[0,1,:lenLoss],'b-',linewidth=3.0, label='network tra')
	plt.plot(netPerf[1,1,:lenLoss], 'r-',linewidth=3.0, label='network val')

	axes = plt.gca()
	xlims = axes.get_xlim()
    	#G indexes in the following lines: [training or validation][group number][acc, youd, sens, or spec]
	bacc_tr = np.mean([perf_Adj[0][i][1] for i in xrange(len(perf_Adj[0]))])
	bacc_va = np.mean([perf_Adj[1][i][1] for i in xrange(len(perf_Adj[1]))])
	plt.plot([xlims[0],1.025*xlims[1]],[bacc_tr, bacc_tr],'g-',linewidth=3.0, label='best tra')
	plt.plot([xlims[0],1.025*xlims[1]],[bacc_va, bacc_va],'m-',linewidth=3.0, label='best val')
	
	mconf_tr = np.mean([perf_Conf[0][i][1] for i in xrange(len(perf_Conf[0]))])
	mconf_va = np.mean([perf_Conf[1][i][1] for i in xrange(len(perf_Conf[1]))])
	plt.plot([xlims[0],1.025*xlims[1]],[mconf_tr, mconf_tr],'g--',linewidth=3.0, label='confident tra')
	plt.plot([xlims[0],1.025*xlims[1]],[mconf_va, mconf_va],'m--',linewidth=3.0, label='confident val')

	maj_tr = np.mean([perf_Opt[0][i][1] for i in xrange(len(perf_Opt[0]))])
	maj_va = np.mean([perf_Opt[1][i][1] for i in xrange(len(perf_Opt[1]))])
	plt.plot([xlims[0],1.025*xlims[1]],[maj_tr, maj_tr],'g:',linewidth=3.0, label='optimistic tra')
	plt.plot([xlims[0],1.025*xlims[1]],[maj_va, maj_va],'m:',linewidth=3.0, label='optimistic val')

	plt.plot(netPerf[0,1,:lenLoss],'b-',linewidth=3.0) # repeat this so it's on top
	plt.plot(netPerf[1,1,:lenLoss], 'r-',linewidth=3.0)
	plt.legend(loc = 4, ncol = 4)
	plt.xlim(xlims[0],1.025*xlims[1])
	plt.ylim(0.50,0.90)
	#plt.ylim(0.5,1.01)
	plt.xlabel('num. epochs / skip', fontsize = 18)
	plt.ylabel('Performance', fontsize = 20)


	# sensitivity
	plt.subplot(223)
	plt.plot(netPerf[0,2,:lenLoss],'b-',linewidth=3.0, label='network tra')
	plt.plot(netPerf[1,2,:lenLoss], 'r-',linewidth=3.0, label='network val')

	axes = plt.gca()
	xlims = axes.get_xlim()

	bsens_tr = np.mean([perf_Adj[0][i][2] for i in xrange(len(perf_Adj[0]))])
	bsens_va = np.mean([perf_Adj[1][i][2] for i in xrange(len(perf_Adj[1]))])
	plt.plot([xlims[0],1.025*xlims[1]],[bsens_tr, bsens_tr],'g-',linewidth=3.0, label='best tra')
	plt.plot([xlims[0],1.025*xlims[1]],[bsens_va, bsens_va],'m-',linewidth=3.0, label='best val')

	mconf_tr = np.mean([perf_Conf[0][i][2] for i in xrange(len(perf_Conf[0]))])
	mconf_va = np.mean([perf_Conf[1][i][2] for i in xrange(len(perf_Conf[1]))])
	plt.plot([xlims[0],1.025*xlims[1]],[mconf_tr, mconf_tr],'g--',linewidth=3.0, label='confident tra')
	plt.plot([xlims[0],1.025*xlims[1]],[mconf_va, mconf_va],'m--',linewidth=3.0, label='confident val')

	maj_tr = np.mean([perf_Opt[0][i][2] for i in xrange(len(perf_Opt[0]))])
	maj_va = np.mean([perf_Opt[1][i][2] for i in xrange(len(perf_Opt[1]))])
	plt.plot([xlims[0],1.025*xlims[1]],[maj_tr, maj_tr],'g:',linewidth=3.0, label='optimistic tra')
	plt.plot([xlims[0],1.025*xlims[1]],[maj_va, maj_va],'m:',linewidth=3.0, label='optimistic val')
	
	plt.plot(netPerf[0,2,:lenLoss],'b-',linewidth=3.0) # repeat this so it's on top
	plt.plot(netPerf[1,2,:lenLoss], 'r-',linewidth=3.0)
	plt.legend(loc = 4, ncol = 4)
	plt.xlim(xlims[0],1.025*xlims[1])
	plt.ylim(0.70,1.00)
	#plt.ylim(0.0,1.01)
	plt.xlabel('num. epochs / skip', fontsize = 18)
	plt.ylabel('Sensitivity', fontsize = 20)


	# specificity
	plt.subplot(224)
	plt.plot(netPerf[0,3,:lenLoss],'b-',linewidth=3.0, label='network tra')
	plt.plot(netPerf[1,3,:lenLoss], 'r-',linewidth=3.0, label='network val')

	axes = plt.gca()
	xlims = axes.get_xlim()

	bsens_tr = np.mean([perf_Adj[0][i][3] for i in xrange(len(perf_Adj[0]))])
	bsens_va = np.mean([perf_Adj[1][i][3] for i in xrange(len(perf_Adj[1]))])
	plt.plot([xlims[0],1.025*xlims[1]],[bsens_tr, bsens_tr],'g-',linewidth=3.0, label='best tra')
	plt.plot([xlims[0],1.025*xlims[1]],[bsens_va, bsens_va],'m-',linewidth=3.0, label='best val')

	mconf_tr = np.mean([perf_Conf[0][i][3] for i in xrange(len(perf_Conf[0]))])
	mconf_va = np.mean([perf_Conf[1][i][3] for i in xrange(len(perf_Conf[1]))])
	plt.plot([xlims[0],1.025*xlims[1]],[mconf_tr, mconf_tr],'g--',linewidth=3.0, label='confident tra')
	plt.plot([xlims[0],1.025*xlims[1]],[mconf_va, mconf_va],'m--',linewidth=3.0, label='confident val')

	maj_tr = np.mean([perf_Opt[0][i][3] for i in xrange(len(perf_Opt[0]))])
	maj_va = np.mean([perf_Opt[1][i][3] for i in xrange(len(perf_Opt[1]))])
	plt.plot([xlims[0],1.025*xlims[1]],[maj_tr, maj_tr],'g:',linewidth=3.0, label='optimistic tra')
	plt.plot([xlims[0],1.025*xlims[1]],[maj_va, maj_va],'m:',linewidth=3.0, label='optimistic val')
	
	plt.plot(netPerf[0,3,:lenLoss],'b-',linewidth=3.0) # repeat this so it's on top
	plt.plot(netPerf[1,3,:lenLoss], 'r-',linewidth=3.0)
	plt.legend(loc = 4, ncol = 4)
	plt.xlim(xlims[0],1.025*xlims[1])
	plt.ylim(0.70,1.00)
	#plt.ylim(0.0,1.01)
	plt.xlabel('num. epochs / skip', fontsize = 18)
	plt.ylabel('Specificity', fontsize = 20)


	plt.subplots_adjust(bottom=0.06, right=.97, left=0.05, top=.97, wspace = 0.16, hspace=0.23)

	plt.draw()
	#plt.show()
	plt.pause(2)
	plt.savefig('figurenn.tiff',dpi=100)
	#plt.close()




def loopPlotter(loopDict):


	netPerf = loopDict['netPerf']
	perf_Maj = loopDict ['perf_Maj']
	perf_Youd = loopDict ['perf_Youd']
	#perf_Opt = loopDict['perf_Opt']
	perf_Conf = loopDict['perf_Conf']
	#woc = loopDict ['woc']


	plt.close()
	fig = plt.figure
	plt.switch_backend('TkAgg')
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())


	# accuracies
	sco = netPerf[1,:,0]

	toBarAcc = [np.mean(sco-perf_Youd[1,:,0]),np.mean(sco-perf_Conf[1,:,0]),np.mean(sco-perf_Maj[1,:,0])]
	toErrAcc = [ss.sem(sco-perf_Youd[1,:,0]),ss.sem(sco-perf_Conf[1,:,0]),ss.sem(sco-perf_Maj[1,:,0])]
	toLabAcc = ['best','confident','majority']

	sp1 = plt.subplot(121)
	sp1.bar(range(3),toBarAcc,yerr=toErrAcc,align='center')
	sp1.set_xticks(range(3))
	sp1.set_xticklabels(toLabAcc,fontsize=16)
	sp1.set_xlabel('method',fontsize=18)
	sp1.set_ylabel('accuracy',fontsize=20)
	#sp1.set_ylim(0,1)
	axes = plt.gca()
	xlims = axes.get_xlim()
	plt.plot([xlims[0],xlims[1]],[0,0],'k',linewidth=2.0)
	plt.plot([xlims[0],xlims[1]],[0,0],'k',linewidth=2.0)


	# Youden's index
	scy = netPerf[1,:,1]

	toBarYoud = [np.mean(scy-perf_Youd[1,:,1]),np.mean(scy-perf_Conf[1,:,1]),np.mean(scy-perf_Maj[1,:,1])]
	toErrYoud = [ss.sem(scy-perf_Youd[1,:,1]),ss.sem(scy-perf_Conf[1,:,1]),ss.sem(scy-perf_Maj[1,:,1])]
	toLabYoud = ['best','confident','majority']

	sp2 = plt.subplot(122)
	sp2.bar(range(3),toBarYoud,yerr=toErrYoud,align='center')
	sp2.set_xticks(range(3))
	sp2.set_xticklabels(toLabAcc,fontsize=16)
	sp2.set_xlabel('method',fontsize=18)
	sp2.set_ylabel('Youden''s index',fontsize=20)
	#sp2.set_ylim(0,1)
	axes = plt.gca()
	xlims = axes.get_xlim()
	plt.plot([xlims[0],xlims[1]],[0,0],'k',linewidth=2.0)
	plt.plot([xlims[0],xlims[1]],[0,0],'k',linewidth=2.0)


	plt.subplots_adjust(bottom=0.1, right=.9, left=0.1, top=.9, wspace = 0.2, hspace=0.5)

	plt.draw()
	#plt.show()
	plt.pause(2)
	plt.savefig('figurenn.tiff',dpi=50)
	#plt.close()




def figPlotter1(path, nIter, sel):
	# This plots the figure for Andres paper
	# subplot 1: a particular learning realization
	# average improvement of the network over several partitions

	plt.close()
	fig = plt.figure(figsize=(12,5))
	plt.clf()
	#plt.switch_backend('TkAgg')
	#mng = plt.get_current_fig_manager()
	#mng.resize(*mng.window.maxsize())

    	gs = gridspec.GridSpec(1, 5)
	gs.update(left=0.08, right=0.96, bottom=0.14, top=0.94, wspace=1.2)



	# left subplot
	# accuracies of one particular iteration (sel):

	pathSel = path + str(sel)

	scoresDict = pickle.load(open( pathSel + '.pkl', 'rb' ) )

	netPerf = scoresDict['netPerf']
	perf_Youd = scoresDict ['perf_Youd']
	perf_Maj = scoresDict['perf_Maj']
	perf_Conf = scoresDict['perf_Conf']

	lenNet = len(netPerf[1][0])
	
    	sp1 = plt.subplot(gs[0, :3])
	sp1.plot(netPerf[0,1,:lenNet],'b-',linewidth=3.0, label='network tra')
	sp1.plot(netPerf[1,1,:lenNet], 'r-',linewidth=3.0, label='network val')
	
	axes = plt.gca()
	xlims = axes.get_xlim()
    	#G indexes in the following lines: [training or validation][group number][acc, youd, sens, or spec]
	bacc_tr = np.mean([perf_Youd[0][i][1] for i in xrange(len(perf_Youd[0]))])
	bacc_va = np.mean([perf_Youd[1][i][1] for i in xrange(len(perf_Youd[1]))])
	sp1.plot([xlims[0],1.025*xlims[1]],[bacc_tr, bacc_tr],'g-',linewidth=3.0, label='best tra')
	sp1.plot([xlims[0],1.025*xlims[1]],[bacc_va, bacc_va],'m-',linewidth=3.0, label='best val')
	
	mconf_tr = np.mean([perf_Conf[0][i][1] for i in xrange(len(perf_Conf[0]))])
	mconf_va = np.mean([perf_Conf[1][i][1] for i in xrange(len(perf_Conf[1]))])
	sp1.plot([xlims[0],1.025*xlims[1]],[mconf_tr, mconf_tr],'g--',linewidth=3.0, label='confident tra')
	sp1.plot([xlims[0],1.025*xlims[1]],[mconf_va, mconf_va],'m--',linewidth=3.0, label='confident val')

	maj_tr = np.mean([perf_Maj[0][i][1] for i in xrange(len(perf_Maj[0]))])
	maj_va = np.mean([perf_Maj[1][i][1] for i in xrange(len(perf_Maj[1]))])
	sp1.plot([xlims[0],1.025*xlims[1]],[maj_tr, maj_tr],'g:',linewidth=3.0, label='majority tra')
	sp1.plot([xlims[0],1.025*xlims[1]],[maj_va, maj_va],'m:',linewidth=3.0, label='majority val')

	sp1.plot(netPerf[0,1,:lenNet],'b-',linewidth=3.0) # repeat this so it's on top
	sp1.plot(netPerf[1,1,:lenNet], 'r-',linewidth=3.0)

	ha,la = sp1.get_legend_handles_labels()
	handles = [ha[0], ha[2], ha[4], ha[6], ha[1], ha[3], ha[5], ha[7]]
	labels = [la[0], la[2], la[4], la[6], la[1], la[3], la[5], la[7]]
	sp1.legend(handles, labels, loc = 4, ncol = 2)

	sp1.set_xlim(xlims[0],1.025*xlims[1])
	#sp1.set_ylim(0.77,0.93)
	#sp1.set_ylim(0.5,1.01)
	ylims = axes.get_ylim()
	sp1.set_ylim(0.975*ylims[0],1.005*ylims[1])
	sp1.set_xlabel('num. epochs / skip', fontsize = 18)
	sp1.set_ylabel('Youden''s index', fontsize = 20)



	# right subplot
	# average improving of network

	netPerf = np.zeros([2,nIter,5]) #G performance in training and validation
	perf_Youd = np.zeros([2,nIter,5]) # performance of most competent in training and validation
	perf_Maj = np.zeros([2,nIter,5]) # performance of choosing the majority strategy in training and validation
	perf_Conf = np.zeros([2,nIter,5]) # performance the most confident in training and validation
	
	widthW = (lenNet+1)/20 #G width of the mooving window for smoothing the loss
	for q in range(nIter):
		pathIter = path + str(q+1)
		scoresDict = pickle.load(open( pathIter + '.pkl', 'rb' ) )

		netLoss = scoresDict['netLoss']
		smooLoss = smo.movingWindow(netLoss[1],widthW)
		pos = np.argmin(smooLoss)
		netPerf[:,q,:] = [scoresDict['netPerf'][0,:,pos], scoresDict['netPerf'][1,:,pos]]
		'''
		if q == 25-1:
			Lo = netLoss[1]
			smLo = smooLoss
		'''
		pM = scoresDict['perf_Maj']
		perf_Maj[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		pM = scoresDict['perf_Youd']
		perf_Youd[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		pM = scoresDict['perf_Conf']
		perf_Conf[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		
	scy = netPerf[1,:,1]
	toBarYoud = [np.mean(scy-perf_Youd[1,:,1]),np.mean(scy-perf_Conf[1,:,1]),np.mean(scy-perf_Maj[1,:,1])]
	toErrYoud = [ss.sem(scy-perf_Youd[1,:,1]),ss.sem(scy-perf_Conf[1,:,1]),ss.sem(scy-perf_Maj[1,:,1])]
	toLabYoud = ['best','confident','majority']
	print widthW
	print toBarYoud
	sp2 = plt.subplot(gs[0, 3:])
	sp2.bar(range(3),toBarYoud,yerr=toErrYoud,align='center')
	sp2.set_xticks(range(3))
	sp2.set_xticklabels(toLabYoud,fontsize=16)
	sp2.set_xlabel('heuristic',fontsize=18)
	sp2.set_ylabel('improvement over heuristic',fontsize=20)
	#sp2.set_ylim(0,1)
	axes = plt.gca()
	xlims = axes.get_xlim()


	plt.draw()
	#plt.show()
	plt.pause(1)
	plt.savefig('figNeural.tiff',dpi=250)
	#plt.close()

	'''
	plt.close()
	fig = plt.figure(figsize=(12,5))
	plt.plot(Lo)
	plt.plot(smLo,'r')
	plt.draw()
	#plt.show()
	plt.pause(2)
	'''




def figPlotter2(path, nIter, sel):
	# This plots the figure for Andres paper
	# subplot 1: a particular learning realization
	# average improvement of the network over several partitions

	plt.close()
	fig = plt.figure(figsize=(12,5))
	plt.clf()
	#plt.switch_backend('TkAgg')
	#mng = plt.get_current_fig_manager()
	#mng.resize(*mng.window.maxsize())

    	gs = gridspec.GridSpec(1, 5)
	gs.update(left=0.08, right=0.96, bottom=0.14, top=0.94, wspace=1.2)



	# left subplot
	# accuracies of one particular iteration (sel):

	pathSel = path + str(sel)

	scoresDict = pickle.load(open( pathSel + '.pkl', 'rb' ) )

	netPerf = scoresDict['netPerf']
	perf_Youd = scoresDict ['perf_Youd']
	perf_Maj = scoresDict['perf_Maj']
	perf_Conf = scoresDict['perf_Conf']

	lenNet = len(netPerf[1][0])
	
    	sp1 = plt.subplot(gs[0, :3])
	#sp1.plot(netPerf[0,1,:lenNet],'b-',linewidth=3.0, label='network tra')
	sp1.plot(netPerf[1,1,:lenNet], 'b-',linewidth=3.0, label='network')
	
	axes = plt.gca()
	xlims = axes.get_xlim()
    	#G indexes in the following lines: [training or validation][group number][acc, youd, sens, or spec]
	#bacc_tr = np.mean([perf_Youd[0][i][1] for i in xrange(len(perf_Youd[0]))])
	bacc_va = np.mean([perf_Youd[1][i][1] for i in xrange(len(perf_Youd[1]))])
	#sp1.plot([xlims[0],1.025*xlims[1]],[bacc_tr, bacc_tr],'g-',linewidth=3.0, label='best tra')
	sp1.plot([xlims[0],1.00*xlims[1]],[bacc_va, bacc_va],'g-',linewidth=3.0, label='best')
	
	#mconf_tr = np.mean([perf_Conf[0][i][1] for i in xrange(len(perf_Conf[0]))])
	mconf_va = np.mean([perf_Conf[1][i][1] for i in xrange(len(perf_Conf[1]))])
	#sp1.plot([xlims[0],1.025*xlims[1]],[mconf_tr, mconf_tr],'g--',linewidth=3.0, label='confident tra')
	sp1.plot([xlims[0],1.00*xlims[1]],[mconf_va, mconf_va],'r-',linewidth=3.0, label='confident')

	#maj_tr = np.mean([perf_Maj[0][i][1] for i in xrange(len(perf_Maj[0]))])
	maj_va = np.mean([perf_Maj[1][i][1] for i in xrange(len(perf_Maj[1]))])
	#sp1.plot([xlims[0],1.025*xlims[1]],[maj_tr, maj_tr],'g:',linewidth=3.0, label='majority tra')
	sp1.plot([xlims[0],1.00*xlims[1]],[maj_va, maj_va],'-',color='#DCB827', linewidth=3.0, label='majority')

	#sp1.plot(netPerf[0,1,:lenNet],'b-',linewidth=3.0) # repeat this so it's on top
	sp1.plot(netPerf[1,1,:lenNet], 'b-',linewidth=3.0)
	'''
	ha,la = sp1.get_legend_handles_labels()
	handles = [ha[0], ha[2], ha[4], ha[6], ha[1], ha[3], ha[5], ha[7]]
	labels = [la[0], la[2], la[4], la[6], la[1], la[3], la[5], la[7]]
	sp1.legend(handles, labels, loc = 4, ncol = 2)
	'''
	sp1.legend(loc = 2, ncol = 2, frameon=False)
	
	sp1.set_xticks(range(501)[0::100])
	sp1.set_xticklabels(range(2001)[0::500])
	#sp1.set_xlim(xlims[0],1.025*xlims[1])
	#sp1.set_ylim(0.77,0.93)
	#sp1.set_ylim(0.5,1.01)
	xlims = axes.get_xlim()
	xran = xlims[1]-xlims[0]
	sp1.set_xlim(0,400)
	#sp1.set_xlim(xlims[0]-0.0*xran,xlims[1]-0.00*xran)
	ylims = axes.get_ylim()
	yran = ylims[1]-ylims[0]
	sp1.set_ylim(maj_va-0.05*yran,ylims[1]+0.08*yran)
	sp1.set_xlabel('training epochs', fontsize = 18)
	sp1.set_ylabel("Youden's index", fontsize = 20)



	# right subplot
	# average improving of network

	netPerf = np.zeros([2,nIter,5]) #G performance in training and validation
	perf_Youd = np.zeros([2,nIter,5]) # performance of most competent in training and validation
	perf_Maj = np.zeros([2,nIter,5]) # performance of choosing the majority strategy in training and validation
	perf_Conf = np.zeros([2,nIter,5]) # performance the most confident in training and validation
	
	widthW = (lenNet+1)/40 #G width of the mooving window for smoothing the loss
	#widthW = 100
	for q in range(nIter):
		pathIter = path + str(q+1)
		scoresDict = pickle.load(open( pathIter + '.pkl', 'rb' ) )

		netLoss = scoresDict['netLoss']
		smooLoss = smo.movingWindow(netLoss[1],widthW)
		pos = np.argmin(smooLoss)
		netPerf[:,q,:] = [scoresDict['netPerf'][0,:,pos], scoresDict['netPerf'][1,:,pos]]
		
		if q == sel-1:
			'''
			ypos = scoresDict['netPerf'][1,1,pos]
			sp1.arrow(pos,ypos+0.18*yran,0,-0.16*yran, color='k', shape='full', lw=2, length_includes_head=True, head_width=0.02*lenNet, head_length=0.02*yran)
			'''
			print 'pos',pos
			
			Lo = netLoss[1]
			smLo = smooLoss
			
		pM = scoresDict['perf_Maj']
		perf_Maj[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		pM = scoresDict['perf_Youd']
		perf_Youd[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		pM = scoresDict['perf_Conf']
		perf_Conf[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		
	scy = netPerf[1,:,1]
	toBarYoud = [np.mean(scy-perf_Youd[1,:,1]),np.mean(scy-perf_Conf[1,:,1]),np.mean(scy-perf_Maj[1,:,1])]
	toErrYoud = [ss.sem(scy-perf_Youd[1,:,1]),ss.sem(scy-perf_Conf[1,:,1]),ss.sem(scy-perf_Maj[1,:,1])]
	toLabYoud = ['best','confident','majority']
	print 'width',widthW
	print toBarYoud
	print [np.mean(scy),np.mean(perf_Youd[1,:,1]),np.mean(perf_Conf[1,:,1]),np.mean(perf_Maj[1,:,1])]
	print [np.std(scy),np.std(perf_Youd[1,:,1]),np.std(perf_Conf[1,:,1]),np.std(perf_Maj[1,:,1])]
	print ss.ranksums(scy, perf_Youd[1,:,1])
	print ss.ranksums(scy, perf_Conf[1,:,1])
	print ss.ranksums(scy, perf_Maj[1,:,1])
	#print [np.sort(scy-perf_Youd[1,:,1]),np.sort(scy-perf_Conf[1,:,1]),np.sort(scy-perf_Maj[1,:,1])]
	
	sp2 = plt.subplot(gs[0, 3:])
	sp2.bar(range(3),toBarYoud,yerr=toErrYoud,align='center')
	sp2.set_xticks(range(3))
	sp2.set_xticklabels(toLabYoud,fontsize=16)
	sp2.set_xlabel('heuristic',fontsize=18)
	sp2.set_ylabel('improvement over heuristic',fontsize=20)
	#sp2.set_ylim(0,1)
	axes = plt.gca()
	xlims = axes.get_xlim()


	plt.draw()
	#plt.show()
	plt.pause(1)
	plt.savefig('figNeural2.png',dpi=250)
	plt.savefig('figNeural2.tiff',dpi=150)
	#plt.close()

	'''
	plt.close()
	fig = plt.figure(figsize=(12,5))
	plt.plot(Lo)
	plt.plot(smLo,'r')
	plt.draw()
	#plt.show()
	plt.pause(1)
	'''
	'''
	plt.close()
	fig = plt.figure(figsize=(12,5))
	plt.hist(scy)
	plt.draw()
	#plt.show()
	plt.pause(2)
	'''



def figPlotterMaj(path, nIter, sel):
	# This plots the figure for Andres paper
	# subplot 1: a particular learning realization
	# average improvement of the network over several partitions

	plt.close()
	fig = plt.figure(figsize=(12,5))
	plt.clf()
	#plt.switch_backend('TkAgg')
	#mng = plt.get_current_fig_manager()
	#mng.resize(*mng.window.maxsize())

    	gs = gridspec.GridSpec(1, 5)
	gs.update(left=0.08, right=0.96, bottom=0.14, top=0.94, wspace=1.2)



	# left subplot
	# accuracies of one particular iteration (sel):

	pathSel = path + str(sel)

	scoresDict = pickle.load(open( pathSel + '.pkl', 'rb' ) )

	netPerf = scoresDict['netPerf']
	perf_Youd = scoresDict ['perf_Youd']
	perf_Maj = scoresDict['perf_Maj']
	perf_Conf = scoresDict['perf_Conf']

	lenNet = len(netPerf[1][0])
	
    	sp1 = plt.subplot(gs[0, :3])
	#sp1.plot(netPerf[0,1,:lenNet],'b-',linewidth=3.0, label='network tra')
	sp1.plot(netPerf[1,1,:lenNet], 'b-',linewidth=3.0, label='network')
	
	axes = plt.gca()
	xlims = axes.get_xlim()
    	#G indexes in the following lines: [training or validation][group number][acc, youd, sens, or spec]
	#bacc_tr = np.mean([perf_Youd[0][i][1] for i in xrange(len(perf_Youd[0]))])
	bacc_va = np.mean([perf_Youd[1][i][1] for i in xrange(len(perf_Youd[1]))])
	#sp1.plot([xlims[0],1.025*xlims[1]],[bacc_tr, bacc_tr],'g-',linewidth=3.0, label='best tra')
	sp1.plot([xlims[0],1.00*xlims[1]],[bacc_va, bacc_va],'g-',linewidth=3.0, label='best')
	
	#mconf_tr = np.mean([perf_Conf[0][i][1] for i in xrange(len(perf_Conf[0]))])
	mconf_va = np.mean([perf_Conf[1][i][1] for i in xrange(len(perf_Conf[1]))])
	#sp1.plot([xlims[0],1.025*xlims[1]],[mconf_tr, mconf_tr],'g--',linewidth=3.0, label='confident tra')
	sp1.plot([xlims[0],1.00*xlims[1]],[mconf_va, mconf_va],'r-',linewidth=3.0, label='confident')

	#maj_tr = np.mean([perf_Maj[0][i][1] for i in xrange(len(perf_Maj[0]))])
	maj_va = np.mean([perf_Maj[1][i][1] for i in xrange(len(perf_Maj[1]))])
	#sp1.plot([xlims[0],1.025*xlims[1]],[maj_tr, maj_tr],'g:',linewidth=3.0, label='majority tra')
	sp1.plot([xlims[0],1.00*xlims[1]],[maj_va, maj_va],'-',color='#DCB827', linewidth=3.0, label='majority')

	#sp1.plot(netPerf[0,1,:lenNet],'b-',linewidth=3.0) # repeat this so it's on top
	sp1.plot(netPerf[1,1,:lenNet], 'b-',linewidth=3.0)
	'''
	ha,la = sp1.get_legend_handles_labels()
	handles = [ha[0], ha[2], ha[4], ha[6], ha[1], ha[3], ha[5], ha[7]]
	labels = [la[0], la[2], la[4], la[6], la[1], la[3], la[5], la[7]]
	sp1.legend(handles, labels, loc = 4, ncol = 2)
	'''
	sp1.legend(loc = 2, ncol = 2, frameon=False)
	
	sp1.set_xticks(range(2001)[0::100])
	sp1.set_xticklabels(range(10001)[0::500])
	#sp1.set_xlim(xlims[0],1.025*xlims[1])
	#sp1.set_ylim(0.77,0.93)
	#sp1.set_ylim(0.5,1.01)
	xlims = axes.get_xlim()
	xran = xlims[1]-xlims[0]
	sp1.set_xlim(0,600)
	#sp1.set_xlim(xlims[0]-0.0*xran,xlims[1]-0.00*xran)
	ylims = axes.get_ylim()
	yran = ylims[1]-ylims[0]
	sp1.set_ylim(np.min([bacc_va,mconf_va,maj_va])-0.005,np.max([bacc_va,mconf_va,maj_va])+0.02)
	sp1.set_xlabel('training epochs', fontsize = 18)
	sp1.set_ylabel("Youden's index", fontsize = 20)



	# right subplot
	# average improving of network

	netPerf = np.zeros([2,nIter,5]) #G performance in training and validation
	perf_Youd = np.zeros([2,nIter,5]) # performance of most competent in training and validation
	perf_Maj = np.zeros([2,nIter,5]) # performance of choosing the majority strategy in training and validation
	perf_Conf = np.zeros([2,nIter,5]) # performance the most confident in training and validation
	
	widthW = (lenNet+1)/40 #G width of the mooving window for smoothing the loss
	#widthW = 100
	for q in range(nIter):
		pathIter = path + str(q+1)
		scoresDict = pickle.load(open( pathIter + '.pkl', 'rb' ) )

		netLoss = scoresDict['netLoss']
		smooLoss = smo.movingWindow(netLoss[1],widthW)
		pos = np.argmin(smooLoss)
		netPerf[:,q,:] = [scoresDict['netPerf'][0,:,pos], scoresDict['netPerf'][1,:,pos]]
		
		if q == sel-1:
			'''
			ypos = scoresDict['netPerf'][1,1,pos]
			sp1.arrow(pos,ypos+0.18*yran,0,-0.16*yran, color='k', shape='full', lw=2, length_includes_head=True, head_width=0.02*lenNet, head_length=0.02*yran)
			'''
			print 'pos',pos
			
			Lo = netLoss[1]
			smLo = smooLoss
			
		pM = scoresDict['perf_Maj']
		perf_Maj[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		pM = scoresDict['perf_Youd']
		perf_Youd[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		pM = scoresDict['perf_Conf']
		perf_Conf[:,q,:] = [[np.mean([pM[0][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[0][0]))],[np.mean([pM[1][j][i] for j in xrange(len(pM[0]))]) for i in xrange(len(pM[1][0]))]]
		
	scy = netPerf[1,:,1]
	toBarYoud = [np.mean(scy-perf_Youd[1,:,1]),np.mean(scy-perf_Conf[1,:,1]),np.mean(scy-perf_Maj[1,:,1])]
	toErrYoud = [ss.sem(scy-perf_Youd[1,:,1]),ss.sem(scy-perf_Conf[1,:,1]),ss.sem(scy-perf_Maj[1,:,1])]
	toLabYoud = ['best','confident','majority']
	print 'width',widthW
	print toBarYoud
	print [np.mean(scy),np.mean(perf_Youd[1,:,1]),np.mean(perf_Conf[1,:,1]),np.mean(perf_Maj[1,:,1])]
	print [np.std(scy),np.std(perf_Youd[1,:,1]),np.std(perf_Conf[1,:,1]),np.std(perf_Maj[1,:,1])]
	print ss.ranksums(scy, perf_Youd[1,:,1])
	print ss.ranksums(scy, perf_Conf[1,:,1])
	print ss.ranksums(scy, perf_Maj[1,:,1])
	#print [np.sort(scy-perf_Youd[1,:,1]),np.sort(scy-perf_Conf[1,:,1]),np.sort(scy-perf_Maj[1,:,1])]
	
	sp2 = plt.subplot(gs[0, 3:])
	sp2.bar(range(3),toBarYoud,yerr=toErrYoud,align='center')
	sp2.set_xticks(range(3))
	sp2.set_xticklabels(toLabYoud,fontsize=16)
	sp2.set_xlabel('heuristic',fontsize=18)
	sp2.set_ylabel('improvement over heuristic',fontsize=20)
	#sp2.set_ylim(0,1)
	axes = plt.gca()
	xlims = axes.get_xlim()


	plt.draw()
	#plt.show()
	plt.pause(1)
	plt.savefig('figNeuralMaj.png',dpi=250)
	plt.savefig('figNeuralMaj.tiff',dpi=100)
	#plt.close()

	'''
	plt.close()
	fig = plt.figure(figsize=(12,5))
	plt.plot(Lo)
	plt.plot(smLo,'r')
	plt.draw()
	#plt.show()
	plt.pause(1)
	'''
	'''
	plt.close()
	fig = plt.figure(figsize=(12,5))
	plt.hist(scy)
	plt.draw()
	#plt.show()
	plt.pause(2)
	'''






