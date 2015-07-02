__author__ = 'robertopagano'

execfile('../spark-scripts/conventions.py')
execfile('../spark-scripts/split.py')
execfile('../spark-scripts/utils.py')
execfile('../spark-scripts/eval.py')

import json
import copy
from pyspark import SparkContext
from os import path

for excludeAlreadyListenedTest in [True, False]:

    # for onlinetr_len in [1, 2, 5, 10]:
    for onlinetr_len in [1]:

        conf = {}

        conf['split'] = {}
        conf['split']['reclistSize'] = 100
        conf['split']['callParams'] = {}
        conf['split']['excludeAlreadyListenedTest'] = excludeAlreadyListenedTest

        conf['split']['prop'] = {}

        conf['split']['minEventsPerUser'] = 5
        # conf['split'][
        #     'inputData'] = 's3n://contentwise-research-poli/split22.split/SenzaRipetizioni_nuovoEval1total_1413851857/'
        conf['split']['inputData'] = 's3n://contentwise-research-poli/30Mdataset/relations/sessions'
        conf['split']['bucketName'] = 'contentwise-research-poli'
        conf['split']['location'] = '30Mdataset/relations/sessions'
        # conf['split']['location'] = 'split22.split/SenzaRipetizioni_nuovoEval1total_1413851857'
        conf['split']['percUsTr'] = 0
        conf['split']['ts'] = int(0.75 * (1421745857 - 1390209860) + 1390209860) - 10000
        conf['split']['minEventPerSession'] = 5
        conf['split']['onlineTrainingLength'] = onlinetr_len
        conf['split']['GTlength'] = 5
        conf['split']['name'] = 'split_complete_ts_1413851857_repetitions_%d_plen_%d' % (not excludeAlreadyListenedTest, onlinetr_len)
        # conf['split']['name'] = 'SenzaRipetizioni_nuovoEval1total_1413851857'
        conf['split']['minEventPerSessionTraining'] = 10
        conf['split']['minEventPerSessionTest'] = 11
        conf['split']['mode'] = 'total'
        conf['split']['type'] = 'file'
        conf['split']['forceSplitCreation'] = False

        conf['evaluation'] = {}
        conf['evaluation']['metric'] = {}
        conf['evaluation']['metric']['type'] = 'recall'
        conf['evaluation']['metric']['prop'] = {}
        conf['evaluation']['metric']['prop']['N'] = [1, 2, 5, 10, 15, 20, 25, 50, 100]
        conf['evaluation']['name'] = 'recall@N'

        conf['general'] = {}
        # conf['general']['clientname'] = "split22.split"
        conf['general']['clientname'] = "complete.split"
        conf['general']['bucketName'] = 'contentwise-research-poli'
        conf['general']['tracksPath'] = '30Mdataset/entities/tracks.idomaar.gz'

        conf['split']['out'] = 's3n://contentwise-research-poli/%s/%s/' % (
        conf['general']['clientname'], conf['split']['name'])
        # conf['split']['out'] = 's3n://contentwise-research-poli/%s/%s/' % (
        #    conf['general']['clientname'], conf['split']['name'])

        sc = SparkContext(appName="Music Eval Test")

        ####CAGH
        conf['algo'] = {}
        conf['algo']['name'] = 'CAGH'
        conf['algo']['props'] = {}

        numGHList = [50]
        minAASimList = [0.4]
        skipThList = [0]

        for th in skipThList:
            conf['algo']['props']['skipTh'] = th
            for numGH in numGHList:
                conf['algo']['props']['numGH'] = numGH
                for minAASim in minAASimList:
                    conf['algo']['props']['minAASim'] = minAASim
                    computeMetrics(conf)


        ####SAGH
        conf['algo'] = {}
        conf['algo']['name'] = 'SAGH'
        conf['algo']['props'] = {}

        numGHList = [100]
        skipThList = [0]

        for th in skipThList:
            conf['algo']['props']['skipTh'] = th
            for numGH in numGHList:
                conf['algo']['props']['numGH'] = numGH
                computeMetrics(conf)


        ####Implicit
        conf['algo'] = {}
        conf['algo']['name'] = 'ImplicitPlaylist'
        conf['algo']['props'] = {}

        clusterSimList = [0.1]
        sessionJaccardShrinkageList = [5]
        expDecayList = [0.7]

        for sessionJaccardShrinkage in sessionJaccardShrinkageList:
            conf['algo']['props']["sessionJaccardShrinkage"] = sessionJaccardShrinkage

            for clusterSim in clusterSimList:
                conf['algo']['props']["clusterSimilarityThreshold"] = clusterSim

                for expDecay in expDecayList:
                    conf['algo']['props']["expDecayFactor"] = expDecay
                    conf['algo']['name'] = 'ImplicitPlaylist_shk_%d_clustSim_%.3f_decay_%.3f' % \
                                           (sessionJaccardShrinkage, clusterSim, expDecay)
                    computeMetrics(conf)

        sc.stop()
