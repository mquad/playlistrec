__author__ = 'robertopagano'

execfile('../spark-scripts/conventions.py')
execfile('../spark-scripts/split.py')
execfile('../spark-scripts/utils.py')
execfile('../spark-scripts/eval.py')
execfile('../spark-scripts/CAGHFunctions.py')
execfile('../spark-scripts/CAGHMain.py')
execfile('../spark-scripts/SAGHFunctions.py')
execfile('../spark-scripts/SAGHMain.py')
execfile('../spark-scripts/implicitPlaylistAlgoFunctions.py')
execfile('../spark-scripts/implicitPlaylistAlgoFallBackMain.py')
execfile('../spark-scripts/implicitPlaylistAlgoMain.py')


import json
import copy
from pyspark import SparkContext, StorageLevel
from os import path

for excludeAlreadyListenedTest in [False]:

    for onlinetr_len in [1, 2, 5, 10, 20]:
    # for onlinetr_len in [1]:

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

        sc = SparkContext(appName="Music")

        splitter(conf)

        train, test = loadDataset(conf)
        train.persist(StorageLevel.MEMORY_AND_DISK)
        test.persist(StorageLevel.MEMORY_AND_DISK)

        ####CAGH
        artistLookupRDD = loadArtistLookup(conf)
        artistLookupRDD.persist(StorageLevel.MEMORY_AND_DISK)

        conf['algo'] = {}
        conf['algo']['name'] = 'CAGH'
        conf['algo']['props'] = {}

        numGHList = [50]
        minAASimList = [0.4]
        skipThList = [0]

        for th in skipThList:
            conf['algo']['props']['skipTh'] = th
            # (track_id, session_id)
            batchTrainingRDD = (train
                                .flatMap(lambda x: ext(json.loads(x))).filter(lambda x: x[1] > th)
                                .map(lambda x: (int(x[0]), int(x[2]))))
            recReqRDD = parseRequests(artistLookupRDD, test, th, conf)
            artistArtistSim = computeArtistArtistSimMat(artistLookupRDD, batchTrainingRDD)

            if len(numGHList) > 1:
                batchTrainingRDD.persist(StorageLevel.MEMORY_AND_DISK)
                recReqRDD.persist(StorageLevel.MEMORY_AND_DISK)
                artistArtistSim.persist(StorageLevel.MEMORY_AND_DISK)

            for numGH in numGHList:
                conf['algo']['props']['numGH'] = numGH
                artistGreatistHitsRDD = extractArtistGreatestHits(artistLookupRDD, batchTrainingRDD, conf)
                if len(minAASimList) > 1:
                    artistGreatistHitsRDD.persist(StorageLevel.MEMORY_AND_DISK)

                for minAASim in minAASimList:
                    conf['algo']['props']['minAASim'] = minAASim
                    recJsonRdd = generateRecommendationsCAGH(artistArtistSim, artistGreatistHitsRDD, recReqRDD, test,
                                                             conf)
                    try:
                        saveRecommendations(conf, recJsonRdd, overwrite=False)
                        computeMetrics(conf)
                    except Exception as e:
                        print e.message


        ####SAGH
        conf['algo'] = {}
        conf['algo']['name'] = 'SAGH'
        conf['algo']['props'] = {}

        numGHList = [100]
        skipThList = [0]

        for th in skipThList:
            conf['algo']['props']['skipTh'] = th
            batchTrainingRDD = (train
                                .flatMap(lambda x: ext(json.loads(x))).filter(lambda x: x[1] > th)
                                .map(lambda x: (int(x[0]), int(x[2]))))
            recReqRDD = parseRequests(artistLookupRDD, test, th, conf)

            if len(numGHList) > 1:
                batchTrainingRDD.persist(StorageLevel.MEMORY_AND_DISK)
                recReqRDD.persist(StorageLevel.MEMORY_AND_DISK)

            for numGH in numGHList:
                conf['algo']['props']['numGH'] = numGH
                recJsonRdd = generateRecommendationsSAGH(batchTrainingRDD, recReqRDD, artistLookupRDD, test, numGH,
                                                         conf)
                try:
                    saveRecommendations(conf, recJsonRdd, overwrite=False)
                    computeMetrics(conf)
                except Exception as e:
                    print e.message


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

                playlists = extractImplicitPlaylists(train, conf)
                if len(expDecayList) > 1:
                    playlists.persist(StorageLevel.MEMORY_AND_DISK)

                for expDecay in expDecayList:
                    conf['algo']['props']["expDecayFactor"] = expDecay
                    conf['algo']['name'] = 'ImplicitPlaylist_shk_%d_clustSim_%.3f_decay_%.3f' % \
                                           (sessionJaccardShrinkage, clusterSim, expDecay)

                    recJsonRDD = executeImplicitPlaylistAlgo(playlists, test, conf)
                    try:
                        saveRecommendations(conf, recJsonRDD, overwrite=False)
                        computeMetrics(conf)
                    except Exception as e:
                        print e.message

        sc.stop()
