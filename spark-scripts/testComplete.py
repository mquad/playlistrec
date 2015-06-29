__author__ = 'robertopagano'



execfile('../spark-scripts/split.py')
execfile('../spark-scripts/utils.py')
execfile('../spark-scripts/eval.py')



import json
import copy
from pyspark import SparkContext
from os import path

for gtlen in [1,2,5,10]:
#for gtlen in [5]:

    conf = {}

    conf['split'] = {}
    conf['split']['reclistSize'] = 100
    conf['split']['callParams'] = {}
    conf['split']['excludeAlreadyListenedTest'] = True

    conf['split']['minEventsPerUser'] = 5
    #conf['split']['inputData'] = 's3n://contentwise-research-poli/split22.split/SenzaRipetizioni_nuovoEval5total_1413851857/'
    conf['split']['inputData'] = 's3n://contentwise-research-poli/30Mdataset/relations/sessions.idomaar'
    conf['split']['bucketName'] = 'contentwise-research-poli'
    conf['split']['percUsTr'] = 0.05
    conf['split']['ts'] = int(0.75 * (1421745857 - 1390209860) + 1390209860) - 10000
    conf['split']['minEventPerSession'] = 5
    conf['split']['onlineTrainingLength'] = 5
    conf['split']['GTlength'] = gtlen
    conf['split']['name'] = 'split_complete_ts_1413851857_no_repetitions_gt_' + str(gtlen)
    #conf['split']['name'] = 'SenzaRipetizioni_nuovoEval5total_1413851857'
    conf['split']['minEventPerSessionTraining'] = 10
    conf['split']['minEventPerSessionTest'] = 11
    conf['split']['mode'] = 'session'
    conf['split']['forceSplitCreation'] = False

    conf['evaluation'] = {}
    conf['evaluation']['metric'] = {}
    conf['evaluation']['metric']['type'] = 'recall'
    conf['evaluation']['metric']['prop'] = {}
    conf['evaluation']['metric']['prop']['N'] = [1,2,5,10,15,20,25,50,100]
    conf['evaluation']['name'] = 'recall@N'

    conf['general'] = {}
    #conf['general']['clientname'] = "split22.split"
    conf['general']['clientname'] = "complete.split"
    conf['general']['bucketName'] = 'contentwise-research-poli'
    conf['general']['tracksPath'] = '30Mdataset/entities/tracks.idomaar.gz'




    sc = SparkContext(appName="Music")

    splitter(conf)

    train,test = loadDataset(conf)
    train.cache()
    test.cache()
    artistLookupRDD = loadArtistLookup(conf)
    artistLookupRDD.cache()

    ####CAGH
    execfile('../spark-scripts/CAGHFunctions.py')
    execfile('../spark-scripts/CAGHMain.py')


    conf['algo'] = {}
    conf['algo']['name'] = 'CAGH'
    conf['algo']['props'] = {}
    #conf['algo']['props']['numGH'] = 10
    #conf['algo']['props']['minAASim'] = 0.5
    #conf['algo']['props']['skipTh'] = 0




    basePath = path.join("s3n://", conf['general']['bucketName'], conf['general']['clientname'])
    splitPath = path.join(basePath, conf['split']['name'])

    numGHList = [50]
    minAASimList = [0.4]
    skipThList = [0]

    for th in skipThList:
        conf['algo']['props']['skipTh'] = th
        # (track_id, session_id)
        batchTrainingRDD = (train
                            .flatMap(lambda x: ext(json.loads(x))).filter(lambda x: x[1] > th)
                            .map(lambda x: (int(x[0]), int(x[2])))
                            .cache())

        recReqRDD = parseRequests(artistLookupRDD, test, th, conf).cache()

        artistArtistSim = computeArtistArtistSimMat(artistLookupRDD, batchTrainingRDD).cache()
        for numGH in numGHList:
            conf['algo']['props']['numGH'] = numGH
            artistGreatistHitsRDD = extractArtistGreatestHits(artistLookupRDD, batchTrainingRDD, conf).cache()
            for minAASim in minAASimList:
                conf['algo']['props']['minAASim'] = minAASim
                recJsonRdd = generateRecommendationsCAGH(artistArtistSim, artistGreatistHitsRDD, recReqRDD, test, conf)
                try:
                    saveRecommendations(conf, recJsonRdd, overwrite=True)
                    computeMetrics(conf)
                except:
                    pass


    ####SAGH
    execfile('../spark-scripts/SAGHFunctions.py')
    execfile('../spark-scripts/SAGHMain.py')

    conf['algo'] = {}
    conf['algo']['name'] = 'SAGH'
    conf['algo']['props'] = {}
    #conf['algo']['props']['numGH'] = 100
    #conf['algo']['props']['skipTh'] = 0


    basePath = path.join("s3n://", conf['general']['bucketName'], conf['general']['clientname'])
    splitPath = path.join(basePath, conf['split']['name'])

    numGHList = [100]
    skipThList = [0]

    for th in skipThList:
        conf['algo']['props']['skipTh'] = th
        batchTrainingRDD = (train
                            .flatMap(lambda x: ext(json.loads(x))).filter(lambda x: x[1] > th)
                            .map(lambda x: (int(x[0]), int(x[2])))
                            .cache())
        recReqRDD = parseRequests(artistLookupRDD, test, th, conf).cache()

        for numGH in numGHList:
            conf['algo']['props']['numGH'] = numGH
            recJsonRdd = generateRecommendationsSAGH(batchTrainingRDD, recReqRDD, artistLookupRDD, test, numGH, conf)
            saveRecommendations(conf, recJsonRdd, overwrite=True)
            computeMetrics(conf)


    ####Implicit
    execfile('../spark-scripts/implicitPlaylistAlgoFunctions.py')
    execfile('../spark-scripts/implicitPlaylistAlgoFallBackMain.py')
    execfile('../spark-scripts/implicitPlaylistAlgoMain.py')

    conf['algo'] = {}
    conf['algo']['name'] = 'ImplicitPlaylist'
    conf['algo']['props'] = {}


    basePath = path.join("s3n://", conf['general']['bucketName'], conf['general']['clientname'])
    splitPath = path.join(basePath, conf['split']['name'])

    clusterSimList = [0.1]
    sessionJaccardShrinkageList = [5]
    expDecayList = [0.7]

    for sessionJaccardShrinkage in sessionJaccardShrinkageList:
        conf['algo']['props']["sessionJaccardShrinkage"] = sessionJaccardShrinkage

        for clusterSim in clusterSimList:
            conf['algo']['props']["clusterSimilarityThreshold"] = clusterSim

            playlists = extractImplicitPlaylists(train, conf).cache()

            for expDecay in expDecayList:
                conf['algo']['props']["expDecayFactor"] = expDecay
                conf['algo']['name'] = 'ImplicitPlaylist_shk_%d_clustSim_%.3f_decay_%.3f' % \
                    (sessionJaccardShrinkage, clusterSim, expDecay )

                recJsonRDD = executeImplicitPlaylistAlgo(playlists, test, conf)
                try:
                    saveRecommendations(conf, recJsonRDD, overwrite=True)
                    computeMetrics(conf)
                except:
                    pass


    sc.stop()