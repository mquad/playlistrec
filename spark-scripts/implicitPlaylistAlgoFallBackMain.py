__author__ = 'robertopagano'


def extractImplicitPlaylists(train, config):
    validSessions = getValidSessionIds(train, config["algo"]["props"]["sessionJaccardShrinkage"],
                                       config["algo"]["props"]["clusterSimilarityThreshold"])
    playlists = getImplicitPlaylists(train, validSessions)
    return playlists


def joinRecommendations(recAlgoMain, recAlgoFallBack, config):
    if recAlgoFallBack is None:
        return recAlgoMain
    recAlgoMainDict = json.loads(recAlgoMain)
    recAlgoFallBackDict = json.loads(recAlgoFallBack)
    finalRecomm = copy.deepcopy(recAlgoMainDict)
    mainRecListSize = len(finalRecomm["linkedinfo"]["response"])
    if mainRecListSize < config["split"]["reclistSize"]:
        recommendedTracksMain = set()
        lastRank = 0
        for recMain in recAlgoMainDict["linkedinfo"]["response"]:
            recommendedTracksMain.add(recMain["id"])
            lastRank = recMain["rank"]
        for recFallBack in recAlgoFallBackDict["linkedinfo"]["response"]:
            trackId = recFallBack["id"]
            if trackId in recommendedTracksMain:
                continue
            lastRank += 1
            newRec = {}
            newRec["type"] = "track"
            newRec["id"] = trackId
            newRec["rank"] = lastRank
            finalRecomm["linkedinfo"]["response"].append(copy.deepcopy(newRec))
            if len(finalRecomm["linkedinfo"]["response"]) == config["split"]["reclistSize"]:
                break
    return json.dumps(finalRecomm)


def computeSAGHFallback(train, test, conf):
    saghConf = copy.deepcopy(conf)
    saghConf['algo']['name'] = saghConf['algo']['props']['fallbackAlgoName']
    saghConf['algo']['props'] = copy.deepcopy(conf['algo']['props']['fallbackAlgoProps'])

    artistLookupRDD = loadArtistLookup(saghConf)

    batchTrainingRDD = (train
                        .flatMap(lambda x: ext(json.loads(x))).filter(lambda x: x[1] >
                                                                                saghConf['algo']['props']['skipTh'])
                        .map(lambda x: (int(x[0]), int(x[2]))))
    recReqRDD = parseRequests(artistLookupRDD, test, saghConf['algo']['props']['skipTh'], saghConf)
    predictedTracksFallBack = generateRecommendationsSAGH(batchTrainingRDD, recReqRDD, artistLookupRDD, test,
                                                          saghConf['algo']['props']['numGH'], saghConf)
    return predictedTracksFallBack


def computeCAGHFallback(train, test, conf):
    caghConf = copy.deepcopy(conf)
    caghConf['algo']['name'] = conf['algo']['props']['fallbackAlgoName']
    caghConf['algo']['props'] = copy.deepcopy(conf['algo']['props']['fallbackAlgoProps'])
    th = caghConf['algo']['props']['skipTh']

    artistLookupRDD = loadArtistLookup(caghConf)
    batchTrainingRDD = (train
                        .flatMap(lambda x: ext(json.loads(x))).filter(lambda x: x[1] > th)
                        .map(lambda x: (int(x[0]), int(x[2]))))
    recReqRDD = parseRequests(artistLookupRDD, test, th, caghConf)
    artistArtistSim = computeArtistArtistSimMat(artistLookupRDD, batchTrainingRDD)
    artistGreatistHitsRDD = extractArtistGreatestHits(artistLookupRDD, batchTrainingRDD, caghConf)
    return generateRecommendationsCAGH(artistArtistSim, artistGreatistHitsRDD, recReqRDD, test, caghConf)


def executeImplicitPlaylistAlgoFallBack(playlists, predictedTracksFallBack, test, config):
    sessionTrackRDD = test.flatMap(lambda line: decay(line, config["algo"]["props"]["expDecayFactor"]))
    predictedPlaylistRDD = matmul_join(sessionTrackRDD, playlists)
    # se io non ho una track nelle mie playlist implicite mi perdo roba
    playlistTracksRDD = playlists.map(lambda x: (x[0][0], x[0][1])).groupByKey()
    # maps playlist id to its tracks
    predictedTracksRDD = predictedPlaylistRDD.map(lambda x: (x[0][1], (x[0][0], x[1]))) \
        .join(playlistTracksRDD).map(lambda x: ((x[1][0][0], x[1][1]), x[1][0][1]))
    sessionPlaylistWeightedRDD = predictedTracksRDD.map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey()
    predictedTraksWithProfile = sessionTrackRDD.map(lambda x: x[0]).groupByKey().join(sessionPlaylistWeightedRDD)
    if config["split"]["excludeAlreadyListenedTest"] != 0:
        predictedTracks = predictedTraksWithProfile.map(
            lambda x: (x[0], bestTracks(x[1][1], x[1][0], config["split"]["reclistSize"])))
    else:
        predictedTracks = predictedTraksWithProfile.map(
            lambda x: (x[0], bestTracks(x[1][1], [], config["split"]["reclistSize"])))
    predictedTracksImplicit = predictedTracks.map(lambda x: recToJson(x))
    predictedTracksImplicitToBeJoined = predictedTracksImplicit.map(lambda x: (int(json.loads(x)["id"]), x))
    predictedTracksFallBackToBeJoined = predictedTracksFallBack.map(lambda x: (int(json.loads(x)["id"]), x))
    return predictedTracksImplicitToBeJoined.leftOuterJoin(predictedTracksFallBackToBeJoined).map(
        lambda x: joinRecommendations(x[1][0], x[1][1], conf))
