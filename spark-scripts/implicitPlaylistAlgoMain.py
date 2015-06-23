__author__ = 'robertopagano'


def extractImplicitPlaylists(train, config):
    validSessions = getValidSessionIds(train, config["algo"]["props"]["sessionJaccardShrinkage"],
                                   config["algo"]["props"]["clusterSimilarityThreshold"])
    playlists = getImplicitPlaylists(train, validSessions)
    return playlists


def executeImplicitPlaylistAlgo(playlists, test, config):
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
    return predictedTracks.map(lambda x: recToJson(x))
