__author__ = 'robertopagano'

import json
import sys
import numpy as np
import itertools


def getTrackCountForUser(trainLine):
    lineDict = json.loads(trainLine)
    l = []
    uid = lineDict["linkedinfo"]["subjects"][0]["id"]
    for x in lineDict["linkedinfo"]["objects"]:
        l.append(((uid, x["id"]), 1))
    return l


def getUserTrackCountRDD(train):
    return train.flatMap(lambda line: getTrackCountForUser(line)).reduceByKey(lambda x, y: x + y)


def getUserSessionTrackList(trainLine):
    lineDict = json.loads(trainLine)
    l = []
    uid = lineDict["linkedinfo"]["subjects"][0]["id"]
    sessionId = int(lineDict["id"])
    for x in lineDict["linkedinfo"]["objects"]:
        l.append(x["id"])
    return uid, (sessionId, l)


def filterSessionsByJ(tuple, threshold, bins=10):
    step = 1 / float(bins)
    res = []
    intra_session_dictionary = tuple[0]
    nsessions = tuple[1]

    res = set()
    filteredListByTh = []
    for key in intra_session_dictionary:
        if intra_session_dictionary[key] >= threshold:
            res.update(key)

    return res


def computeIntraSessionJaccardOriginalSessionId(list_of_sessions, shrinkage):
    user_session_dict = {}
    for session in list_of_sessions:
        session_id = session[0]
        track_list = session[1]
        S1 = set()
        for track in track_list:
            S1.add(track)
        user_session_dict[session_id] = {}
        user_session_dict[session_id]["track-set"] = S1

    jaccard_intra_sessions = {}
    nsessions = len(list_of_sessions)

    for key1 in user_session_dict:
        for key2 in user_session_dict:
            if key1 == key2:
                continue

            # jaccard_intra_sessions.has_key((key1,key2))
            if not (jaccard_intra_sessions.has_key((key1, key2))) and not (
            jaccard_intra_sessions.has_key((key2, key1))):
                jaccard_intra_sessions[(key1, key2)] = compute_jaccard_index(user_session_dict[key1]["track-set"],
                                                                             user_session_dict[key2]["track-set"],
                                                                             shrinkage)

    return jaccard_intra_sessions, nsessions


def compute_jaccard_index(set_1, set_2, shrinkage=10):
    n = len(set_1.intersection(set_2))
    den = float(len(set_1) + len(set_2) - n)
    if n == 0 or den == 0:
        return 0
    return n / float(len(set_1) + len(set_2) - n + shrinkage)


def getValidSessionIds(train, sessionJaccardShrinkage, clusterSimilarityThreshold):
    grouped_user_sessions = train.map(lambda line: getUserSessionTrackList(line))
    gusPerUserAggregated = grouped_user_sessions.groupByKey()
    validSessionIds = gusPerUserAggregated.map(
        lambda x: filterSessionsByJ(computeIntraSessionJaccardOriginalSessionId(x[1], sessionJaccardShrinkage),
                                    clusterSimilarityThreshold)).collect()

    validSessions = set()
    for sessions in validSessionIds:
        validSessions.update(sessions)

    return validSessions


def extractTracksListFromEvent(trainLine):
    lineDict = json.loads(trainLine)
    l = []
    for x in lineDict["linkedinfo"]["objects"]:
        l.append(x["id"])
    return l


def unrollPlaylists(x):
    t_list = x[0]
    pl_id = x[1]
    res = []
    for t in t_list:
        res.append(((pl_id, t), 1))
    return res


def getImplicitPlaylists(train, validSessions):
    events_session_filtered = train.filter(lambda line: int(json.loads(line)["id"]) in validSessions)
    return events_session_filtered.map(lambda line: extractTracksListFromEvent(line)).zipWithIndex().flatMap(
        lambda x: unrollPlaylists(x))


def decay(trainLine, factor=.5):
    lineDict = json.loads(trainLine)
    w = 1
    l = []
    uid = lineDict["linkedinfo"]["subjects"][0]["id"]
    sessionId = int(lineDict["id"])
    ts = int(lineDict["ts"])
    for x in lineDict["linkedinfo"]["objects"][::-1]:
        l.append((((uid, sessionId, trainLine), x["id"]), w))
        w *= factor
    return l


# computes the recommendation list, by picking the first N tracks from the top scored playlists
def bestTracks(pl_rec, excluded_tracks, N=100):
    excluded_tracks_set = set(excluded_tracks)
    l = list(pl_rec)
    l.sort(key=lambda x: float(x[1]), reverse=True)
    resList = []
    rank = 0
    for i in l:
        tracks = []
        for t in i[0]:
            if t not in excluded_tracks_set:
                tracks.append((t, rank))
                rank += 1
        resList.extend(tracks)
        if len(resList) >= N:
            break
    return resList[0:N]


# converts the recommended tracks into the required JSON format
def recToJson(x):
    finalDict = json.loads(x[0][2])
    finalDict["linkedinfo"]["response"] = []
    count = 0
    for t in x[1]:
        finalDict["linkedinfo"]["response"].append({})
        finalDict["linkedinfo"]["response"][count]["type"] = "track"
        finalDict["linkedinfo"]["response"][count]["id"] = t[0]
        finalDict["linkedinfo"]["response"][count]["rank"] = t[1]
        count += 1

    return json.dumps(finalDict)
