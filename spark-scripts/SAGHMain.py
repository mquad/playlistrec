__author__ = 'massimo'

import json
from operator import add


def addResponse(reqJson, respJson):
    req = json.loads(reqJson)
    req['linkedinfo'].update({'response': respJson})
    return json.dumps(req)


def formatResponse(rank_list, response_size=100):
    return [{'type': 'track', 'id': id, 'rank': rank} for rank, (id, _) in enumerate(rank_list[:response_size])]


def loadArtistLookup(conf):
    basePath = path.join('s3n://', conf['general']['bucketName'])
    trackPath = path.join(basePath, conf['general']['tracksPath'])
    trackRDD = sc.textFile(trackPath).cache()
    tabSplit = lambda x: x.split("\t")
    ext = lambda x: (int(x[1]), int(readJson(x)[0]['id']))
    artistLookupRDD = (trackRDD
                       .map(tabSplit)
                       .map(ext)
                       .distinct())
    return artistLookupRDD


def parseRequests(artistLookupRDD, test, th, conf):
    ext2 = lambda x: [(int(x['id']), int(l['id']), l['playratio']) for l in x['linkedinfo']['objects']]
    recReqRDD = (test
                 .flatMap(lambda x: ext2(json.loads(x)))
                 .filter(lambda x: x[2] > th)
                 .map(lambda x: (x[1], x[0]))
                 .join(artistLookupRDD)
                 .map(lambda x: ((x[1][1], x[1][0]), [(x[0])]))
                 .reduceByKey(add)
                 .map(lambda x: (x[0][0], (x[0][1], x[1]))))
    return recReqRDD


def generateRecommendationsSAGH(batchTrainingRDD, recReqRDD, artistLookupRDD, test, numGH, conf):
    joinedRDD = artistLookupRDD.join(batchTrainingRDD).map(lambda x: (x[1][1], (x[0], x[1][0])))
    # find greatest hits for each artist
    numGH = conf['algo']['props']['numGH']
    sort = lambda x: (x[0], sorter(x[1], numGH))
    uni = lambda x: x[1] > 1
    artistGreatistHitsRDD = (joinedRDD
                             .map(parser)
                             .reduceByKey(add)
                             .filter(uni)
                             .map(prep)
                             .reduceByKey(add)
                             .map(sort))
    joinedRec = recReqRDD.join(artistGreatistHitsRDD)
    recLength = conf['split']['reclistSize']
    s = lambda x: (x[0], sorter(x[1], recLength))
    rec = (joinedRec.
           map(val).
           reduceByKey(add).
           map(s))
    responses = (test.map(lambda x: (int(json.loads(x)['id']), x))
                 .join(rec)
                 .map(lambda x: addResponse(x[1][0], formatResponse(x[1][1], json.loads(x[1][0])['properties']['reclistSize']))))
    return responses
