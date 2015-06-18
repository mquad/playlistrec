__author__ = 'mquadrana'

import json
from operator import add


def addResponse(reqJson, respJson):
    req = json.loads(reqJson)
    req['linkedinfo'].update({'response': respJson})
    return json.dumps(req)


def formatResponse(rank_list, response_size=100):
    return [{'type': 'track', 'id': id, 'rank': rank} for id, rank in rank_list[:response_size]]


def sessionAggregation(train, test, config):
    trainSessions = train.map(lambda x: genSessions(x, config, 'train'))
    # this call generates an RDD having, for each user session with timestamp ts,
    # the list of all the sessions with a timestamp <= ts
    # the structure of a row in the RDD is the following:
    # (user_id, ts), list_of_sessions
    trainSessionHistory = trainSessions.join(trainSessions) \
        .filter(lambda x: x[1][0].ts >= x[1][1].ts) \
        .map(lambda x: ((x[0], x[1][0].ts), [x[1][1]])) \
        .reduceByKey(add)
    # APPLY THE HISTORY AGGREGATION POLICY to the session history RDD
    # this will generate a dictionary with the current session and the history
    # (represented, for convenience, as a dictionary with the aggregated values
    # for the tracks played in sessions previous to the current one)
    trainSessionHistoryAggr = trainSessionHistory.map(lambda x: applyHistoryAggrPolicy(x, config, find_current=True))
    # generate recommendation requests and group them all together for each user
    testRequests = test.map(lambda x: genSessions(x, config, 'test')) \
        .map(lambda x:(x[0], [x[1]])) \
        .reduceByKey(add)
    # generate the history for each group of test requests,
    # by taking all sessions from the training set (which are previous to each request by construction)
    # NOTE: History is kept constant for all the test requests
    testRequestsHistory = testRequests.join(trainSessions) \
        .map(lambda x: (x[0], (x[1][0], [x[1][1]]))) \
        .reduceByKey(lambda x, y: (x[0], x[1]+y[1]))
    # APPLY THE HISTORY AGGREGATION POLICY TO TEST REQUESTS
    testRequestsHistoryAggr = testRequestsHistory.map(lambda x: applyHistoryAggrPolicy(x, config, find_current=False))
    return trainSessionHistoryAggr, testRequestsHistoryAggr


def generateTraining(trainSessionHistoryAggr, config):
    # APPLY THE TRAINING GENERATION POLICY, then group training lines by user_id
    training = trainSessionHistoryAggr.flatMap(lambda x: applyTrainGenPolicy(x, config)) \
        .map(lambda x: (x.user_id, [x])) \
        .reduceByKey(add)
    return training


def filterTrainSessionsPerUser(training, trainSessionHistoryAggr, testRequestsHistoryAggr, config):
    # COLLABORATIVITY STRATEGY
    # pick the most recent history previous to the "split timestamp" for each user in trainSessionHistoryAggr
    recentTrainHistoryAggr = trainSessionHistoryAggr.map(lambda x: (x['current'].user_id, (x['current'].ts, x['history'])))\
        .filter(lambda x: x[0] < config['split']['ts'] and x[1][1])\
        .reduceByKey(lambda x, y: x if x[0] > y[0] else y)\
        .map(lambda x: (x[0], x[1][1]))
    collabMap = buildCollaborativityMap(recentTrainHistoryAggr, testRequestsHistoryAggr, config)
    trainSessionsPerUser = collabMap.map(lambda x: (x[1], x[0]))\
        .join(training)\
        .map(lambda x: (x[1][0], x[1][1]))\
        .reduceByKey(add)
    # generate the list of test entries for each user
    testRequestPerUser = testRequestsHistoryAggr.map(lambda x:(x['current'][0].user_id,
        [TestEntry(req.req_id, req.user_id, x['history'], req.tracks, req.properties) for req in x['current']])) \
        .reduceByKey(add)
    return trainSessionsPerUser, testRequestPerUser


def generateRecommendationsLibFM(trainSessionsPerUser, testRequestsPerUser, test, config):
    shuffle_enabled = config['algo']['props']['shuffleTraining']
    max_requests = config['algo']['props']['maxRequestsPerUser']
    # prepare training and test data and run LibFM
    joinedPerUser = trainSessionsPerUser.join(testRequestsPerUser) # (user_id, (TrainingSessions, TestSessions))
    # convert each line into LibFM textual format
    convertedLibFM = joinedPerUser.map(lambda x: (x[0], toLibFM(x[1], shuffle_lines=shuffle_enabled, max_requests=max_requests)))
    # generates an RDD of (req_id, (track_id, rank)) tuples
    predictions = convertedLibFM.flatMap(lambda x: runLibFM(x[0], x[1], config))
    # generate recommendations in the right format
    responses = test.map(lambda x: (json.loads(x)['id'], x))\
        .join(predictions)\
        .map(lambda x: addResponse(x[1][0], formatResponse(x[1][1], json.loads(x[1][0])['properties']['reclistSize'])))
    return responses