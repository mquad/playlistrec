__author__ = 'mquadrana'

import json
import numpy as np
import errno
from subprocess import call
from collections import namedtuple
from hashlib import sha1
from heapq import nlargest
from operator import add
from os import path, makedirs
from random import shuffle

Request = namedtuple('Request', ['req_id', 'user_id', 'ts', 'tracks', 'properties'])
Session = namedtuple('Session', ['user_id', 'ts', 'tracks'])
TestEntry = namedtuple('TestEntry', ['req_id', 'user_id', 'history', 'tracks', 'properties'])

'''
Auxiliary functions for managing consecutive track plays
'''

def d_unique(iterable, aggr_fun=max):
    current = iterable[0]
    for next in iterable[1:]:
        if next['id'] == current['id']:
            current['playratio'] = aggr_fun(current['playratio'], next['playratio'])
        else:
            yield current
            current = next


def genSessions(sessionJson, config, mode='train'):
    session = json.loads(sessionJson)
    user_id = session['linkedinfo']['subjects'][0]['id']
    objects = session['linkedinfo']['objects']
    ts = int(session['ts'])
    duplicate_policy = config['algo']['props']['duplicatePolicy']
    # apply the consecutive duplicate tracks handling policy
    if duplicate_policy == 'unique_max':
        obj_iterable = d_unique(objects, max)
    elif duplicate_policy == 'unique_min':
        obj_iterable = d_unique(objects, min)
    else:
        obj_iterable = objects
    # generate sessions
    tracks = []
    for idx, x in enumerate(obj_iterable):
        if x['playratio'] is not None:
            playratio = np.clip(float(x['playratio']), 0.0, 1.0)
            playratio = 1 if playratio > .1 else 0  # binarization
        else:
            # TODO: define how to consider None playratios
            # a playratio of 1 is introduced for the last played track and tracks that have not been totally
            #  skipped (i.e., optimistic forecasting)
            playratio = 0 if x['playtime'] <= 0 and idx < len(objects)-1 else 1
        tracks.append((x['id'], playratio))
    if mode == 'train':
        return user_id, Session(user_id, ts, tracks)
    elif mode == 'test':
        req_id = session['id']
        properies = session['properties']
        return user_id, Request(req_id, user_id, ts, tracks, properies)


def applyHistoryAggrPolicy(sessionWithHistory, config, find_current=False):
    history_policy = config['algo']['props']['historyAggrPolicy']
    current_ts = None if not find_current else sessionWithHistory[0][1]
    current_session = None if find_current else sessionWithHistory[1][0]
    history = sessionWithHistory[1] if find_current else sessionWithHistory[1][1]
    track_aggr_dict = dict()
    for session in history:
        if find_current and session.ts == current_ts:
            current_session = session
        else:
            # TODO: implement other history aggregation policies (e.g. exponential decay)
            # apply the history aggregation policy
            if history_policy == 'count':
                for track_id, _ in session.tracks:
                    track_aggr_dict[track_id] = track_aggr_dict.get(track_id, 0) + 1
            elif history_policy == 'avg':
                for track_id, playratio in session.tracks:
                    track_aggr_dict[track_id] = track_aggr_dict.get(track_id, np.array([0.0, 0])) + np.array([playratio, 1])
            elif history_policy == 'max':
                for track_id, playratio in session.tracks:
                    track_aggr_dict[track_id] = max(track_aggr_dict.get(track_id, 0.0), playratio)
            elif history_policy == 'min':
                for track_id, playratio in session.tracks:
                    track_aggr_dict[track_id] = min(track_aggr_dict.get(track_id, 0.0), playratio)

    if history_policy == 'avg':
        track_aggr_dict = {k: sum/count for k, (sum, count) in track_aggr_dict.iteritems()}
    return {"current": current_session, "history": [(k, v) for k, v in track_aggr_dict.iteritems()]}


'''
Auxiliary functions required by the training generation policies
'''


def leaveoneout(iterable):
    for i in xrange(len(iterable)):
        yield iterable[:i] + iterable[i+1:], iterable[i]


def sequential(iterable):
    for i in xrange(len(iterable)):
        yield iterable[:i], iterable[i]


TrainingEntry = namedtuple('TrainingEntry', ['user_id', 'history', 'current', 'target'])


def applyTrainGenPolicy(sessionWithAggrHistory, config):
    train_policy = config['algo']['props']['trainGenPolicy']
    current_session = sessionWithAggrHistory['current']
    history = sessionWithAggrHistory['history']
    # training generation function: how to split the current session for each training entry
    if train_policy == 'leaveoneout':
        train_gen_fun = leaveoneout
    elif train_policy == 'sequential':
        train_gen_fun = sequential
    # split the current session and generate the training entry
    training_list = []
    for current, target in train_gen_fun(current_session.tracks):
        training_list.append(TrainingEntry(current_session.user_id, history, current, target))
    return training_list


'''
Collaborativity strategies
'''

def normalize_rows(rdd):
    norms = rdd.map(lambda x: (x[0], x[1][1] * x[1][1])) \
        .reduceByKey(add) \
        .map(lambda x: (x[0], np.sqrt(x[1])))
    return rdd.join(norms).map(lambda x: (x[0], (x[1][0][0], x[1][0][1]/ x[1][1])))


def get_nlargest(n, iterable, key=lambda x: x):
    if n <= 10:
        # heapq.nlargest is faster than sorted for small n  values
        return nlargest(n, iterable, key)
    else:
        return sorted(iterable, key=key, reverse=True)[:n]


# rdd in the form (row_idx, (col_idx, value))
def uu_sim(rdd, del_diag=False, knn_filter=True, k=5):
    # normalize each row
    u1_normalized = normalize_rows(rdd).map(lambda x: (x[1][0], (x[0], x[1][1])))
    sim = u1_normalized.join(u1_normalized)
    if del_diag:
        sim = sim.filter(lambda x: x[1][0][0] != x[1][1][0])
    sim = sim.map(lambda x: ((x[1][0][0], x[1][1][0]), x[1][0][1] * x[1][1][1])) \
        .reduceByKey(add)
    sim_vec = sim.map(lambda x: (x[0][0], [(x[0][1], x[1])]))\
        .reduceByKey(add)
    if knn_filter:
        # TODO: switch to pure Spark-like knn filter
        return sim_vec.map(lambda x: (x[0], get_nlargest(k, x[1], lambda t: t[1])))
    else:
        return sim_vec


# generates a RDD with (K,V) pairs, having for each user_id K in testRequestsHistoryAggr any user_id V associated with it
# according to the chosen collaborativity strategy
def buildCollaborativityMap(recentTrainHistoryAggr, testRequestsHistoryAggr, config):
    collab = config['algo']['props']['collaborativity']
    if collab == 'individual':
        return testRequestsHistoryAggr.map(lambda x: (x['current'][0].user_id, x['current'][0].user_id))
    elif collab == 'knn':
        k = config['algo']['props']['neighborhoodSize']
        # build the user-user similarity matrix
        trainUserProfiles = recentTrainHistoryAggr.flatMap(lambda x: zip([x[0]]*len(x[1]), x[1]))
        sim = uu_sim(trainUserProfiles, del_diag=False, knn_filter=True, k=k)
        return sim.flatMap(lambda x: zip([x[0]]*len(x[1]), map(lambda t: t[0], x[1])))
    else:
        #TODO: All collaborativity
        pass

'''
Generate the textual files to pass to libFM
'''
def mkdir_p(dir_path):
    try:
        makedirs(dir_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and path.isdir(dir_path):
            pass



def get_and_incr(key, dict, counter):
    if key not in dict:
        dict[key] = counter
        counter += 1
    return dict[key], counter


def reverse_dict(dict):
    return {v: k for k, v in dict.iteritems()}


# fm_tuple is a tuple made of (trainingSessions, testRequests)
# converts each line of the rdd into a textual file in libSVM format, ready to be launched by libFM
# TODO: perform 2 passes to get the number of users and tracks
def toLibFM(fm_tuple, shuffle_lines=False, max_requests=-1):
    training, test = fm_tuple
    user_lookup, user_count = dict(), 0
    history_lookup, history_count = dict(), 0
    session_lookup, session_count = dict(), 0

    # first pass to build the user and history lookup dictionaries
    for tr_entry in training:
        _, user_count = get_and_incr(tr_entry.user_id, user_lookup, user_count)
        for track_id, _ in tr_entry.history:
            _, history_count = get_and_incr(track_id, history_lookup, history_count)
        for track_id, _ in tr_entry.current:
            _, session_count = get_and_incr(track_id, session_lookup, session_count)
    for te_entry in test:
        for track_id, _ in te_entry.history:
            _, history_count = get_and_incr(track_id, history_lookup, history_count)
        for track_id, _ in te_entry.tracks:
            _, session_count = get_and_incr(track_id, session_lookup, session_count)

    # compute base indexes and update lookup dictionaries
    history_base_id = user_count
    session_base_id = history_base_id + history_count
    target_base_id = session_base_id + session_count

    history_lookup = {k: v+history_base_id for k, v in history_lookup.iteritems()}
    session_lookup = {k: v+session_base_id for k, v in session_lookup.iteritems()}


    target_lookup, target_count = dict(), target_base_id

    # LIBFM: TRAINING
    training_libfm = []
    for tr_entry in training:
        # LIBFM: USER-ID
        user_libfm_id = user_lookup[tr_entry.user_id]
        user_libfm = "%d:1 " % user_libfm_id
        # LIBFM; HISTORY
        history_libfm = ""
        for track_id, value in tr_entry.history:
            history_libfm_id = history_lookup[track_id]
            history_libfm += "%d:%.3f " % (history_libfm_id, value)
        # LIBFM: SESSION
        session_libfm = ""
        for track_id, value in tr_entry.current:
            session_libfm_id = session_lookup[track_id]
            session_libfm += "%d:%.3f " % (session_libfm_id, value)
        # LIBFM: TARGET
        target_id, target_value = tr_entry.target
        target_libfm_id, target_count = get_and_incr(target_id, target_lookup, target_count)
        target_libfm = "%d:1 " % target_libfm_id
        # append a new line to the training string
        training_libfm.append("%.3f %s%s%s%s" % (target_value, target_libfm, user_libfm, history_libfm, session_libfm))

    # LIBFM: TEST
    requests_libfm = []
    n_req = 0
    for te_entry in test:
        # LIBFM: USER-ID
        user_libfm_id = user_lookup[te_entry.user_id]
        user_libfm = "%d:1 " % user_libfm_id
        # LIBFM; HISTORY
        history_libfm = ""
        for track_id, value in te_entry.history:
            history_libfm_id = history_lookup[track_id]
            history_libfm += "%d:%.3f " % (history_libfm_id, value)
        # LIBFM: SESSION
        session_libfm = ""
        for track_id, value in te_entry.tracks:
            session_libfm_id = session_lookup[track_id]
            session_libfm += "%d:%.3f " % (session_libfm_id, value)
        # append a new line to the test string (the request_id of the recommendation is stored in a comment for every line)
        requests_libfm.append("%s%s%s#%s" % (user_libfm, history_libfm, session_libfm, te_entry.req_id))
        n_req += 1
        if max_requests > 0 and n_req >= max_requests:
            break

    # TODO: edit libFM to avoid this step
    # TODO: check for tracks in tgt that are already in the current session
    # generate one test line for each possible target track for each request
    test_libfm = ['\n'.join(["0.000 %d:1 %s" % (target_libfm_id, request)
                                       for _, target_libfm_id in target_lookup.iteritems()]) for request in requests_libfm]

    if shuffle_lines:
        shuffle(training_libfm)
        shuffle(test_libfm)

    return {"training": training_libfm, "test": test_libfm,  "lookup": reverse_dict(target_lookup)}


def runLibFM(user_id, fm_data, config):
    # persist training and test data to disk
    confighash = sha1('_'.join(map(str, config['algo']['props'].values()))).hexdigest()
    tmpdir = path.join(config['general']['tmpDir'], config['split']['name'], confighash, 'u_%d' % user_id)
    track_lookup = fm_data['lookup']
    train_file = path.join(tmpdir, 'train.txt')
    test_file = path.join(tmpdir, 'test.txt')
    pred_file = path.join(tmpdir, 'pred.txt')
    log_file = path.join(tmpdir, 'log_libfm.txt')
    mkdir_p(tmpdir)
    with open(train_file, 'w') as train:
        train.write('\n'.join(fm_data['training']))
    with open(test_file, 'w') as test:
        test.write('\n'.join(fm_data['test']))
    # TODO: binarize and transpose
    # call libFM on this sample
    logto = open(log_file, 'w')
    libFM_bin = config['algo']['props']['libFMbin']
    libFM_opt = config['algo']['props']['libFMopt']
    cmd = "%s %s -train %s -test %s -out %s" % (libFM_bin, libFM_opt, train_file, test_file, pred_file)
    call(cmd, stdout=logto, stderr=logto, shell=True)

    #retrive the results and compute the final ranking
    requests = []
    for test_req in fm_data['test']:
        for test_line in test_req.split('\n'):
            tsplit = test_line.split()
            # create tuples (request_id, target_id)
            requests.append((int(tsplit[-1].lstrip('#')), track_lookup[int(tsplit[1].split(':')[0])]))
    predictions = zip(requests, [float(pred) for pred in open(pred_file, 'r')])

    ntracks = len(track_lookup)
    num_req = len(predictions) / ntracks

    assert(len(predictions) % ntracks == 0)
    rank = []
    for r in xrange(num_req):
        predictions[r*ntracks:(r+1)*ntracks] = sorted(predictions[r*ntracks:(r+1)*ntracks], key=lambda x: x[1], reverse=True)
        # TOGGLE COMMENTS TO RETURN THE PREDICTED PLAYTIME (FOR DEBUG ONLY)
        # rank.append((predictions[r*ntracks][0][0], [(p[0][1], p[1], idx) for idx, p in enumerate(predictions[r*ntracks:(r+1)*ntracks])]))
        rank.append((predictions[r*ntracks][0][0], [(p[0][1], idx) for idx, p in enumerate(predictions[r*ntracks:(r+1)*ntracks])]))

    #return zip(predictions, rank)
    return rank