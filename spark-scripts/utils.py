__author__ = 'robertopagano'

import json
import boto
import re
from os import path
'''
Matrix multiplication, similarity and kNN utility functions
'''

# rdd1,rdd2 in the form ((row_idx, col_idx), value)
# computes the matrix-matrix product RDD1 * RDD2^t
def matmul_join(rdd1, rdd2):
    rdd1mult = rdd1.map(lambda x: (x[0][1], (x[0][0], x[1])))
    rdd2mult = rdd2.map(lambda x: (x[0][1], (x[0][0], x[1])))
    return rdd1mult.join(rdd2mult) \
        .map(lambda x: ((x[1][0][0], x[1][1][0]), x[1][0][1] * x[1][1][1])) \
        .reduceByKey(lambda x, y: x+y)

'''
Load/Save utility functions
'''

def loadDataset(config):
    inFile = path.join('s3n://', config['general']['bucketName'], config['general']['clientname'], config['split']['name'])
    # trainBatch = sc.textFile(path.join(inFile, 'train/batchTraining'))
    # trainBatch = (sc
    #               .textFile(path.join(inFile, 'train/batchTraining'))
    #               .filter(lambda x: int(json.loads(x)['ts']) < config['split']['ts']))
    # testBatch = sc.textFile(path.join(inFile, 'test/batchTraining'))
    # testOnline = sc.textFile(path.join(inFile, 'test/onlineTraining'))
    testBatch = sc.textFile(path.join(inFile, 'test/batchTraining'))
    testOnline = sc.textFile(path.join(inFile, 'test/onlineTraining'))
    batch = testBatch
    return batch, testOnline


def s3_delete_recursive(mybucket, path):
    result = mybucket.delete_keys([k.name for k in mybucket.list(prefix=path)])
    if not result.errors:
        print "%s successfully deleted." % path
    else:
        print "An error has occurred while deleting keys in %s." % path
        print "Errors:"
        print '\n'.join(result.errors)


def saveRecommendations(conf, recJsonRdd, overwrite=False):
    s3 = boto.connect_s3()
    mybucket = s3.get_bucket(conf['general']['bucketName'])
    algo_conf = conf['algo']['name'] + '_' + \
                '#'.join([str(v) for k, v in conf['algo']['props'].iteritems()])
    algo_conf = re.sub(r'[^A-Za-z0-9#_]', '', algo_conf)
    base_path = path.join(conf['general']['clientname'], conf['split']["name"], 'Rec', algo_conf)
    rec_key = path.join(base_path, 'recommendations_$folder$')
    key = mybucket.get_key(rec_key)
    # check if recommendations have been already computed for this configuration of the recommendation algorithm
    if key:
        if not overwrite:
            print "%s already exists.\nRe-run with overwrite=True to overwrite it." % base_path
            return
        s3_delete_recursive(mybucket, base_path)
    # save recommendations to S3 storage
    outfile = path.join('s3n://', conf['general']['bucketName'], base_path, "recommendations")
    # recJsonRdd.repartition(10).saveAsTextFile(outfile)
    recJsonRdd.saveAsTextFile(outfile)
    print "Recommendations successfully written to %s" % outfile
