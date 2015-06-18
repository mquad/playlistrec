# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import time,json
from os import path

def computeMetrics(conf):
    '''
    conf = {}
    conf['evaluation'] = {}
    conf['evaluation']['metric'] = {}
    conf['evaluation']['metric']['type'] = 'recall'
    conf['evaluation']['metric']['prop'] = {}
    conf['evaluation']['metric']['prop']['N'] = [1,2,5,10,15,20,25,50,100]
    conf['evaluation']['name'] = 'recall@N'
    conf['split'] = {}
    conf['split']['name'] = 'mieProve000'
    conf['general'] = {}
    conf['general']['clientname'] = "split"
    conf['general']['bucketName'] = 'contentwise-research-poli'
    conf['algo'] = {}
    conf['algo']['name'] = 'GHTraining'
    '''

    basePath = path.join("s3n://", conf['general']['bucketName'], conf['general']['clientname'])
    # basePath = "s3n://" + conf['general']['bucketName'] + "/"+conf['general']['clientname']+"/"
    splitPath = path.join(basePath, conf['split']['name'])
    # splitPath = basePath+conf['split']['name']+"/"
    GTpath = path.join(splitPath, "GT")
    # GTpath = splitPath+"GT"

    algo_conf = conf['algo']['name'] + '_' + \
                '#'.join([str(v) for k, v in conf['algo']['props'].iteritems() if not k.startswith('libFM')])
    confPath = path.join(splitPath, 'Rec', algo_conf)
    recPath = path.join(confPath, "recommendations")
    #recPath = splitPath+"/Rec/"+ conf['algo']['name']+"/recommendations/"

    gtRDD = sc.textFile(GTpath).map(lambda x: json.loads(x)).persist(StorageLevel.MEMORY_AND_DISK)
    recRDD = sc.textFile(recPath).map(lambda x: json.loads(x)).persist(StorageLevel.MEMORY_AND_DISK)


    recommendationRDD = recRDD \
        .flatMap(lambda x: ([(x['id'], (k['id'],k['rank'],x)) for k in x['linkedinfo']['response']]))

    groundTruthRDD = gtRDD \
        .flatMap(lambda x: ([(x['linkedinfo']['gt'][0]['id'], (k['id'],x)) for k in x['linkedinfo']['objects']]))

    hitRDDPart = recommendationRDD.join(groundTruthRDD).filter(lambda x: x[1][0][0] == x[1][1][0])
    hitRDDPart.map(lambda x: {"type": "linebyline", "id":-1, "ts":-1, "linkedinfo":
        {"recom": x[1][0][2],"GT": x[1][1][1]}}).map(lambda x: json.dumps(x)) \
        .repartition(10)\
        .saveAsTextFile(path.join(confPath, conf['evaluation']['name'], "lineByLine"))
    #.saveAsTextFile(path.join(confPath, "lineByLine"))

    hitRDD = hitRDDPart.map(lambda x: (x[0],x[1][0][1],1.0)).persist(StorageLevel.MEMORY_AND_DISK)
    '''
    {"type": "metric", "id": -1, "ts" : -1, "properties": {"name": "recall@20" ,"value": 0.25}, 
    "linkedinfo":{"subjects":[], "objects" : [] }}
    '''
    totRec = float(groundTruthRDD.count())
    result = []

    for n in conf['evaluation']['metric']['prop']['N']:
        temp = {}
        temp['type'] = 'metric'
        temp['id'] = -1
        temp['ts'] = time.time()
        temp['properties'] = {}
        temp['properties']['name'] = conf['evaluation']['name']
        temp['evaluation'] = {}
        temp['evaluation']['N'] = n
        temp['evaluation']['value'] = hitRDD.filter(lambda x: x[1] <n).map(lambda x: x[2]).sum()/totRec
        temp['linkedinfo'] = {}
        temp['linkedinfo']['subjects'] = []
        temp['linkedinfo']['subjects'].append({})
        temp['linkedinfo']['subjects'][0]['splitName'] = conf['split']['name']
        temp['linkedinfo']['subjects'][0]['algoName'] = conf['algo']['name']
        result.append(temp)
    metricsPath = path.join(confPath, conf['evaluation']['name'], "metrics")
    sc.parallelize(result).map(lambda x: json.dumps(x)) \
        .saveAsTextFile(metricsPath)
    print "%s successfully written to %s" % (conf['evaluation']['name'], metricsPath)
