__author__ = 'massimo'

def readJson(x, field='artists', pos=4):
    return json.loads(x[pos])[field]


def sorter(x, length):
    return sorted(x, key=lambda k: k[1], reverse=True)[0:length]


def validator(x):
    recID = x[1][0][0]
    trackList = x[1][0][1]
    recList = x[1][1]
    rec = [l for l in recList if l[0] not in trackList]
    return recID, rec


parser = lambda x: (x[1], 1)
prep = lambda x: (x[0][1], [(x[0][0], x[1])])
val = lambda x: validator(x)
ext = lambda x: ([(k['id'], k['playratio'], x['id']) for k in x['linkedinfo']['objects']])