import json


def writejson(dic, filename):
    ''' Writes dictionary into a json file
    '''
    j = json.dumps(dic, indent=4)
    f = open(filename, 'w')
    print >> f, j
    f.close()


def readjson(filename):
    ''' Reads json file into a dictionary
    '''
    with open(filename) as f:
        dic = json.load(f)
    return dic
