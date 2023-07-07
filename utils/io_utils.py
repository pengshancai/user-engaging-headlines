import json
import pickle
# IO
def loadFromJson(filename):
    f = open(filename, 'r', encoding='utf-8')
    data = json.load(f, strict=False)
    f.close()
    return data


def saveToJson(filename, data):
    f = open(filename, 'w', encoding='utf-8')
    json.dump(data, f, indent=4)
    f.close()
    return True


def saveToPKL(filename, data):
    with open(filename, 'wb')as f:
        pickle.dump(data, f)
    return


def loadFromPKL(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
