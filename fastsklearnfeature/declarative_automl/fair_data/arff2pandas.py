import arff as liacarff
import pandas as pd


def __load(arff):
    """
    load liac-arff to pandas DataFrame
    :param dict arff:arff dict created liac-arff
    :rtype: DataFrame
    :return: pandas DataFrame
    """
    attrs = arff['attributes']
    attrs_t = []
    for attr in attrs:
        if isinstance(attr[1], list):
            attrs_t.append("%s@{%s}" % (attr[0], ','.join(attr[1])))
        else:
            attrs_t.append("%s@%s" % (attr[0], attr[1]))

    df = pd.DataFrame(data=arff['data'], columns=attrs_t)
    return df


def load(fp):
    """
    load file to pandas DataFrame
    :param file fp:
    :rtype: DataFrame
    :return: pandas DataFrame
    """
    data = liacarff.load(fp)
    return __load(data)


def loads(s):
    """
    load str to pandas DataFrame
    :param str s:
    :rtype: DataFrame
    :return: pandas DataFrame
    """
    data = liacarff.loads(s)
    return __load(data)


def __dump(df,relation='data',description=''):
    """
    dump DataFrame to liac-arff
    :param DataFrame df:
    :param str relation:
    :param str description:
    :rtype: dict
    :return: liac-arff dict
    """
    attrs = []
    for col in df.columns:
        attr = col.split('@')
        if attr[1].count('{')>0 and attr[1].count('}')>0:
            vals = attr[1].replace('{','').replace('}','').split(',')
            attrs.append((attr[0],vals))
        else:
            attrs.append((attr[0],attr[1]))

    data = list(df.values)
    result = {
        'attributes':attrs,
        'data':data,
        'description':description,
        'relation':relation
    }
    return result


def dump(df,fp):
    """
    dump DataFrame to file
    :param DataFrame df:
    :param file fp:
    """
    arff = __dump(df)
    liacarff.dump(arff,fp)


def dumps(df):
    """
    dump DataFrame to str
    :param DataFrame df:
    :rtype: str
    :return: dumped arff
    """
    arff = __dump(df)
    return liacarff.dumps(arff)