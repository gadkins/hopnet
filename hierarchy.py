def dict_generator(indict, pre=None):
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict) and not None:
                for d in dict_generator(value, [key] + pre):
                    yield d
            elif isinstance(value, list) or isinstance(value, tuple):
                for v in value:
                    for d in dict_generator(v, [key] + pre):
                        yield d
            else:
                yield pre + [key, value]
    else:
        yield indict


def get_parents(tree, idx):
    for sub in dict_generator(tree):
        if idx in sub:
    	    p = [item for item in sub if item is not None and item is not -1]
    	    p.sort()
    	    return [i for i in p if i < idx]

