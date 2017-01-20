from __future__ import division

person_tree = {-1: {0: {2: {9: None}},
					1: {3: {10: None,
							11: None,
							12: None,
							13: None,
							14: None,
							15: None,
							16: None,
							17: None,
							18: None,
							19: None},
						4: {20: None,
						 	21: None},
						5: {22: None,
							23: None,
							24: None},
						6: {25: None,
							26: None,
							27: None},
						7: {28: None,
							29: None,
							30: None},
						8: {31: None,
							32: None,
							33: None}}}}

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

def get_parents(key, tree=person_tree):
	for sub in dict_generator(tree):
		if key in sub:
			p = list(set(sub) - set([-1, None]))
			p.sort()
			return [i for i in p if i < key]
	raise Exception('Key \'%s\' not found' % key)

def get_family(key, tree=person_tree):
	family = [key]
	for sub in dict_generator(tree):
		if key in sub:
			p = list(set(set(sub) - set([-1, None])))
			family.extend([i for i in p if i is not key])
	family = list(set(family))
	family.sort()
	return family

def get_depth(key, tree=person_tree):
	return len(get_parents(tree, key)) + 1

# Weighted Approximate Ranking (WARP)
def warp(key, tree=person_tree):
	if isinstance(keys, list):
		return [(k, 1/get_depth(tree, k)) for k in keys]
	elif isinstance(keys, int):
		return [(keys, 1/get_depth(tree, keys))]
	else:
		raise Exception("Key must be of type \'list\' or \'int\'")


