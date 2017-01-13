""" Returns the parent nodes of a dict-based general tree with
    root -1 and leaves None  
    Example: tree = {-1: {0: {2: {6: None}},
					      1: {3: {7: None,
								  8: None,
								  9: None,
								  10: None,
								  11: None},
							  4: {12: None,
						 		  13: None},
							  5: {14: None,
								  15: None,
								  16: None}}}}
"""
def get_parents(tree, idx):
    for sub in dict_generator(tree):
        if idx in sub:
    	    p = [item for item in sub if item is not None and item is not -1]
    	    p.sort()
    	    return [i for i in p if i < idx]


""" Credit for the following function belongs to Valentin Bryukhanov
    from stackoverflow.com article: 'Python: Recommended way to walk 
    complex dictionary structures imported from JSON?' 
"""
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

