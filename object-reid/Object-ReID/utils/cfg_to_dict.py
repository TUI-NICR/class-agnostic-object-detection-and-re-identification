from yacs.config import CfgNode


def cfg_to_dict(cfg_node):
    d = {}
    items = list(cfg_node.items())
    for k, v in items:
        if isinstance(v, CfgNode):
            d[k] = cfg_to_dict(v)
        else:
            d[k] = v
    return d
