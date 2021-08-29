# -*- coding:utf-8 -*-
"""将dict 转换为 Dict2Object"""


class Dict2Obj(object):
    """dict to Dict2Obj
    d: data"""

    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [Dict2Obj(x) if isinstance(
                    x, dict) else x for x in b])
            else:
                setattr(self, a, Dict2Obj(b) if isinstance(b, dict) else b)
