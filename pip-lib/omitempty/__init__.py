# -*- coding: UTF-8 -*-

import sys
from uuid import uuid4

__version__ = "0.1.1"


class mod_omitempty(object):
    def __init__(self):
        self.__version__ = __version__
        self.__mark = "___%s" % uuid4()

    def __call__(self, d):
        return self.omitempty(d)

    def _omitempty_rec(self, d):
        d[self.__mark] = True
        d2 = {}

        for k, v in d.items():
            if k == self.__mark:
                continue

            if not v:  # empty
                continue

            if isinstance(v, dict) and self.__mark not in v:
                d2[k] = self._omitempty_rec(v)
            else:
                d2[k] = v

        return d2

    def _remove_marks(self, d):
        if self.__mark not in d:
            return
        del d[self.__mark]
        for v in d.values():
            if isinstance(v, dict):
                self._remove_marks(v)

    def omitempty(self, d):
        try:
            d2 = self._omitempty_rec(d)
        finally:
            self._remove_marks(d)

        return d2

sys.modules[__name__] = mod_omitempty()
