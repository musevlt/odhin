# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 07:59:14 2016

@author: raphael.bacher@gipsa-lab.fr
"""


def _setup_logging():
    from mpdaf.log import setup_logging, clear_loggers
    setup_logging(name='', level='INFO', color=True,
                  fmt='%(levelname)s %(message)s')
    clear_loggers('mpdaf')


_setup_logging()

from .deblend import *
from .grouping import *
from .odhin import ODHIN
from .parameters import *
from .utils import *
from .version import __version__
