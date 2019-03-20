# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 07:59:14 2016

@author: raphael.bacher@gipsa-lab.fr
"""

from mpdaf.log import setup_logging, clear_loggers
setup_logging(name='', level='INFO', color=True,
              fmt='%(levelname)s %(message)s')
clear_loggers('mpdaf')

from .odhin import ODHIN
from .parameters import DEFAULT_PARAMS, Params
from .version import __version__
