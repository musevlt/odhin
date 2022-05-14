"""
Created on Thu Jun 30 07:59:14 2016

@author: raphael.bacher@gipsa-lab.fr
"""

from .deblend import *  # noqa
from .grouping import *  # noqa
from .odhin import ODHIN  # noqa
from .parameters import *  # noqa
from .utils import *  # noqa
from .version import version as __version__  # noqa


def _setup_logging():
    import sys
    from mpdaf.log import setup_logging, clear_loggers
    setup_logging(name='', level='INFO', color=True, stream=sys.stdout,
                  fmt='%(levelname)s %(message)s')
    clear_loggers('astropy')
    clear_loggers('mpdaf')


_setup_logging()
