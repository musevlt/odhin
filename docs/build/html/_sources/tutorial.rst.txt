Tutorial
=================

Preliminary imports::

   import numpy as np

   import matplotlib.pyplot as plt

   from deblend import main_deblending,deblend_utils,fonction_presentation


Import source::

   src = Source.from_file(src_filename) 

Create deblend object::

   debl=main_deblending.Deblending(src)

Correct spatial alignement::

   debl.corrAlign()
 
Create intensity maps from HST::

   debl.createAbundanceMap(segmap=src.images['HST_SEGMAP'].data)

Do deblending::

   debl.findSources()

Show results::

   fonctions_presentation.plotMainDeblend(debl300)
