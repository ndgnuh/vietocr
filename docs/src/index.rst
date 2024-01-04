.. vietocrpp documentation master file, created by
   sphinx-quickstart on Tue Dec 19 15:06:01 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====================================
VietOCR* the unnamed VietOCR's fork
=====================================

.. note::
   The main reason I have not registered the project to PyPI because
   of name clash, I need to change the name to something else before
   publishing it.


Getting started
===============

To use VietOCR++, first install it from the repository:

.. code-block:: shell

   pip install git+https://gitlab.com/ndgnuh/vietocr.git


Basic inference:

.. code-block:: python

   from vietocr import Predictor, get_config
   import cv2

   image = cv2.imread("image.png")
   config = get_config("fvtr_t@vn")
   model = Predictor(config)
   text, score = model.predict(image)

What's next?
============

.. toctree::
   :maxdepth: 2

   training
   modelling/index
