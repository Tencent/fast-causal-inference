# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../src/package_util/python/causal_inference/fast_causal_inference'))


project = 'Fast-Causal-Inference'
copyright = '2024, Tencent'
author = 'Tencent'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode',"myst_parser",
              "sphinx.ext.autosummary"]

html_theme = "sphinx_rtd_theme"

html_theme_path = [os.path.dirname(__file__)]
