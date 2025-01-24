# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Sphinx documentation builder
"""

import os
import sys
import subprocess

# -- Project information -----------------------------------------------------
project = "Twirly"
copyright = ""  # pylint: disable=redefined-builtin
author = ""

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
]
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["style.css", "custom.css", "gallery.css"]

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True

# -----------------------------------------------------------------------------
# Autodoc
# -----------------------------------------------------------------------------

autodoc_default_options = {
    "inherited-members": None,
}

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------

html_theme = "qiskit_sphinx_theme"  # use the theme in subdir 'theme'

# html_sidebars = {'**': ['globaltoc.html']}
html_last_updated_fmt = "%Y/%m/%d"

html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}
