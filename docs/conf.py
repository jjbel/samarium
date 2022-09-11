# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from subprocess import run

project = "Samarium"
copyright = "2022, Jai Bellare"
author = "Jai Bellare"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["breathe", "myst_parser", "sphinx.ext.mathjax", "sphinx_copybutton"]

exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

source_suffix = [".rst", ".md"]

# Breathe configuration

breathe_projects = {"samarium": "build/doxygenxml/"}

breathe_default_project = "samarium"

breathe_domain_by_extension = {
    "hpp": "cpp",
}

primary_domain = "cpp"
highlight_language = "cpp"

# Configuration for mathjax extension
#
# Set path for mathjax js to a https URL as sometimes the docs are displayed under https
# and we can't load an http mathjax file from an https view of the docs. So we change to a https
# mathjax file which we can load from http or https. We break the url over two lines.
mathjax_path = (
    "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)

print("Running Doxygen...", end=" ")
run(["doxygen", "Doxyfile.cfg"], check=True)
print("done")
