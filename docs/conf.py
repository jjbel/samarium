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
version = "2.0.0"
release = "2.0.0"

extensions = [
    "breathe",
    "myst_parser",
    "sphinx-mathjax-offline",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinxext.opengraph",
]

exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

html_css_files = [
    "custom.css",
]

html_theme_options = {
    "dark_css_variables": {
        "color-brand-primary": "#1c76fc",
        "color-brand-content": "#1c76fc"
    },
}

# pygments_style = "one-dark"
pygments_dark_style = "one-dark"

source_suffix = [".rst", ".md"]

# Breathe configuration

breathe_projects = {"samarium": "build/doxygenxml/"}

breathe_default_project = "samarium"

breathe_domain_by_extension = {
    "hpp": "cpp",
}

breathe_default_members = ("members", "undoc-members")

primary_domain = "cpp"
highlight_language = "cpp"

ogp_site_url = "https://strangequark1041.github.io/samarium/"
ogp_site_name = "Samarium Docs"

run(["doxygen", "Doxyfile.cfg"], check=True)
