# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from subprocess import run
from pathlib import Path
from os import chdir

project = "Samarium"
copyright = "2022, Jai Bellare"
author = "Jai Bellare"
version = "1.1.0"
release = "1.1.0"

extensions = [
    "breathe",
    "myst_parser",
    "sphinx-mathjax-offline",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinxext.opengraph",
    "sphinx.ext.extlinks",
    "sphinxcontrib.asciinema",
    "sphinx.ext.graphviz"
]

HEAD_REF = run(
    ["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True
).stdout.decode("unicode_escape")[:-1]

extlinks = {
    "github": (
        f"https://github.com/jjbel/samarium/blob/{HEAD_REF}/%s",
        "%s",
    ),
    "src": (
        f"https://github.com/jjbel/samarium/blob/{HEAD_REF}/src/samarium/%s",
        "samarium/%s",
    ),
}

exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "samarium docs"

html_static_path = ["_static"]

html_css_files = [
    "custom.css",
]

html_theme_options = {
    "light_css_variables": {
        "font-stack": "'Noto Sans', monospace",
        "font-stack--monospace": "'Noto Sans Mono', monospace",
    },
    "dark_css_variables": {
        "color-background-primary": "#111214",
        "color-background-secondary": "#0e0f11",
        "color-sidebar-background-border": "#090a0b",
        "color-sidebar-search-border": "#1c1e21",
        "color-sidebar-item-background--hover": "#212227"
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/jjbel/samarium",
            "html": Path('_static/github-logo.svg').read_text(),
            "class": "",
        },
    ],
}

source_suffix = [".rst", ".md"]

# Breathe configuration
breathe_projects = {"samarium": "build/doxygenxml/"}
breathe_default_project = "samarium"
breathe_domain_by_extension = {
    "hpp": "cpp",
}
breathe_default_members = ("members", "undoc-members")
breathe_show_include = False

primary_domain = "cpp"
highlight_language = "cpp"

mathjax3_config = {
    "chtml": {
        "scale": 1.1
    },
    "svg": {
        "scale": 1.1
    }
}

ogp_site_url = "https://jjbel.github.io/samarium/"
ogp_site_name = "Samarium Docs"

run(["doxygen", "Doxyfile.cfg"], check=True)

chdir(Path('../..'))
run(['python3', Path('scripts/includes.py')], check=True)
chdir('docs/src')
