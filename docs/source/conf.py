# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

from packaging.version import Version

import sparsax  # noqa: E402

project = "sparsax"
copyright = "2024-, sparsax developers"  # noqa: A001
author = "sparsax developers"

version = Version(sparsax.__version__).public  # remove commit hash
release = version

language = "en"
html_title = project

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_immaterial",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

bibtex_bibfiles = ["_static/references.bib"]
bibtex_reference_style = "author_year"

master_doc = "index"

templates_path = [
    "_templates",
]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "exclude-members": "pullback,grad,_vjp,_fwd,_bwd,def_vmap,defvjp,perform,make_node,infer_shape,__init__,__props__",
}
suppress_warnings = ["ref.ref"]

html_theme = "sphinx_immaterial"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/file-code",
    },
    "site_url": "https://knaaptime.github.io/sparsax",
    "repo_url": "https://github.com/knaaptime/sparsax/",
    "edit_uri": "blob/main/docs",
    "repo_name": "knaaptime/sparsax",
    "features": [
        "navigation.sections",
        "navigation.top",
        "search.share",
        "search.suggest",
        "toc.follow",
        "toc.sticky",
        "content.code.copy",
        "content.action.edit",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme)",
            "toggle": {
                "icon": "material/brightness-auto",
                "name": "Switch to light mode",
            },
        },
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "indigo",
            "accent": "indigo",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "indigo",
            "accent": "indigo",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to system preference",
            },
        },
    ],
    "version_dropdown": True,
    "version_json": "https://knaaptime.github.io/sparsax/versions.json",
}
nb_execution_mode = "force"
nb_execution_timeout = -1
nb_kernel_rgx_aliases = {".*": "python3"}
nb_merge_streams = True
nb_execution_raise_on_error = True
nb_execution_show_tb = True
autodoc_typehints = "none"


sys.path.insert(0, os.path.abspath("../../src"))


def linkcode_resolve(domain, info):
    def find_source():
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(sparsax.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "sparsax/%s#L%d-L%d" % find_source()  # noqa: UP031
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    tag = "dev" if "dev" in release else ("v" + release)
    return f"https://github.com/knaaptime/sparsax/blob/{tag}/{filename}"
