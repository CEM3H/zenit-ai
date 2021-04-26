"""
Конфиг для билда пакета
"""


from setuptools import find_packages, setup
from sphinx.setup_command import BuildDoc

cmdclass = {"build_sphinx": BuildDoc}

name = "ZenitAI"
version = "0.5"
release = "2"

setup(
    cmdclass=cmdclass,
    name="ZenitAI",
    version=version + "." + release,
    packages=find_packages(),
    author="Stepan Kadochnikov",
    descrption="""
    A collection of tools and utils for data analysis.
    Contains functions and tools for WOE-transformations and other utils
    """,
    # these are optional and override conf.py settings
    command_options={
        "build_sphinx": {
            "project": ("setup.py", name),
            "version": ("setup.py", version),
            "release": ("setup.py", release),
            "source_dir": ("setup.py", "docs"),
        }
    },
)
