import setuptools
from sphinx.setup_command import BuildDoc

cmdclass = {"build_sphinx": BuildDoc}
setuptools.setup()
