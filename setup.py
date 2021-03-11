"""
Конфиг для билда пакета
"""


from setuptools import find_packages, setup

setup(
    name="ZenitAI",
    version="0.5.1",
    packages=find_packages(),
    descrption="""
    A collection of tools and utils for data analysis.
    Contains functions and tools for WOE-transformations and other utils
    """,
)
