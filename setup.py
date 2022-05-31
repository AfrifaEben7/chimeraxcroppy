"""Python setup.py for lls_crop package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("lls_crop", "VERSION")
    '0.5.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="lls_crop",
    version=read("lls_crop", "VERSION"),
    description="lls_crop created by bscott711",
    url="https://github.com/bscott711/chimeraxcroppy/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="bscott711",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["lls_crop = lls_crop.cli:main"]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
)
