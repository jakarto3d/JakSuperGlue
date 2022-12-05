# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read().splitlines()


setup(
    name="jaksuperglue",
    version="0.0.0",
    description="Adaptation of SuperGlue: Learning Feature Matching with Graph Neural Networks",
    long_description=readme,
    author="Hugo Vazquez",
    author_email="hugo.vazquez@jakarto.com",
    url="https://github.com/jakarto3d/JakSuperGlue",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=requirements,
    license="Jakarto Licence",
    zip_safe=False,
)