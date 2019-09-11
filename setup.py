# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from setuptools import find_packages, setup

requirements = ['numpy>=1.16',
                'mpmath>=1.1.0',
                'torch==1.1.0',
                ]

setup(packages=find_packages(exclude=['docs']),
      install_requires=requirements,
      include_package_data=True)