# -*- coding: utf-8 -*-

from setuptools import setup, find_packages




with open('LICENSE') as f:
    license = f.read()

setup(
    name='declarativeautoml',
    version='0.0.1',
    description='Declarative AutoML',
    long_description='description',
    author='Felix Neutatz',
    author_email='neutatz@gmail.com',
    url='https://github.com/FelixNeutatz/DeclarativeAutoML',
    license=license,
    package_data={'config': ['fastsklearnfeature/configuration/resources']},
    include_package_data=True,
    install_requires=["numpy",#==1.16.5
                      "pandas==0.25.3",
                      #"scikit-learn==0.24.1",#0.22.1
                      "optuna",
                      #"openml==0.12.0",
                      "diffprivlib==0.2.0",
                      "anytree",
                      "scipy==1.4.1",
                      "ipykernel",
                      #"auto-sklearn==0.10.0",
                      "libtmux",
                      "matplotlib",
                      "imbalanced-learn==0.8.0",
                      "smac==0.13.1",
                      "pyDOE==0.3.8",
                      "emcee==3.0.2",
                      "scikit-optimize==0.8.1"
                      ],
    packages=find_packages(exclude=('tests', 'docs'))
)

