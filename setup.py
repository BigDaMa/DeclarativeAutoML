# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='declarativeautoml',
    version='0.0.1',
    description='Declarative AutoML',
    long_description=readme,
    author='Felix Neutatz',
    author_email='neutatz@gmail.com',
    url='https://github.com/FelixNeutatz/DeclarativeAutoML',
    license=license,
    package_data={'config': ['fastsklearnfeature/configuration/resources']},
    include_package_data=True,
    install_requires=["numpy",
                      "pandas",
                      "scikit-learn==0.22.1",
                      "optuna",
                      "openml",
                      "diffprivlib==0.2.0",
                      "anytree",
                      "scipy==1.4.1",
                      "ipykernel",
                      "auto-sklearn==0.10.0",
                      "libtmux"
                      ],
    packages=find_packages(exclude=('tests', 'docs'))
)

