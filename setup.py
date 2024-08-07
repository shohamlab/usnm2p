# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2024-08-07 15:49:30
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2024-08-07 15:53:07

from setuptools import setup

requirements_file = 'requirements.txt'

def get_requirements():
    with open(requirements_file, 'r', encoding="utf8") as f:
        reqs = f.read().splitlines() 
    return reqs


setup(
    name='usnm2p',
    version='1.0',
    description='Analysis of two-photon calcium imaging data acquired upon ultrasound stimulation',
    keywords=('two-photon imaging neural activity ultrasound neuromodulation'),
    author='Theo Lemaire',
    author_email='theo.lemaire1@gmail.com',
    license='MIT',
    packages=['usnm2p'],
    install_requires=get_requirements(),
    zip_safe=False
)