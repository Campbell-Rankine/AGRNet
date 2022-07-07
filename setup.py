from setuptools import setup, find_packages

setup(
    name='AGRNet',
    version='1.0.0',
    url='https://github.com/Campbell-Rankine/AGRNet',
    author='Campbell Rankine',
    author_email='campbellrankine@gmail.com',
    description='ProGAN implementation from NVidias 2019 paper',
    packages=find_packages(),    
    install_requires=['python >= 3.9.7', 'pytorch >= 1.10.2', 'torchvision >= 0.10.2'],
)