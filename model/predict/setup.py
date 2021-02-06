from setuptools import setup, find_packages


setup(
    name='predict',
    version='0.1',
    install_requires=[],
    packages=find_packages(),
    include_package_data=True,
    description='Serves next word predictions via REST API'
)
