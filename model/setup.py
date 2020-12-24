from setuptools import find_packages
from setuptools import setup

# with open('requirements.txt') as f:
#     requirements = f.readlines()
#
# REQUIRED_PACKAGES = [x.strip() for x in requirements]
REQUIRED_PACKAGES = []

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Next Word Model training application package.'
)
