from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow-gpu==2.5.3', 'Pillow==5.4.1', 'h5py', 'matplotlib', 'PyMaxflow']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)


