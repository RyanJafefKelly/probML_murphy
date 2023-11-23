
from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='my_package',
    version='1.0',
    install_requires=requirements,
    packages=['my_package'],
    entry_points={
        'console_scripts': [
            'my_script=my_package.my_script:main'
        ]
    }
)
