import os
import setuptools

if os.path.exists('requirements.txt'):
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()

setuptools.setup(
    name="pymddrive",
    version="0.0.1",
    author="Rui-Hao Bi",
    author_email="biruihao@westlake.edu.cn",
    description="A convenient personal package for molecular dynamics simulation.",
    long_description="A convenient personal package for molecular dynamics simulation.",
    packages=['pymddrive'],
    entry_points={
        'console_scripts': [
            'simulation_scripts = pymddrive.simulation_scripts:main',
        ]
    },
    requires=requirements,
)

