# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Twirly setup file."""

import os
from setuptools import setup, find_packages

DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(DIR, "requirements.txt")) as f:
    REQUIREMENTS = f.read().splitlines()

with open(os.path.join(DIR, "twirly", "VERSION.txt"), "r") as f:
    VERSION = f.read().rstrip()

with open(os.path.join(DIR, "README.md")) as readme_file:
    README = readme_file.read()

setup(
    name="Twirly",
    version=VERSION,
    description="A Qiskit experiments library for internal and prototype experiments",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.ibm.com/IBM-Q-Software/twirly",
    author="Ian Hincks",
    author_email="hello@qiskit.org",
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit quantum pauli clifford twirl",
    packages=find_packages(exclude=["test*", "examples*"]),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.7",
    zip_safe=False,
)
