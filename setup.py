# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
# LogCluster — Universal Log Pattern Grouping
# Adapted from Salesforce LogAI
#

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    readme = f.read()

with open("LICENSE.txt", encoding="utf-8") as f:
    license_text = f.read()

setup(
    name="logcluster",
    version="1.0.0",
    description="LogCluster — Universal Log Pattern Grouping",
    long_description_content_type="text/markdown",
    long_description=readme,
    author="LogCluster Contributors (Adapted from Salesforce LogAI)",
    author_email="logcluster@example.com",
    python_requires=">=3.8.0,<4",
    install_requires=[
        "dash>=2.5.1",
        "dash-bootstrap-components>=1.2.1",
        "plotly>=5.9.0",
        "pandas>=1.2.0",
        "numpy>=1.21.4",
        "scikit-learn>=1.0.1",
        "nltk>=3.6.5",
        "attrs>=21.2.0",
        "PyYAML>=6.0",
        "cachetools>=4.2.4",
    ],
    license=license_text,
    packages=["src"],
    include_package_data=True,
)
