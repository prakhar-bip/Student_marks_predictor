from setuptools import setup, find_packages
import os
from typing import List

x = "-e ."


def get_req(file_path: str) -> List[str]:

    reqs = []
    with open(file_path) as f:
        reqs = f.readlines()
        reqs = [req.replace("\n", "") for req in reqs]

        if x in reqs:
            reqs.remove(x)

    return reqs


setup(
    name="Student marks Prediction",
    author="Prakhar Batwal",
    author_email="prakharbatwal517@gmail.com",
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_req("requirements.txt"),
)
