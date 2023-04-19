from setuptools import setup, find_packages


setup(
    name="frontpage",
    version="0.1.0",
    description="Your frontpage.",
    packages=find_packages(exclude=["notebooks", "tests"]),
)
