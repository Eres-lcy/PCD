from setuptools import find_namespace_packages, setup


setup(
    name="simpler_env",
    version="0.0.1",
    packages=find_namespace_packages(include=["simpler_env", "simpler_env.*"]),
    python_requires=">=3.10",
)
