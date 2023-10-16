from setuptools import setup, find_packages

setup(
    name="KATE",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "keras",  # add other dependencies if needed
    ],
    author="Clément Carré",
    author_email="clement.carre@bionomeex.com",
    description="KATE is Keras Automated Testing Engine.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ClementCarre-BionomeeX/KATE",  # if you have a repo for this
)
