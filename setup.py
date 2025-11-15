import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quickdraw",
    version="0.0.0",
    description="cortex",
    author="mayukhmainak2000@gmail.com, mayukh@gatech.edu",
    author_email="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mayukhdeb/quickdraw",
    packages=setuptools.find_packages(),
    install_requires=None,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)