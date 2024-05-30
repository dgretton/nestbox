from setuptools import setup, find_packages

setup(
    name="nestbox",
    version="0.1.0",
    author="Dana Gretton",
    author_email="dgretton@mit.edu",
    description="Networked consensus coordinate system alignment",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dgretton/nestbox",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "torch",
    ],
)

