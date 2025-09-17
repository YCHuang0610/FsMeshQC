from setuptools import setup, find_packages

setup(
    name="FsMeshQC",
    version="0.1.0",
    packages=find_packages(),
    description="FreeSurfer Mesh Quality Control Tool",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="ychunhuang", 
    author_email="ychunhuang@foxmail.com", 
    url="https://github.com/ychunhuang/FsMeshQC",
    install_requires=[
        "numpy",
        "nibabel",
        "pandas",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    entry_points={
        "console_scripts": [
            "fsmeshqc=FsMeshQC.main:main",
        ],
    },
)