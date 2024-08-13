from setuptools import setup, find_packages

setup(
    name="grfcsc",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",     
        "matplotlib",
        "scipy",
        "cupy",
    ],
    author="Ka Hei Choi",
    author_email="K.Choi@physik.uni-muenchen.de",
    description="A package for Gaussian Random Field simulations and statistics analysis in De Sitter space and Radiation Dominated Universe.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/khchoi-lmu-physik/grf_qftcs", 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    keywords="random field simulation"
)
