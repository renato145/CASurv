import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "fastai<2",
    "bcolz",
    "nibabel"
]

setuptools.setup(
    name="posthocos",
    version="0.0.1",
    author="Renato Hermoza",
    author_email="renato.hermozaaragones@adelaide.edu.au",
    description="Post-hoc Overall Survival Time Prediction from Brain MRI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/renato145/rpsalweaklydet",
    install_requires=requirements,
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
)
