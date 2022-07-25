import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    "fastai=2.5.2",
]

setuptools.setup(
    name="casurv",
    version="0.0.1",
    author="Renato Hermoza",
    author_email="renato.hermozaaragones@adelaide.edu.au",
    description="Censor-aware Semi-supervised Learning for Survival Time Prediction from Medical Images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/renato145/CASurv",
    install_requires=requirements,
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
)
