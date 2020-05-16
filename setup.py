import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="corgie",
    version="0.0.1",
    author="Sergiy Popovych",
    author_email="sergiy.popovich@gmail.com",
    description="Connectomics Registration General Inference Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seugnlab/corgie",
    include_package_data=True,
    package_data={'': ['*.py']},
    install_requires=[
      'torch',
      'torchvision',
      'numpy',
      'six',
      'mazepa',
      'click',
      'click-option-group'
    ],
    entry_points={
        "console_scripts": [
            "corgie = corgie.main:cli",
            "corgie-help = corgie.main:helper",
        ],
    },
    packages=setuptools.find_packages(),
)
