import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepscribe2",
    version="0.1",
    author="Edward Williams",
    author_email="eddiecwilliams@gmail.com",
    description="Such deep, so scribe, wow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.13.1",
        "torchvision>=0.14.1",
        "pytorch-lightning>=1.9.0",
        "torchmetrics",
        "numpy",
        "timm",
        "editdistance",
        "pandas",
        "scikit-learn",
        "Pillow",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
