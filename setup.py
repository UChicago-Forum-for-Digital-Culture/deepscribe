import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepscribe2",
    version="0.1",
    author="Example Author",
    author_email="eddiecwilliams@gmail.com",
    description="Such deep, so scribe, wow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "pytorch-lightning",
        "torchmetrics",
        "numpy",
        "timm",
        "editdistance",
        "pandas",
        "scikit-learn",
        "Pillow",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
