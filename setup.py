from setuptools import setup, find_packages

setup(
    name="mylib",
    version="0.1.0",
    description="Custom Keras fine-tuning library with flexible callbacks and model editing.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.12",
        "keras>=2.12",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn"
    ],
    python_requires=">=3.8",
)
