from setuptools import setup, find_packages

setup(
    name="cnn_base",
    version="0.1.0",
    description="Base for a model for image prediction with CNN using Keras.",
    author="Meet Vyas, Prince Koladiya",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.12",
        "keras>=2.12",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "kagglehub",
        "pillow",
    ],
    extras_require={
        "viz": [
            "opencv-python>=4.8.0",
            "umap-learn>=0.5.4",
            "plotly>=5.16.0",
            "lime",
            "shap",
        ]
    },
    python_requires=">=3.8",
)
