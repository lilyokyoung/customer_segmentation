"""
Setup configuration for Customer Segmentation AI Agent
"""
from setuptools import setup, find_packages

# Core requirements (always needed)
core_requirements = [
    "numpy>=1.21.0,<1.25.0",
    "pandas>=1.5.0,<2.1.0", 
    "scikit-learn>=1.2.0,<1.4.0",
    "scipy>=1.9.0,<1.12.0",
    "matplotlib>=3.5.0,<3.8.0",
    "seaborn>=0.11.0,<0.13.0",
    "pyyaml>=6.0.0,<6.1.0",
    "joblib>=1.2.0,<1.4.0"
]

# Optional ML models (can be installed separately)
ml_requirements = [
    "xgboost>=1.6.0,<2.0.0",
    "lightgbm>=3.3.0,<4.1.0", 
    "catboost>=1.1.0,<1.3.0",
    "imbalanced-learn>=0.10.0,<0.12.0"
]

# Visualization extras
viz_requirements = [
    "plotly>=5.10.0,<5.16.0"
]

setup(
    name="customer-segmentation-ai-agent",
    version="1.0.0",
    description="Enhanced AI-Agent Customer Segmentation with Advanced Analytics",
    author="Lily",
    url="https://github.com/lilyokyoung/customer_segmentation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8,<3.12",
    
    # Core requirements only for basic installation
    install_requires=core_requirements,
    
    # Optional extras
    extras_require={
        "ml": ml_requirements,
        "viz": viz_requirements,
        "full": ml_requirements + viz_requirements
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ]
)