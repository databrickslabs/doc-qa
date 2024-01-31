from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='doc-qa',
    version='0.0.4',
    author='Quinn Leng',
    author_email='quinn.leng@databricks.com',
    description='Question-Answering Evaluation Tool based on Databricks documentation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/databricks/doc_qa',
    packages=find_packages(exclude=["tests", "*tests.*", "*tests"]),
    extras_require={"dev": ["pytest", "pytest-cov", "pytest-mock", "requests_mock"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'tenacity>=8.0.1',
        'tiktoken>=0.3.3',
        'python-dotenv==1.0.0',
        'anthropic==0.13.0',
        'faiss-cpu==1.7.4',
        'databricks-vectorsearch>=0.16',
        'InstructorEmbedding==1.0.1',
        'sentence-transformers==2.2.2',
        'chroma==0.2.0',
        'chromadb==0.4.18',
    ],
)
