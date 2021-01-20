from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tensorflow-datasets-ko',
    version='0.2',
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='Custom TensorFlow Datasets for Korean Text Data',
    author='Jihee Ryu',
    author_email='chrisjihee@etri.re.kr',
    url='https://github.com/chrisjihee/tensorflow-datasets-ko',
    download_url='https://github.com/chrisjihee/tensorflow-datasets-ko/archive/v0.2.tar.gz',
    install_requires=["tensorflow-datasets[c4]"],
    packages=find_packages(exclude=[]),
    keywords=['TensorFlow', 'Dataset', 'C4', 'Korean', 'Python'],
    python_requires='>=3.6',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
