from setuptools import setup, find_packages

setup(
    name='odhin',
    version='1.0-beta2',
    install_requires=['scikit-learn', 'scipy', 'numpy', 'matplotlib',
                      'astropy', 'mpdaf', 'photutils', 'tqdm'],
    packages=find_packages(),
    zip_safe=False,
    package_data={
        'odhin': ['data/*.dat'],
    },
    include_package_data=True,
)
