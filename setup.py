from setuptools import find_packages, setup

setup(
    name='odhin',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=['scikit-learn', 'scipy', 'numpy', 'matplotlib',
                      'astropy', 'mpdaf', 'photutils', 'tqdm'],
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
)
