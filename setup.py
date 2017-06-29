from setuptools import setup, find_packages

setup(
    name='deblend',
    version='0.2',
    install_requires=['scikit-learn','scipy','numpy', 'matplotlib', 'astropy', 'mpdaf',
                       'adjustText','termcolor'],
    packages=find_packages(),
    zip_safe=False,
    package_data={
        'deblend': ['deblend/*.dat'],                                               
    },
    include_package_data=True,
) 

