from setuptools import setup, find_packages                                    
                                                                               
setup(                                                                         
    name='deblend',                                                              
    version='0.1',                                                             
    install_requires=['numpy', 'matplotlib', 'astropy', 
                       'scipy','adjustText','termcolor'],
    packages=find_packages(),                                                  
    zip_safe=False,                                                            
    package_data={                                                             
        'deblend': ['deblend/*.dat'],                                               
    },                                                                         
    include_package_data=True,                                                 
) 
