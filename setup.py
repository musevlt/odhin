from setuptools import setup, find_packages                                    
                                                                               
setup(                                                                         
    name='deblend',                                                              
    version='0.1',                                                             
    packages=find_packages(),                                                  
    zip_safe=False,                                                            
    package_data={                                                             
        'deblend': ['deblend/*.dat'],                                               
    },                                                                         
    include_package_data=True,                                                 
) 
