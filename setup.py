import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="m2mcluster",
    version="0.1.dev1",
    author="Jeremy J. Webb",
    author_email="webb@astro.utoronto.ca",
    description="A python packaged for m2m modelling of star clusters",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    packages=["m2mcluster"],
    setup_requires=['amuse-framework','numpy>=1.8','scipy'],
    install_requires=['galpy','seaborn'],
    )
