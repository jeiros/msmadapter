from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read()

requirements = list(filter(None, requirements.split('\n')))

setup(
    name='MSMadapter',
    version='0.1',
    packages=find_packages(),
    license='MIT',
    author='Juan Eiros',
    author_email='jeiroz@gmail.com',
    package_data={
        'MSMadapter' : ['README.md', 'requirements.txt', 'templates/*']
    },
    install_requires=requirements,
    include_package_data=True
)
