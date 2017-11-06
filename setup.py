from setuptools import setup, find_packages
package_name = 'msmadapter'
version = '0.1'
setup(
    name=package_name,
    version=version,
    description='Automate adaptive sampling with AMBER and MSMs',
    packages=find_packages(),
    license='MIT',
    author='Juan Eiros',
    author_email='jeiroz@gmail.com',
    include_package_data=True,
    url='https://github.com/jeiros/{}'.format(package_name),
    download_url='https://github.com/jeiros/{}/archive/{}.tar.gz'.format(package_name, version)
)
