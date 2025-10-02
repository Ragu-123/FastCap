 
from setuptools import setup, find_packages

# Read the contents of your requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='fastcap',
    version='0.1.0',
    author='RAGUNATH R',
    author_email='ragunathravi73@gmail.com',
    description='FastCap: An Efficient Image Captioning Model',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Ragu-123/FastCap.git',
    license='GNU General Public License v3 (GPLv3)',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    keywords='image-captioning, computer-vision, natural-language-processing, deep-learning, pytorch',
)
