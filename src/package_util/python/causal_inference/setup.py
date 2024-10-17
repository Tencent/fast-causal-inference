from distutils.core import setup
from setuptools import find_packages

with open("README.md", "r") as f:
    long_description = f.read()

requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        if "#" not in line:
            requirements.append(line.strip('\n').strip())

setup(name='fast-causal-inference',
      version='3.0.5',
      description='fast causal inference package',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='',
      python_requires='>=3.6.0',
      author=['cooperxiong', 'easonsliu', 'fhbai', 'broccozhang'],
      author_email=['cooperxiong@tencent.com', 'easonsliu@tencent.com', 'fhbai@tencent.com', 'broccozhang@tencent.com'],
      install_requires=requirements,
      license='Apache License 2.0',
      packages=find_packages(exclude=("test",)),
      include_package_data=True,
      platforms=["all"],
      classifiers=[
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8'
      ],
      )
