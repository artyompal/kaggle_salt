from setuptools import setup

setup(
   name='lenin',
   version='0.1',
   author=['brainopia', 'gazay'],
   packages=['lenin', 'lenin.augmentors', 'lenin.datasets', 'lenin.preloader'],
   install_requires=['torchbearer', 'scikit-learn'],
)
