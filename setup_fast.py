import os
from numpy.distutils.core import setup

DESCRIPTION = "Python implementation of Friedman's Supersmoother"
LONG_DESCRIPTION = """
SuperSmoother in Python
=======================
This is an efficient implementation of Friedman's SuperSmoother based in
Python. It makes use of numpy for fast numerical computation.

For more information, see the github project page:
http://github.com/jakevdp/supersmoother
"""
NAME = "supersmoother"
AUTHOR = "Jake VanderPlas"
AUTHOR_EMAIL = "jakevdp@uw.edu"
MAINTAINER = "Jake VanderPlas"
MAINTAINER_EMAIL = "jakevdp@uw.edu"
URL = 'http://github.com/jakevdp/supersmoother'
DOWNLOAD_URL = 'http://github.com/jakevdp/supersmoother'
LICENSE = 'BSD 3-clause'

import supersmoother
VERSION = supersmoother.__version__

def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('supersmoother')

    return config

setup(configuration=configuration,
      name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['supersmoother',
                'supersmoother.tests',
            ],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4'],
     )
