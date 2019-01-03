#from future.utils import iteritems
import sys
sys.path.insert(0,"/opt/anaconda/lib/python3.6/site-packages")
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


ext = [Extension('GenerateNumPy3',
                sources=['GenerateNumPy3.pyx'],
                library_dirs=['/usr/local/cuda-9.1/lib64', '/home/thomasnemeh/CudaPrograms/BrownianMotion'],
                libraries=['cudart', 'gpubm'],
                language='c++',
                runtime_library_dirs=['/usr/local/cuda-9.1/lib64', '/home/thomasnemeh/CudaPrograms/BrownianMotion'],
                include_dirs = [])]



setup(name='GenerateNumPy3',
	  cmdclass = {'build_ext': build_ext},
      ext_modules = ext)