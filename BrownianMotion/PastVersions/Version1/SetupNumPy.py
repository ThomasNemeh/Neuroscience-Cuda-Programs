#from future.utils import iteritems
import sys
sys.path.insert(0,"/home/psimen/anaconda3/lib/python3.6/site-packages")
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


ext = [Extension('GenerateNumPy',
                sources=['GenerateNumPy.pyx'],
                library_dirs=['/usr/local/cuda-9.1/lib64', '/home/thomasnemeh/CudaPrograms/BrownianMotion/PastVersions/Version1'],
                libraries=['cudart', 'CythonSBM'],
                language='c++',
                runtime_library_dirs=['/usr/local/cuda-9.1/lib64', '/home/thomasnemeh/CudaPrograms/PastVersions/Version1'],
                include_dirs = ['/home/psimen/anaconda3/lib/python3.6/site-packages/Cython/Includes', '/usr/local/cuda-9.1/include', '/home/psimen/anaconda3/lib/python3.6/site-packages'])]



setup(name='GenerateNumPy',
	  cmdclass = {'build_ext': build_ext},
      ext_modules = ext)