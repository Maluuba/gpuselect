from distutils.core import setup, Extension
from pip.req import parse_requirements
install_reqs = parse_requirements('requirements.txt', session=False)
# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]

setup( name='gpuselect',
      version='1.0',
      description='select a gpu based on utilization',
      author='Hannes Schulz',
      author_email='hannes.schulz@maluuba.com',
      url='http://github.com/temporaer/gpuselect',
      packages=['gpuselect'],
      install_requires=reqs,
      ext_modules=[
          Extension('gpuselect._gpuselect',
                    ['gpuselect/gpuselect.cpp'],
                    include_dirs=['/usr/local/cuda/include', '/usr/include/python2.7'],
                    library_dirs=['/usr/local/cuda/lib64', '/usr/lib64'],
                    libraries=['cudart', 'cuda', 'boost_python-py27', 'python2.7'],
                    extra_compile_args=['-fPIC']
                    )
      ]
)
