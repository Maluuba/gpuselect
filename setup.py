from distutils.core import setup, Extension
from pip.req import parse_requirements
deps = ["https://github.com/fbcotter/py3nvml/archive/master.zip#egg=py3nvml-0.001"]
reqs = [
    "nvidia-ml-py ; python_version < '3'",
    "py3nvml==0.001 ; python_version > '2'"
]

setup( name='gpuselect',
      version='1.1',
      description='select a gpu based on utilization',
      author='Hannes Schulz',
      author_email='hannes.schulz@microsoft.com',
      url='http://github.com/Maluuba/gpuselect',
      packages=['gpuselect'],
      install_requires=reqs,
      dependency_links=deps,
      ext_modules=[
          Extension('gpuselect._gpuselect',
                    ['gpuselect/gpuselect.cpp'],
                    include_dirs=['/usr/local/cuda/include'],
                    library_dirs=['/usr/local/cuda/lib64', '/usr/lib64'],
                    libraries=['cudart', 'cuda', 'boost_python'],
                    extra_compile_args=['-fPIC']
                    )
      ]
)
