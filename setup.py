# flake8: noqa
import os
from setuptools import setup, Extension, find_packages
import pathlib

# workaround for numpy and Cython install dependency
# the solution is from https://stackoverflow.com/a/54138355
def my_build_ext(pars):
    # import delayed:
    from setuptools.command.build_ext import build_ext as _build_ext
    class build_ext(_build_ext):
        def finalize_options(self):
            # got error `'dict' object has no attribute '__NUMPY_SETUP__'`
            # Follow this solution https://github.com/SciTools/cf-units/blob/master/setup.py#L99
            def _set_builtin(name, value):
                if isinstance(__builtins__, dict):
                    __builtins__[name] = value
                else:
                    setattr(__builtins__, name, value)

            _build_ext.finalize_options(self)
            # Prevent numpy from thinking it is still in its setup process:
            _set_builtin('__NUMPY_SETUP__', False)
            import numpy
            self.include_dirs.append(numpy.get_include())

    #object returned:
    return build_ext(pars)


here = os.path.abspath(os.path.dirname(__file__))
# Get the long description from the README file
#with open(os.path.join(here, 'README.md')) as f:
#    long_description = f.read()
# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()



if os.name == 'nt':
    extra_compile_args = ["-Ox"]
else:
    extra_compile_args = ['-std=c++0x', '-pthread', '-O3']

topn_ext = Extension(
    'red_string_grouper.topn.topn',
    sources=[
        './red_string_grouper/topn/topn_threaded.pyx',
        './red_string_grouper/topn/topn_parallel.cpp'
    ],
    extra_compile_args=extra_compile_args,
    language='c++'
)

array_wrappers_ext = Extension(
    'red_string_grouper.sparse_dot_topn.array_wrappers',
    sources=[
        './red_string_grouper/sparse_dot_topn/array_wrappers.pyx',
    ],
    extra_compile_args=extra_compile_args,
    language='c++'
)

sparse_dot_topn_original_ext = Extension(
    'red_string_grouper.sparse_dot_topn.sparse_dot_topn',
    sources=[
        './red_string_grouper/sparse_dot_topn/sparse_dot_topn.pyx',
        './red_string_grouper/sparse_dot_topn/sparse_dot_topn_source.cpp'
    ],
    extra_compile_args=extra_compile_args,
    language='c++'
)

sparse_dot_topn_threaded_ext = Extension(
    'red_string_grouper.sparse_dot_topn.sparse_dot_topn_threaded',
    sources=[
        './red_string_grouper/sparse_dot_topn/sparse_dot_topn_threaded.pyx',
        './red_string_grouper/sparse_dot_topn/sparse_dot_topn_source.cpp',
        './red_string_grouper/sparse_dot_topn/sparse_dot_topn_parallel.cpp'
    ],
    extra_compile_args=extra_compile_args,
    language='c++'
)


setup(
    name='red_string_grouper',
    version='0.0.6',
    description='Row Equivalence Discoverer (red) based on string_grouper. '
    'This package finds similarities between rows of a table.',
    keywords='record-linkage string-comparison cosine-similarity tf-idf'
    'string_grouper sparse_dot_topn python cython',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/ParticularMiner/red_string_grouper',
    download_url='https://github.com/ParticularMiner/red_string_grouper/'
                 'archive/refs/tags/v0.0.6.tar.gz',
    author='Particular Miner', 
    author_email='particularminer@fake.com',
    license='MIT',
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=42',
        'cython>=0.29.15',
        'string_grouper>=0.5.0'
    ],
    install_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=42',
        'cython>=0.29.15',
        'numpy>=1.16.6', # select this version for Py2/3 compatible
        'string_grouper>=0.5.0'
    ],
    zip_safe=False,
    packages=find_packages(),
    cmdclass={'build_ext': my_build_ext},
    ext_modules=[
        topn_ext,
        array_wrappers_ext,
        sparse_dot_topn_original_ext,
        sparse_dot_topn_threaded_ext
    ],
    package_data = {
        'red_string_grouper': ['./red_string_grouper/sparse_dot_topn/*.pxd']
    },
    include_package_data=True,    
)