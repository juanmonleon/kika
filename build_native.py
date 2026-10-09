"""Optional extension build. Set KIKA_BUILD_NATIVE=0 to skip compilation."""
import os
import sys
import warnings
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext
from setuptools.errors import PlatformError

class StrictBuild(build_ext):
    def build_extensions(self):
        msvc=self.compiler.compiler_type=='msvc'
        for extension in self.extensions:
            extension.extra_compile_args=['/O2','/fp:strict'] if msvc else ['-O3','-fno-fast-math','-ffp-contract=off']
            if self.compiler.compiler_type=='mingw32':extension.extra_link_args=['-static-libgcc']
            if sys.platform!='win32':extension.libraries=['m']
        super().build_extensions()

def build():
    if os.environ.get('KIKA_BUILD_NATIVE')=='0':return
    distribution=Distribution({'ext_modules':[Extension(
        'kika.processing.resonances._rm_native',
        ['kika/processing/resonances/_rm_native.c'],depends=['kika/processing/resonances/_rm_kernel.h'],optional=True)]})
    command=StrictBuild(distribution)
    command.inplace=True
    if os.environ.get('KIKA_NATIVE_COMPILER'):command.compiler=os.environ['KIKA_NATIVE_COMPILER']
    command.ensure_finalized()
    try:command.run()
    except (FileNotFoundError,PlatformError) as error:
        # Some compiler initializers fail before setuptools reaches the
        # per-extension optional guard (notably MinGW without gcc on PATH).
        warnings.warn(f'Optional RM extension unavailable; using NumPy: {error}',RuntimeWarning)

if __name__=='__main__':build()
