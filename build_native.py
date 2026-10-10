"""Build the compiled resonance kernel. Required: a failed build is an error.

The app and generated scripts reconstruct with it; without it the NumPy
reference path is correct but too slow for whole actinides, and a silent
fallback is how a package ships without it. ``KIKA_BUILD_NATIVE=0`` skips the
build explicitly (pure-Python development only). ``KIKA_NATIVE_COMPILER``
forces a compiler; on Windows without Visual Studio, MinGW ``gcc`` on PATH is
tried automatically.
"""
import os
import shutil
import sys
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext
from setuptools.errors import CCompilerError, PlatformError

class StrictBuild(build_ext):
    def build_extensions(self):
        msvc=self.compiler.compiler_type=='msvc'
        for extension in self.extensions:
            extension.extra_compile_args=['/O2','/fp:strict'] if msvc else ['-O3','-fno-fast-math','-ffp-contract=off']
            if self.compiler.compiler_type=='mingw32':extension.extra_link_args=['-static-libgcc']
            if sys.platform!='win32':extension.libraries=['m']
        super().build_extensions()

def _run(compiler=None):
    distribution=Distribution({'ext_modules':[Extension(
        'kika.processing.resonances._rm_native',
        ['kika/processing/resonances/_rm_native.c'],depends=['kika/processing/resonances/_rm_kernel.h'])]})
    command=StrictBuild(distribution)
    command.inplace=True
    if compiler:command.compiler=compiler
    command.ensure_finalized()
    command.run()

def build():
    if os.environ.get('KIKA_BUILD_NATIVE')=='0':return
    forced=os.environ.get('KIKA_NATIVE_COMPILER')
    try:
        _run(forced)
    except (FileNotFoundError,PlatformError,CCompilerError) as error:
        if forced or sys.platform!='win32' or not shutil.which('gcc'):
            raise RuntimeError('kika requires its compiled resonance kernel and it did not build: '
                f'{error}. Install a C compiler (MSVC, gcc or clang), or set '
                'KIKA_BUILD_NATIVE=0 to skip it knowingly.') from error
        _run('mingw32')

if __name__=='__main__':build()
