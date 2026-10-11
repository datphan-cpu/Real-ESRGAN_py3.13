import importlib
from basicsr.utils import scandir
from os import path as osp

# automatically scan and import arch modules for registry
# scan all the files that end with '_arch.py' under the archs folder
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = []
for file_name in arch_filenames:
    try:
        _arch_modules.append(importlib.import_module(f'gfpgan.archs.{file_name}'))
    except (ImportError, ModuleNotFoundError):
        # Skip legacy arch modules requiring uncompiled C++ ops if not available
        pass
