"""Import-only stand-in for transformer_engine.

Lets Megatron-LM import on a box without TE so the checkpoint-READING path
(convert/mg_cpu_loader.py) can run; that path never instantiates a TE module.
Every submodule `transformer_engine.x.y` is synthesised on demand and every
attribute resolves to _Stub, which raises on use. Any real call fails loudly.
"""
import importlib.abc, importlib.machinery, sys, types
__version__ = "2.14.1"  # PEP 440 so megatron.core.utils.get_te_version() parses it
__path__ = []  # namespace-like package; submodules come from the finder below

class _Meta(type):
    # te.pytorch.ops.Sequential, te.pytorch.tensor.Float8Tensor, ...: any attribute
    # of a stub resolves to the stub class again, so import-time references work.
    def __getattr__(cls, name):
        if name.startswith("__"): raise AttributeError(name)
        return cls
    def __call__(cls, *a, **k): raise RuntimeError("transformer_engine is an import-only stub here")
class _Stub(metaclass=_Meta):
    def __init_subclass__(cls, **k): pass
    def __class_getitem__(cls, item): return cls

def __getattr__(name):
    if name.startswith("__"): raise AttributeError(name)
    return _Stub

class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("transformer_engine."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None
    def create_module(self, spec):
        m = types.ModuleType(spec.name); m.__path__ = []; m.__version__ = __version__
        m.__getattr__ = __getattr__; m._Stub = _Stub
        return m
    def exec_module(self, module): pass
sys.meta_path.insert(0, _Finder())
