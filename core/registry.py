class Registry:
    """Registry to dynamically register and instantiate modules."""
    _modules = {}

    @classmethod
    def register(cls, name):
        def wrapper(module_cls):
            cls._modules[name] = module_cls
            return module_cls
        return wrapper

    @classmethod
    def get(cls, name, **kwargs):
        if name not in cls._modules:
            raise KeyError(f"Module {name} not found in registry.")
        return cls._modules[name](**kwargs)

    @classmethod
    def list_modules(cls):
        return list(cls._modules.keys())
