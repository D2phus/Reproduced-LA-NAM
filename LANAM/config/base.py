from types import SimpleNamespace


class Config(SimpleNamespace):
    """
    @article{agarwal2020neural,
      title={Neural additive models: Interpretable machine learning with neural nets},
      author={Agarwal, Rishabh and Frosst, Nicholas and Zhang, Xuezhou and Caruana, Rich and Hinton, Geoffrey E},
      journal={arXiv preprint arXiv:2004.13912},
      year={2020}
    }
    
    Wrapper around SimpleNamespace.
    allows dot notation attribute access."""

    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return Config(**entry)

        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, Config(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))

    def update(self, **kwargs):
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, Config(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))
            else:
                setattr(self, key, val)
                
    def get_dict(self):
        """
        return the dict type config
        """
        return vars(self)