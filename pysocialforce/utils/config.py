"""Config"""
from pathlib import Path
from typing import Dict
import toml


class Config:
    """Config loading and updating
    Attribute
    -------------
    config: dict

    Methods
    -------------
    from_dict: update from a dict
    load_config: update from file
    sub_config: return a sub dict wrapped in Config()
    """

    def __init__(self, config=None) -> None:
        self.config = {}
        if config:
            self.config = config

    def from_dict(self, config: Dict) -> None:
        """Update from dict"""
        self.config.update(config)

    def load_config(self, filename: str) -> None:
        """update from file"""
        user_config = toml.load(filename)
        self.from_dict(user_config)

    def sub_config(self, field_name: str) -> "Config":
        """return a sub dict wrapped in Config()"""
        sub_dict = self.config.get(field_name)
        if isinstance(sub_dict, dict):
            return Config(sub_dict)
        return Config()

    def __call__(self, entry: str, default=None):
        return self.config.get(entry) or default


class DefaultConfig(Config):
    """Default configs"""

    def __init__(self, default_config="default.toml"):
        config_dir = Path(__file__).resolve().parent
        super().__init__(toml.load(config_dir.joinpath(default_config)))
