import configparser
import json

from fastNLP.loader.base_loader import BaseLoader


class ConfigLoader(BaseLoader):
    """loader for configuration files"""

    def __int__(self, data_name, data_path):
        super(ConfigLoader, self).__init__(data_name, data_path)
        self.config = self.parse(super(ConfigLoader, self).load())

    @staticmethod
    def parse(string):
        raise NotImplementedError

    @staticmethod
    def load_config(file_path, sections):
        """
        :param file_path: the path of config file
        :param sections: the dict of sections
        :return:
        """
        cfg = configparser.ConfigParser()
        cfg.read(file_path)
        for s in sections:
            attr_list = [i for i in type(sections[s]).__dict__.keys() if
                         not callable(getattr(sections[s], i)) and not i.startswith("__")]
            gen_sec = cfg[s]
            for attr in attr_list:
                try:
                    val = json.loads(gen_sec[attr])
                    print(s, attr, val, type(val))
                    assert type(val) == type(getattr(sections[s], attr)), \
                        'type not match, except %s but got %s' % \
                        (type(getattr(sections[s], attr)), type(val))
                    setattr(sections[s], attr, val)
                except Exception as e:
                    # attribute attr in section s did not been set, default val will be used
                    pass
