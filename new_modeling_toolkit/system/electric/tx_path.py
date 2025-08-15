# Should transmission just be a subclass of GenericResource? A lot of the attributes are overlapping, just with different names
from new_modeling_toolkit.system.asset import Asset


class TxPath(Asset):
    pass
