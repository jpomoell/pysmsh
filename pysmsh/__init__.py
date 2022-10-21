from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass  # package is not installed

from .core.mesh import Mesh
from .core.dinterval import DiscreteInterval
from .core.field import Field
import pysmsh.core.colocation as colocation
