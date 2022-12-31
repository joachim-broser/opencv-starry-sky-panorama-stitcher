#!/bin/python3.10

from descartes.patch import Polygon
from matplotlib.path import Path
from numpy import asarray, concatenate, ones
def PolygonPathFixed(polygon):
    """Constructs a compound matplotlib path from a Shapely or GeoJSON-like
    geometric object"""

    def coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(getattr(ob, 'coords', None) or ob)
        vals = ones(n, dtype=Path.code_type) * Path.LINETO
        vals[0] = Path.MOVETO
        return vals

    if hasattr(polygon, 'geom_type'):  # Shapely
        ptype = polygon.geom_type
        if ptype == 'Polygon':
            polygon = [Polygon(polygon)]
        elif ptype == 'MultiPolygon':
            polygon = [Polygon(p) for p in polygon]
        else:
            raise ValueError(
                "A polygon or multi-polygon representation is required")

    else:  # GeoJSON
        polygon = getattr(polygon, '__geo_interface__', polygon)
        ptype = polygon["type"]
        if ptype == 'Polygon':
            polygon = [Polygon(polygon)]
        elif ptype == 'MultiPolygon':
            polygon = [Polygon(p) for p in polygon['coordinates']]
        else:
            raise ValueError(
                "A polygon or multi-polygon representation is required")

    vertices = concatenate([
        concatenate([asarray(t.exterior.xy)] +
                    [asarray(r)[:, :2] for r in t.interiors])
        for t in polygon])
    codes = concatenate([
        concatenate([coding(t.exterior)] +
                    [coding(r) for r in t.interiors]) for t in polygon])

    return Path(vertices.T, codes)


from matplotlib.patches import PathPatch

def PolygonPatch(polygon, **kwargs):
    """Constructs a matplotlib patch from a geometric object

    The `polygon` may be a Shapely or GeoJSON-like object with or without holes.
    The `kwargs` are those supported by the matplotlib.patches.Polygon class
    constructor. Returns an instance of matplotlib.patches.PathPatch.

    Example (using Shapely Point and a matplotlib axes):

      >>> b = Point(0, 0).buffer(1.0)
      >>> patch = PolygonPatch(b, fc='blue', ec='blue', alpha=0.5)
      >>> axis.add_patch(patch)

    """
    return PathPatch(PolygonPathFixed(polygon), **kwargs)
