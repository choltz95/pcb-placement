# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.12
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_PcbDB')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_PcbDB')
    _PcbDB = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_PcbDB', [dirname(__file__)])
        except ImportError:
            import _PcbDB
            return _PcbDB
        try:
            _mod = imp.load_module('_PcbDB', fp, pathname, description)
        finally:
            if fp is not None:
                fp.close()
        return _mod
    _PcbDB = swig_import_helper()
    del swig_import_helper
else:
    import _PcbDB
del _swig_python_version_info

try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except __builtin__.Exception:
    class _object:
        pass
    _newclass = 0

class SwigPyIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SwigPyIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SwigPyIterator, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _PcbDB.delete_SwigPyIterator
    __del__ = lambda self: None

    def value(self) -> "PyObject *":
        return _PcbDB.SwigPyIterator_value(self)

    def incr(self, n: 'size_t'=1) -> "swig::SwigPyIterator *":
        return _PcbDB.SwigPyIterator_incr(self, n)

    def decr(self, n: 'size_t'=1) -> "swig::SwigPyIterator *":
        return _PcbDB.SwigPyIterator_decr(self, n)

    def distance(self, x: 'SwigPyIterator') -> "ptrdiff_t":
        return _PcbDB.SwigPyIterator_distance(self, x)

    def equal(self, x: 'SwigPyIterator') -> "bool":
        return _PcbDB.SwigPyIterator_equal(self, x)

    def copy(self) -> "swig::SwigPyIterator *":
        return _PcbDB.SwigPyIterator_copy(self)

    def next(self) -> "PyObject *":
        return _PcbDB.SwigPyIterator_next(self)

    def __next__(self) -> "PyObject *":
        return _PcbDB.SwigPyIterator___next__(self)

    def previous(self) -> "PyObject *":
        return _PcbDB.SwigPyIterator_previous(self)

    def advance(self, n: 'ptrdiff_t') -> "swig::SwigPyIterator *":
        return _PcbDB.SwigPyIterator_advance(self, n)

    def __eq__(self, x: 'SwigPyIterator') -> "bool":
        return _PcbDB.SwigPyIterator___eq__(self, x)

    def __ne__(self, x: 'SwigPyIterator') -> "bool":
        return _PcbDB.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n: 'ptrdiff_t') -> "swig::SwigPyIterator &":
        return _PcbDB.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n: 'ptrdiff_t') -> "swig::SwigPyIterator &":
        return _PcbDB.SwigPyIterator___isub__(self, n)

    def __add__(self, n: 'ptrdiff_t') -> "swig::SwigPyIterator *":
        return _PcbDB.SwigPyIterator___add__(self, n)

    def __sub__(self, *args) -> "ptrdiff_t":
        return _PcbDB.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self
SwigPyIterator_swigregister = _PcbDB.SwigPyIterator_swigregister
SwigPyIterator_swigregister(SwigPyIterator)

class kicadPcbDataBase(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, kicadPcbDataBase, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, kicadPcbDataBase, name)
    __repr__ = _swig_repr

    def __init__(self, fileName: 'std::string'):
        this = _PcbDB.new_kicadPcbDataBase(fileName)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_destroy__ = _PcbDB.delete_kicadPcbDataBase
    __del__ = lambda self: None

    def printLayer(self) -> "void":
        return _PcbDB.kicadPcbDataBase_printLayer(self)

    def printNet(self) -> "void":
        return _PcbDB.kicadPcbDataBase_printNet(self)

    def printInst(self) -> "void":
        return _PcbDB.kicadPcbDataBase_printInst(self)

    def printComp(self) -> "void":
        return _PcbDB.kicadPcbDataBase_printComp(self)

    def printNetclass(self) -> "void":
        return _PcbDB.kicadPcbDataBase_printNetclass(self)

    def printPcbRouterInfo(self) -> "void":
        return _PcbDB.kicadPcbDataBase_printPcbRouterInfo(self)

    def printFile(self) -> "void":
        return _PcbDB.kicadPcbDataBase_printFile(self)

    def printSegment(self) -> "void":
        return _PcbDB.kicadPcbDataBase_printSegment(self)

    def printUnconnectedPins(self) -> "void":
        return _PcbDB.kicadPcbDataBase_printUnconnectedPins(self)

    def printKiCad(self, *args) -> "void":
        return _PcbDB.kicadPcbDataBase_printKiCad(self, *args)

    def printNodes(self) -> "void":
        return _PcbDB.kicadPcbDataBase_printNodes(self)

    def printLockedInst(self) -> "void":
        return _PcbDB.kicadPcbDataBase_printLockedInst(self)

    def printDesignStatistics(self) -> "void":
        return _PcbDB.kicadPcbDataBase_printDesignStatistics(self)

    def printRoutedSegmentsWLAndNumVias(self) -> "void":
        return _PcbDB.kicadPcbDataBase_printRoutedSegmentsWLAndNumVias(self)

    def buildKicadPcb(self) -> "bool":
        return _PcbDB.kicadPcbDataBase_buildKicadPcb(self)

    def removeRoutedSegmentsAndVias(self) -> "void":
        return _PcbDB.kicadPcbDataBase_removeRoutedSegmentsAndVias(self)

    def getPcbRouterInfo(self, arg2: 'std::vector< std::set< std::pair< double,double > >,std::allocator< std::set< std::pair< double,double > > > > *') -> "bool":
        return _PcbDB.kicadPcbDataBase_getPcbRouterInfo(self, arg2)

    def getPinShapeRelativeCoordsToModule(self, pad: 'padstack', inst: 'instance', coords: 'points_2d const &', coordsRe: 'points_2d *') -> "void":
        return _PcbDB.kicadPcbDataBase_getPinShapeRelativeCoordsToModule(self, pad, inst, coords, coordsRe)

    def getPinPosition(self, *args) -> "bool":
        return _PcbDB.kicadPcbDataBase_getPinPosition(self, *args)

    def getPinPositionX(self, p: 'Pin') -> "double":
        return _PcbDB.kicadPcbDataBase_getPinPositionX(self, p)

    def getPinPositionY(self, p: 'Pin') -> "double":
        return _PcbDB.kicadPcbDataBase_getPinPositionY(self, p)

    def getCompBBox(self, compId: 'int const', bBox: 'point_2d *') -> "bool":
        return _PcbDB.kicadPcbDataBase_getCompBBox(self, compId, bBox)

    def getCompBBoxW(self, compId: 'int const') -> "double":
        return _PcbDB.kicadPcbDataBase_getCompBBoxW(self, compId)

    def getCompBBoxH(self, compId: 'int const') -> "double":
        return _PcbDB.kicadPcbDataBase_getCompBBoxH(self, compId)

    def getPinLayer(self, instId: 'int const &', padStackId: 'int const &') -> "std::vector< int,std::allocator< int > >":
        return _PcbDB.kicadPcbDataBase_getPinLayer(self, instId, padStackId)

    def getPadstackRotatedWidthAndHeight(self, inst: 'instance', pad: 'padstack', width: 'double &', height: 'double &') -> "void":
        return _PcbDB.kicadPcbDataBase_getPadstackRotatedWidthAndHeight(self, inst, pad, width, height)

    def getComponent(self, *args) -> "component &":
        return _PcbDB.kicadPcbDataBase_getComponent(self, *args)

    def getInstance(self, *args) -> "instance &":
        return _PcbDB.kicadPcbDataBase_getInstance(self, *args)

    def getNet(self, *args) -> "net &":
        return _PcbDB.kicadPcbDataBase_getNet(self, *args)

    def getNetclass(self, id: 'int const') -> "netclass &":
        return _PcbDB.kicadPcbDataBase_getNetclass(self, id)

    def getFileName(self) -> "std::string":
        return _PcbDB.kicadPcbDataBase_getFileName(self)

    def getInstances(self) -> "std::vector< instance,std::allocator< instance > > &":
        return _PcbDB.kicadPcbDataBase_getInstances(self)

    def getComponents(self) -> "std::vector< component,std::allocator< component > > &":
        return _PcbDB.kicadPcbDataBase_getComponents(self)

    def getNets(self) -> "std::vector< net,std::allocator< net > > &":
        return _PcbDB.kicadPcbDataBase_getNets(self)

    def getUnconnectedPins(self) -> "std::vector< Pin,std::allocator< Pin > > &":
        return _PcbDB.kicadPcbDataBase_getUnconnectedPins(self)

    def getNetclasses(self) -> "std::vector< netclass,std::allocator< netclass > > &":
        return _PcbDB.kicadPcbDataBase_getNetclasses(self)

    def isInstanceId(self, id: 'int const') -> "bool":
        return _PcbDB.kicadPcbDataBase_isInstanceId(self, id)

    def isComponentId(self, id: 'int const') -> "bool":
        return _PcbDB.kicadPcbDataBase_isComponentId(self, id)

    def isNetId(self, id: 'int const') -> "bool":
        return _PcbDB.kicadPcbDataBase_isNetId(self, id)

    def isNetclassId(self, id: 'int const') -> "bool":
        return _PcbDB.kicadPcbDataBase_isNetclassId(self, id)

    def getNumCopperLayers(self) -> "int":
        return _PcbDB.kicadPcbDataBase_getNumCopperLayers(self)

    def getLayerId(self, layerName: 'std::string const &') -> "int":
        return _PcbDB.kicadPcbDataBase_getLayerId(self, layerName)

    def getLayerName(self, layerId: 'int const') -> "std::string":
        return _PcbDB.kicadPcbDataBase_getLayerName(self, layerId)

    def getCopperLayers(self) -> "std::map< int,std::string > &":
        return _PcbDB.kicadPcbDataBase_getCopperLayers(self)

    def isCopperLayer(self, *args) -> "bool":
        return _PcbDB.kicadPcbDataBase_isCopperLayer(self, *args)

    def getBoardBoundaryByPinLocation(self, minX: 'double &', maxX: 'double &', minY: 'double &', maxY: 'double &') -> "void":
        return _PcbDB.kicadPcbDataBase_getBoardBoundaryByPinLocation(self, minX, maxX, minY, maxY)

    def addClearanceDrc(self, obj1: 'Object &', obj2: 'Object &') -> "void":
        return _PcbDB.kicadPcbDataBase_addClearanceDrc(self, obj1, obj2)

    def getBoardBoundaryByEdgeCuts(self, minX: 'double &', maxX: 'double &', minY: 'double &', maxY: 'double &') -> "void":
        return _PcbDB.kicadPcbDataBase_getBoardBoundaryByEdgeCuts(self, minX, maxX, minY, maxY)

    def printClearanceDrc(self) -> "void":
        return _PcbDB.kicadPcbDataBase_printClearanceDrc(self)

    def getInstancesCount(self) -> "int":
        return _PcbDB.kicadPcbDataBase_getInstancesCount(self)

    def getNumNets(self) -> "int":
        return _PcbDB.kicadPcbDataBase_getNumNets(self)

    def getLargestClearance(self) -> "double":
        return _PcbDB.kicadPcbDataBase_getLargestClearance(self)

    def testInstAngle(self) -> "void":
        return _PcbDB.kicadPcbDataBase_testInstAngle(self)
kicadPcbDataBase_swigregister = _PcbDB.kicadPcbDataBase_swigregister
kicadPcbDataBase_swigregister(kicadPcbDataBase)

class component(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, component, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, component, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _PcbDB.new_component(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_destroy__ = _PcbDB.delete_component
    __del__ = lambda self: None

    def getId(self) -> "int":
        return _PcbDB.component_getId(self)

    def getName(self) -> "std::string &":
        return _PcbDB.component_getName(self)

    def getPadstacks(self) -> "std::vector< padstack,std::allocator< padstack > > &":
        return _PcbDB.component_getPadstacks(self)

    def isPadstackId(self, id: 'int const') -> "bool":
        return _PcbDB.component_isPadstackId(self, id)

    def getPadstackId(self, name: 'std::string const &', id: 'int *') -> "bool":
        return _PcbDB.component_getPadstackId(self, name, id)

    def getPadstack(self, *args) -> "bool":
        return _PcbDB.component_getPadstack(self, *args)

    def hasFrontCrtyd(self) -> "bool":
        return _PcbDB.component_hasFrontCrtyd(self)

    def hasBottomCrtyd(self) -> "bool":
        return _PcbDB.component_hasBottomCrtyd(self)
component_swigregister = _PcbDB.component_swigregister
component_swigregister(component)

class instance(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, instance, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, instance, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = _PcbDB.new_instance()
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_destroy__ = _PcbDB.delete_instance
    __del__ = lambda self: None

    def getName(self) -> "std::string const &":
        return _PcbDB.instance_getName(self)

    def getComponentId(self) -> "int":
        return _PcbDB.instance_getComponentId(self)

    def getId(self) -> "int":
        return _PcbDB.instance_getId(self)

    def setAngle(self, angle: 'double &') -> "void":
        return _PcbDB.instance_setAngle(self, angle)

    def setX(self, x: 'double &') -> "void":
        return _PcbDB.instance_setX(self, x)

    def setY(self, y: 'double &') -> "void":
        return _PcbDB.instance_setY(self, y)

    def setLayer(self, layer: 'int &') -> "void":
        return _PcbDB.instance_setLayer(self, layer)

    def getAngle(self) -> "double":
        return _PcbDB.instance_getAngle(self)

    def getX(self) -> "double":
        return _PcbDB.instance_getX(self)

    def getY(self) -> "double":
        return _PcbDB.instance_getY(self)

    def getW(self) -> "double":
        return _PcbDB.instance_getW(self)

    def getH(self) -> "double":
        return _PcbDB.instance_getH(self)

    def getLayer(self) -> "int":
        return _PcbDB.instance_getLayer(self)

    def isLocked(self) -> "bool":
        return _PcbDB.instance_isLocked(self)

    def isFlipped(self) -> "bool":
        return _PcbDB.instance_isFlipped(self)
instance_swigregister = _PcbDB.instance_swigregister
instance_swigregister(instance)

class track(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, track, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, track, name)
    __repr__ = _swig_repr
    __swig_setmethods__["m_id"] = _PcbDB.track_m_id_set
    __swig_getmethods__["m_id"] = _PcbDB.track_m_id_get
    if _newclass:
        m_id = _swig_property(_PcbDB.track_m_id_get, _PcbDB.track_m_id_set)
    __swig_setmethods__["m_track_radius"] = _PcbDB.track_m_track_radius_set
    __swig_getmethods__["m_track_radius"] = _PcbDB.track_m_track_radius_get
    if _newclass:
        m_track_radius = _swig_property(_PcbDB.track_m_track_radius_get, _PcbDB.track_m_track_radius_set)
    __swig_setmethods__["m_via_radius"] = _PcbDB.track_m_via_radius_set
    __swig_getmethods__["m_via_radius"] = _PcbDB.track_m_via_radius_get
    if _newclass:
        m_via_radius = _swig_property(_PcbDB.track_m_via_radius_get, _PcbDB.track_m_via_radius_set)
    __swig_setmethods__["m_clearance"] = _PcbDB.track_m_clearance_set
    __swig_getmethods__["m_clearance"] = _PcbDB.track_m_clearance_get
    if _newclass:
        m_clearance = _swig_property(_PcbDB.track_m_clearance_get, _PcbDB.track_m_clearance_set)
    __swig_setmethods__["m_pads"] = _PcbDB.track_m_pads_set
    __swig_getmethods__["m_pads"] = _PcbDB.track_m_pads_get
    if _newclass:
        m_pads = _swig_property(_PcbDB.track_m_pads_get, _PcbDB.track_m_pads_set)
    __swig_setmethods__["m_paths"] = _PcbDB.track_m_paths_set
    __swig_getmethods__["m_paths"] = _PcbDB.track_m_paths_get
    if _newclass:
        m_paths = _swig_property(_PcbDB.track_m_paths_get, _PcbDB.track_m_paths_set)

    def __init__(self):
        this = _PcbDB.new_track()
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_destroy__ = _PcbDB.delete_track
    __del__ = lambda self: None
track_swigregister = _PcbDB.track_swigregister
track_swigregister(track)

class netclass(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, netclass, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, netclass, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _PcbDB.new_netclass(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_destroy__ = _PcbDB.delete_netclass
    __del__ = lambda self: None

    def getId(self) -> "int":
        return _PcbDB.netclass_getId(self)

    def getName(self) -> "std::string &":
        return _PcbDB.netclass_getName(self)

    def getClearance(self) -> "double":
        return _PcbDB.netclass_getClearance(self)

    def getTraceWidth(self) -> "double":
        return _PcbDB.netclass_getTraceWidth(self)

    def getViaDia(self) -> "double":
        return _PcbDB.netclass_getViaDia(self)

    def getViaDrill(self) -> "double":
        return _PcbDB.netclass_getViaDrill(self)

    def getMicroViaDia(self) -> "double":
        return _PcbDB.netclass_getMicroViaDia(self)

    def getMicroViaDrill(self) -> "double":
        return _PcbDB.netclass_getMicroViaDrill(self)
netclass_swigregister = _PcbDB.netclass_swigregister
netclass_swigregister(netclass)

class net(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, net, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, net, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _PcbDB.new_net(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def getId(self) -> "int":
        return _PcbDB.net_getId(self)

    def getName(self) -> "std::string &":
        return _PcbDB.net_getName(self)

    def getNetclassId(self) -> "int":
        return _PcbDB.net_getNetclassId(self)

    def addPin(self, _pin: 'Pin') -> "void":
        return _PcbDB.net_addPin(self, _pin)

    def getPins(self) -> "std::vector< Pin,std::allocator< Pin > > &":
        return _PcbDB.net_getPins(self)

    def getSegments(self) -> "std::vector< Segment,std::allocator< Segment > > &":
        return _PcbDB.net_getSegments(self)

    def getVias(self) -> "std::vector< Via,std::allocator< Via > > &":
        return _PcbDB.net_getVias(self)

    def getSegment(self, _id: 'int &') -> "Segment &":
        return _PcbDB.net_getSegment(self, _id)

    def getVia(self, _id: 'int &') -> "Via &":
        return _PcbDB.net_getVia(self, _id)

    def clearSegments(self) -> "void":
        return _PcbDB.net_clearSegments(self)

    def clearVias(self) -> "void":
        return _PcbDB.net_clearVias(self)

    def getSegmentCount(self) -> "int":
        return _PcbDB.net_getSegmentCount(self)

    def addSegment(self, _segment: 'Segment const &') -> "void":
        return _PcbDB.net_addSegment(self, _segment)

    def getViaCount(self) -> "int":
        return _PcbDB.net_getViaCount(self)

    def addVia(self, _via: 'Via const &') -> "void":
        return _PcbDB.net_addVia(self, _via)

    def isDiffPair(self) -> "bool":
        return _PcbDB.net_isDiffPair(self)

    def getDiffPairId(self) -> "int":
        return _PcbDB.net_getDiffPairId(self)

    def isBus(self) -> "bool":
        return _PcbDB.net_isBus(self)
    __swig_destroy__ = _PcbDB.delete_net
    __del__ = lambda self: None
net_swigregister = _PcbDB.net_swigregister
net_swigregister(net)

padType_SMD = _PcbDB.padType_SMD
padType_THRU_HOLE = _PcbDB.padType_THRU_HOLE
padType_CONNECT = _PcbDB.padType_CONNECT
padType_NP_THRU_HOLE = _PcbDB.padType_NP_THRU_HOLE
class Pin(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Pin, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Pin, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _PcbDB.new_Pin(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_destroy__ = _PcbDB.delete_Pin
    __del__ = lambda self: None

    def __eq__(self, p: 'Pin') -> "bool":
        return _PcbDB.Pin___eq__(self, p)

    def getPadstackId(self) -> "int":
        return _PcbDB.Pin_getPadstackId(self)

    def getCompId(self) -> "int":
        return _PcbDB.Pin_getCompId(self)

    def getInstId(self) -> "int":
        return _PcbDB.Pin_getInstId(self)

    def getLayers(self) -> "std::vector< int,std::allocator< int > > const &":
        return _PcbDB.Pin_getLayers(self)
Pin_swigregister = _PcbDB.Pin_swigregister
Pin_swigregister(Pin)

class pad(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, pad, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, pad, name)
    __repr__ = _swig_repr

    def __eq__(self, t: 'pad') -> "bool":
        return _PcbDB.pad___eq__(self, t)

    def __lt__(self, t: 'pad') -> "bool":
        return _PcbDB.pad___lt__(self, t)
    __swig_setmethods__["m_radius"] = _PcbDB.pad_m_radius_set
    __swig_getmethods__["m_radius"] = _PcbDB.pad_m_radius_get
    if _newclass:
        m_radius = _swig_property(_PcbDB.pad_m_radius_get, _PcbDB.pad_m_radius_set)
    __swig_setmethods__["m_clearance"] = _PcbDB.pad_m_clearance_set
    __swig_getmethods__["m_clearance"] = _PcbDB.pad_m_clearance_get
    if _newclass:
        m_clearance = _swig_property(_PcbDB.pad_m_clearance_get, _PcbDB.pad_m_clearance_set)
    __swig_setmethods__["m_pos"] = _PcbDB.pad_m_pos_set
    __swig_getmethods__["m_pos"] = _PcbDB.pad_m_pos_get
    if _newclass:
        m_pos = _swig_property(_PcbDB.pad_m_pos_get, _PcbDB.pad_m_pos_set)
    __swig_setmethods__["m_shape"] = _PcbDB.pad_m_shape_set
    __swig_getmethods__["m_shape"] = _PcbDB.pad_m_shape_get
    if _newclass:
        m_shape = _swig_property(_PcbDB.pad_m_shape_get, _PcbDB.pad_m_shape_set)
    __swig_setmethods__["m_size"] = _PcbDB.pad_m_size_set
    __swig_getmethods__["m_size"] = _PcbDB.pad_m_size_get
    if _newclass:
        m_size = _swig_property(_PcbDB.pad_m_size_get, _PcbDB.pad_m_size_set)

    def __init__(self):
        this = _PcbDB.new_pad()
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_destroy__ = _PcbDB.delete_pad
    __del__ = lambda self: None
pad_swigregister = _PcbDB.pad_swigregister
pad_swigregister(pad)

class padstack(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, padstack, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, padstack, name)
    __repr__ = _swig_repr

    def __init__(self):
        this = _PcbDB.new_padstack()
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this
    __swig_destroy__ = _PcbDB.delete_padstack
    __del__ = lambda self: None

    def getId(self) -> "int":
        return _PcbDB.padstack_getId(self)

    def getName(self) -> "std::string const &":
        return _PcbDB.padstack_getName(self)

    def getType(self) -> "padType":
        return _PcbDB.padstack_getType(self)

    def getPadShape(self) -> "padShape":
        return _PcbDB.padstack_getPadShape(self)

    def getSize(self) -> "point_2d":
        return _PcbDB.padstack_getSize(self)

    def getPos(self) -> "point_2d":
        return _PcbDB.padstack_getPos(self)

    def getAngle(self) -> "double":
        return _PcbDB.padstack_getAngle(self)

    def getRoundRectRatio(self) -> "double":
        return _PcbDB.padstack_getRoundRectRatio(self)

    def getShapeCoords(self) -> "points_2d":
        return _PcbDB.padstack_getShapeCoords(self)

    def getShapePolygon(self) -> "points_2d const &":
        return _PcbDB.padstack_getShapePolygon(self)

    def getLayers(self) -> "std::vector< std::string,std::allocator< std::string > > const &":
        return _PcbDB.padstack_getLayers(self)
padstack_swigregister = _PcbDB.padstack_swigregister
padstack_swigregister(padstack)

class MyVector(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, MyVector, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, MyVector, name)
    __repr__ = _swig_repr

    def iterator(self) -> "swig::SwigPyIterator *":
        return _PcbDB.MyVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self) -> "bool":
        return _PcbDB.MyVector___nonzero__(self)

    def __bool__(self) -> "bool":
        return _PcbDB.MyVector___bool__(self)

    def __len__(self) -> "std::vector< Pin >::size_type":
        return _PcbDB.MyVector___len__(self)

    def __getslice__(self, i: 'std::vector< Pin >::difference_type', j: 'std::vector< Pin >::difference_type') -> "std::vector< Pin,std::allocator< Pin > > *":
        return _PcbDB.MyVector___getslice__(self, i, j)

    def __setslice__(self, *args) -> "void":
        return _PcbDB.MyVector___setslice__(self, *args)

    def __delslice__(self, i: 'std::vector< Pin >::difference_type', j: 'std::vector< Pin >::difference_type') -> "void":
        return _PcbDB.MyVector___delslice__(self, i, j)

    def __delitem__(self, *args) -> "void":
        return _PcbDB.MyVector___delitem__(self, *args)

    def __getitem__(self, *args) -> "std::vector< Pin >::value_type const &":
        return _PcbDB.MyVector___getitem__(self, *args)

    def __setitem__(self, *args) -> "void":
        return _PcbDB.MyVector___setitem__(self, *args)

    def pop(self) -> "std::vector< Pin >::value_type":
        return _PcbDB.MyVector_pop(self)

    def append(self, x: 'Pin') -> "void":
        return _PcbDB.MyVector_append(self, x)

    def empty(self) -> "bool":
        return _PcbDB.MyVector_empty(self)

    def size(self) -> "std::vector< Pin >::size_type":
        return _PcbDB.MyVector_size(self)

    def swap(self, v: 'MyVector') -> "void":
        return _PcbDB.MyVector_swap(self, v)

    def begin(self) -> "std::vector< Pin >::iterator":
        return _PcbDB.MyVector_begin(self)

    def end(self) -> "std::vector< Pin >::iterator":
        return _PcbDB.MyVector_end(self)

    def rbegin(self) -> "std::vector< Pin >::reverse_iterator":
        return _PcbDB.MyVector_rbegin(self)

    def rend(self) -> "std::vector< Pin >::reverse_iterator":
        return _PcbDB.MyVector_rend(self)

    def clear(self) -> "void":
        return _PcbDB.MyVector_clear(self)

    def get_allocator(self) -> "std::vector< Pin >::allocator_type":
        return _PcbDB.MyVector_get_allocator(self)

    def pop_back(self) -> "void":
        return _PcbDB.MyVector_pop_back(self)

    def erase(self, *args) -> "std::vector< Pin >::iterator":
        return _PcbDB.MyVector_erase(self, *args)

    def __init__(self, *args):
        this = _PcbDB.new_MyVector(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x: 'Pin') -> "void":
        return _PcbDB.MyVector_push_back(self, x)

    def front(self) -> "std::vector< Pin >::value_type const &":
        return _PcbDB.MyVector_front(self)

    def back(self) -> "std::vector< Pin >::value_type const &":
        return _PcbDB.MyVector_back(self)

    def assign(self, n: 'std::vector< Pin >::size_type', x: 'Pin') -> "void":
        return _PcbDB.MyVector_assign(self, n, x)

    def resize(self, *args) -> "void":
        return _PcbDB.MyVector_resize(self, *args)

    def insert(self, *args) -> "void":
        return _PcbDB.MyVector_insert(self, *args)

    def reserve(self, n: 'std::vector< Pin >::size_type') -> "void":
        return _PcbDB.MyVector_reserve(self, n)

    def capacity(self) -> "std::vector< Pin >::size_type":
        return _PcbDB.MyVector_capacity(self)
    __swig_destroy__ = _PcbDB.delete_MyVector
    __del__ = lambda self: None
MyVector_swigregister = _PcbDB.MyVector_swigregister
MyVector_swigregister(MyVector)

# This file is compatible with both classic and new-style classes.


