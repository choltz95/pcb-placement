import sys
import os
import re
import math
import time
import numpy as np
import logging
import pdb

datatypes = {
        'float32' : np.float32,
        'float64' : np.float64
        }

class Params (object):
    def __init__(self):
        self.num_bins_x = 0
        self.num_bins_y = 0
        self.scale_factor = 1.0
        self.target_density = 0
        self.routability_opt_flag = False

class PlaceDB (object):
    """
    @brief placement database
    """
    def __init__(self):
        self.num_physical_nodes = 0 # number of real nodes, including movable nodes, terminals, and terminal_NIs
        self.num_terminals = 0 # number of terminals, essentially fixed macros
        self.num_terminal_NIs = 0 # number of terminal_NIs that can be overlapped, essentially IO pins
        self.node_name2id_map = {} # node name to id map, cell name
        self.node_names = [] # 1D array, cell name
        self.node_x = [] # 1D array, cell position x
        self.node_y = [] # 1D array, cell position y
        self.node_orient = [] # 1D array, cell orientation
        self.node_size_x = [] # 1D array, cell width
        self.node_size_y = [] # 1D array, cell height

        self.node2orig_node_map = None # some fixed cells may have non-rectangular shapes; we flatten them and create new nodes
                                        # this map maps the current multiple node ids into the original one

        self.pin_direct = None # 1D array, pin direction IO
        self.pin_offset_x = [] # 1D array, pin offset x to its node
        self.pin_offset_y = [] # 1D array, pin offset y to its node

        self.net_name2id_map = {} # net name to id map
        self.net_names = [] # net name
        self.net_weights = [] # weights for each net

        self.net2pin_map = [] # array of 1D array, each row stores pin id
        self.flat_net2pin_map = [] # flatten version of net2pin_map
        self.flat_net2pin_start_map = [] # starting index of each net in flat_net2pin_map

        self.node2pin_map = [] # array of 1D array, contains pin id of each node
        self.flat_node2pin_map = [] # flatten version of node2pin_map
        self.flat_node2pin_start_map = [] # starting index of each node in flat_node2pin_map

        self.pin2node_map = [] # 1D array, contain parent node id of each pin
        self.pin2net_map = [] # 1D array, contain parent net id of each pin

        self.rows = None # NumRows x 4 array, stores xl, yl, xh, yh of each row
        self.regions = []

        self.xl = 0.0
        self.yl = 0.0
        self.xh = 0.0
        self.yh = 0.0

        self.row_height = 1.0

        self.site_width = 1.0
        self.bin_size_x = None
        self.bin_size_y = None
        self.num_bins_x = None
        self.num_bins_y = None
        self.bin_center_x = None
        self.bin_center_y = None

        self.num_movable_pins = 0

        self.total_movable_node_area = 0.0 # total movable cell area
        self.total_fixed_node_area = 0.0 # total fixed cell area
        self.total_space_area = 0.0 # total placeable space area excluding fixed cells
        self.total_filler_node_area = 0.0
        self.num_filler_nodes = 0

        self.dtype = None

    def scale_pl(self, scale_factor):
        """
        @brief scale placement solution only
        @param scale_factor scale factor
        """
        self.node_x *= scale_factor
        self.node_y *= scale_factor

    def scale(self, scale_factor):
        """
        @brief scale distances
        @param scale_factor scale factor
        """
        logging.info("scale coordinate system by %g" % (scale_factor))
        self.scale_pl(scale_factor)
        self.node_size_x *= scale_factor
        self.node_size_y *= scale_factor
        self.pin_offset_x *= scale_factor
        self.pin_offset_y *= scale_factor
        self.xl *= scale_factor
        self.yl *= scale_factor
        self.xh *= scale_factor
        self.yh *= scale_factor
        self.row_height *= scale_factor
        self.site_width *= scale_factor
        if self.rows: self.rows *= scale_factor
        self.total_space_area *= scale_factor * scale_factor # this is area
        # may have performance issue
        # I assume there are not many boxes
        for i in range(len(self.regions)):
            self.regions[i] *= scale_factor

    @property
    def num_movable_nodes(self):
        """
        @return number of movable nodes
        """
        return self.num_physical_nodes - self.num_terminals - self.num_terminal_NIs

    @property
    def num_nodes(self):
        """
        @return number of movable nodes, terminals, terminal_NIs, and fillers
        """
        return self.num_physical_nodes + self.num_filler_nodes

    @property
    def num_nets(self):
        """
        @return number of nets
        """
        return len(self.net2pin_map)

    @property
    def num_pins(self):
        """
        @return number of pins
        """
        return len(self.pin2net_map)

    @property
    def width(self):
        """
        @return width of layout
        """
        return self.xh-self.xl

    @property
    def height(self):
        """
        @return height of layout
        """
        return self.yh-self.yl

    @property
    def area(self):
        """
        @return area of layout
        """
        return self.width*self.height

    def bin_index_x(self, x):
        """
        @param x horizontal location
        @return bin index in x direction
        """
        if x < self.xl:
            return 0
        elif x > self.xh:
            return int(np.floor((self.xh-self.xl)/self.bin_size_x))
        else:
            return int(np.floor((x-self.xl)/self.bin_size_x))

    def bin_index_y(self, y):
        """
        @param y vertical location
        @return bin index in y direction
        """
        if y < self.yl:
            return 0
        elif y > self.yh:
            return int(np.floor((self.yh-self.yl)/self.bin_size_y))
        else:
            return int(np.floor((y-self.yl)/self.bin_size_y))

    def bin_xl(self, id_x):
        """
        @param id_x horizontal index
        @return bin xl
        """
        return self.xl+id_x*self.bin_size_x

    def bin_xh(self, id_x):
        """
        @param id_x horizontal index
        @return bin xh
        """
        return min(self.bin_xl(id_x)+self.bin_size_x, self.xh)

    def bin_yl(self, id_y):
        """
        @param id_y vertical index
        @return bin yl
        """
        return self.yl+id_y*self.bin_size_y

    def bin_yh(self, id_y):
        """
        @param id_y vertical index
        @return bin yh
        """
        return min(self.bin_yl(id_y)+self.bin_size_y, self.yh)

    def num_bins(self, l, h, bin_size):
        """
        @brief compute number of bins
        @param l lower bound
        @param h upper bound
        @param bin_size bin size
        @return number of bins
        """
        return int(np.ceil((h-l)/bin_size))

    def bin_centers(self, l, h, bin_size):
        """
        @brief compute bin centers
        @param l lower bound
        @param h upper bound
        @param bin_size bin size
        @return array of bin centers
        """
        num_bins = self.num_bins(l, h, bin_size)
        centers = np.zeros(num_bins, dtype=self.dtype)
        for id_x in range(num_bins):
            bin_l = l+id_x*bin_size
            bin_h = min(bin_l+bin_size, h)
            centers[id_x] = (bin_l+bin_h)/2
        return centers

    def net_hpwl(self, x, y, net_id):
        """
        @brief compute HPWL of a net
        @param x horizontal cell locations
        @param y vertical cell locations
        @return hpwl of a net
        """
        pins = self.net2pin_map[net_id]
        nodes = self.pin2node_map[pins]
        hpwl_x = np.amax(x[nodes]+self.pin_offset_x[pins]) - np.amin(x[nodes]+self.pin_offset_x[pins])
        hpwl_y = np.amax(y[nodes]+self.pin_offset_y[pins]) - np.amin(y[nodes]+self.pin_offset_y[pins])

        return (hpwl_x+hpwl_y)*self.net_weights[net_id]

    def hpwl(self, x, y):
        """
        @brief compute total HPWL
        @param x horizontal cell locations
        @param y vertical cell locations
        @return hpwl of all nets
        """
        wl = 0
        for net_id in range(len(self.net2pin_map)):
            wl += self.net_hpwl(x, y, net_id)
        return wl

    def overlap(self, xl1, yl1, xh1, yh1, xl2, yl2, xh2, yh2):
        """
        @brief compute overlap between two boxes
        @return overlap area between two rectangles
        """
        return max(min(xh1, xh2)-max(xl1, xl2), 0.0) * max(min(yh1, yh2)-max(yl1, yl2), 0.0)

    def density_map(self, x, y):
        """
        @brief this density map evaluates the overlap between cell and bins
        @param x horizontal cell locations
        @param y vertical cell locations
        @return density map
        """
        bin_index_xl = np.maximum(np.floor(x/self.bin_size_x).astype(np.int32), 0)
        bin_index_xh = np.minimum(np.ceil((x+self.node_size_x)/self.bin_size_x).astype(np.int32), self.num_bins_x-1)
        bin_index_yl = np.maximum(np.floor(y/self.bin_size_y).astype(np.int32), 0)
        bin_index_yh = np.minimum(np.ceil((y+self.node_size_y)/self.bin_size_y).astype(np.int32), self.num_bins_y-1)

        density_map = np.zeros([self.num_bins_x, self.num_bins_y])

        for node_id in range(self.num_physical_nodes):
            for ix in range(bin_index_xl[node_id], bin_index_xh[node_id]+1):
                for iy in range(bin_index_yl[node_id], bin_index_yh[node_id]+1):
                    density_map[ix, iy] += self.overlap(
                            self.bin_xl(ix), self.bin_yl(iy), self.bin_xh(ix), self.bin_yh(iy),
                            x[node_id], y[node_id], x[node_id]+self.node_size_x[node_id], y[node_id]+self.node_size_y[node_id]
                            )

        for ix in range(self.num_bins_x):
            for iy in range(self.num_bins_y):
                density_map[ix, iy] /= (self.bin_xh(ix)-self.bin_xl(ix))*(self.bin_yh(iy)-self.bin_yl(iy))

        return density_map

    def density_overflow(self, x, y, target_density):
        """
        @brief if density of a bin is larger than target_density, consider as overflow bin
        @param x horizontal cell locations
        @param y vertical cell locations
        @param target_density target density
        @return density overflow cost
        """
        density_map = self.density_map(x, y)
        return np.sum(np.square(np.maximum(density_map-target_density, 0.0)))

    def print_node(self, node_id):
        """
        @brief print node information
        @param node_id cell index
        """
        logging.debug("node %s(%d), size (%g, %g), pos (%g, %g)" % (self.node_names[node_id], node_id, self.node_size_x[node_id],
                                                                    self.node_size_y[node_id], self.node_x[node_id], self.node_y[node_id]))
        pins = "pins "
        for pin_id in self.node2pin_map[node_id]:
            pins += "%s(%s, %d) " % (self.node_names[self.pin2node_map[pin_id]], self.net_names[self.pin2net_map[pin_id]], pin_id)
        logging.debug(pins)

    def print_net(self, net_id):
        """
        @brief print net information
        @param net_id net index
        """
        logging.debug("net %s(%d)" % (self.net_names[net_id], net_id))
        pins = "pins "
        for pin_id in self.net2pin_map[net_id]:
            pins += "%s(%s, %d) " % (self.node_names[self.pin2node_map[pin_id]], self.net_names[self.pin2net_map[pin_id]], pin_id)
        logging.debug(pins)

    def print_row(self, row_id):
        """
        @brief print row information
        @param row_id row index
        """
        logging.debug("row %d %s" % (row_id, self.rows[row_id]))


    def initialize(self, params):
        """
        @brief initialize data members after reading
        @param params parameters
        """

        # initialize class variables as numpy arrays
        self.node_x = np.array(self.node_x)
        self.node_y = np.array(self.node_y)
        self.node_size_x = np.array(self.node_size_x)
        self.node_size_y = np.array(self.node_size_y)
        self.pin_offset_x = np.array(self.pin_offset_x)
        self.pin_offset_y = np.array(self.pin_offset_y)
        self.net_weights = np.array(self.net_weights)

        self.total_movable_node_area = self.area
        self.total_space_area = self.area

        # scale
        # adjust scale_factor if not set
        if params.scale_factor == 0.0 or self.site_width != 1.0:
            params.scale_factor = 1.0 / self.site_width
            logging.info("set scale_factor = %g, as site_width = %g" % (params.scale_factor, self.site_width))
        self.scale(params.scale_factor)

        content = """
================================= Benchmark Statistics =================================
#nodes = %d, #terminals = %d, # terminal_NIs = %d, #movable = %d, #nets = %d
die area = (%g, %g, %g, %g) %g
row height = %g, site width = %g
""" % (
                self.num_physical_nodes, self.num_terminals, self.num_terminal_NIs, self.num_movable_nodes, len(self.net_names),
                self.xl, self.yl, self.xh, self.yh, self.area,
                self.row_height, self.site_width
                )

        # set number of bins
        # derive bin dimensions by keeping the aspect ratio
        aspect_ratio = (self.yh - self.yl) / (self.xh - self.xl)
        num_bins_x = int(math.pow(2, max(np.ceil(math.log2(math.sqrt(self.num_movable_nodes / aspect_ratio))), 0)))
        num_bins_y = int(math.pow(2, max(np.ceil(math.log2(math.sqrt(self.num_movable_nodes * aspect_ratio))), 0)))
        self.num_bins_x = max(params.num_bins_x, num_bins_x)
        self.num_bins_y = max(params.num_bins_y, num_bins_y)
        # set bin size
        self.bin_size_x = (self.xh-self.xl)/self.num_bins_x
        self.bin_size_y = (self.yh-self.yl)/self.num_bins_y

        # bin center array
        self.bin_center_x = self.bin_centers(self.xl, self.xh, self.bin_size_x)
        self.bin_center_y = self.bin_centers(self.yl, self.yh, self.bin_size_y)

        content += "num_bins = %dx%d, bin sizes = %gx%g\n" % (self.num_bins_x, self.num_bins_y, self.bin_size_x/self.row_height, self.bin_size_y/self.row_height)

        # set num_movable_pins
        if self.num_movable_pins is None:
            self.num_movable_pins = 0
            for node_id in self.pin2node_map:
                if node_id < self.num_movable_nodes:
                    self.num_movable_pins += 1
        content += "#pins = %d, #movable_pins = %d\n" % (self.num_pins, self.num_movable_pins)
        # set total cell area
        self.total_movable_node_area = float(np.sum(self.node_size_x[:self.num_movable_nodes]*self.node_size_y[:self.num_movable_nodes]))
        # total fixed node area should exclude the area outside the layout and the area of terminal_NIs
        self.total_fixed_node_area = float(np.sum(
                np.maximum(
                    np.minimum(self.node_x[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs] +
                               self.node_size_x[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs], self.xh)
                    - np.maximum(self.node_x[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs], self.xl),
                    0.0) * np.maximum(
                        np.minimum(self.node_y[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs] +
                                   self.node_size_y[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs], self.yh)
                        - np.maximum(self.node_y[self.num_movable_nodes:self.num_physical_nodes - self.num_terminal_NIs], self.yl),
                        0.0)
                ))
        content += "total_movable_node_area = %g, total_fixed_node_area = %g, total_space_area = %g\n" % (self.total_movable_node_area, self.total_fixed_node_area, self.total_space_area)

        target_density = min(self.total_movable_node_area / self.total_space_area, 1.0)
        if target_density > params.target_density:
            logging.warn("target_density %g is smaller than utilization %g, ignored" % (params.target_density, target_density))
            params.target_density = target_density
        content += "utilization = %g, target_density = %g\n" % (self.total_movable_node_area / self.total_space_area, params.target_density)

        #content += "total_filler_node_area = %g, #fillers = %d, filler sizes = %gx%g\n" % (self.total_filler_node_area, self.num_filler_nodes, filler_size_x, filler_size_y)
        #if params.routability_opt_flag:
        #    content += "================================== routing information =================================\n"
        #    content += "routing grids (%d, %d)\n" % (self.num_routing_grids_x, self.num_routing_grids_y)
        #    content += "routing grid sizes (%g, %g)\n" % (self.routing_grid_size_x, self.routing_grid_size_y)
        #    content += "routing capacity H/V (%g, %g) per tile\n" % (self.unit_horizontal_capacity * self.routing_grid_size_y, self.unit_vertical_capacity * self.routing_grid_size_x)
        content += "========================================================================================"

        logging.info(content)

    def read_pl(self, params, pl_file):
        """
        @brief read .pl file
        @param pl_file .pl file
        """
        tt = time.time()
        logging.info("reading %s" % (pl_file))
        count = 0
        with open(pl_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("UCLA") or line.startswith("#") or "POW" in line or "GND" in line:
                    continue
                # node positions
                pos = re.search(r"(\w+)\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s+([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s*:?\s*(\w+)?", line)
                if pos:
                    node_id = self.node_name2id_map[pos.group(1)]
                    self.node_x[node_id] = float(pos.group(2))
                    self.node_y[node_id] = float(pos.group(6))
                    if pos.group(10) is not None:
                        self.node_orient[node_id] = int(pos.group(10))
                    else:
                        self.node_orient[node_id] = 0
                    orient = pos.group(4)
        if params.scale_factor != 1.0:
            self.scale_pl(params.scale_factor)
        logging.info("read_pl takes %.3f seconds" % (time.time()-tt))

    def read_nodes(self, params, nodes_file):
        pass

    def read_nets(self, params, nets_file):
        pass

    def read_block(self, params, blocks_file):
        blocks = {}
        with open(blocks_file, 'r') as f:
            lines = f.read().splitlines()
        components = {}
        node_i = 0
        for line in lines:
            line = line.strip()
            if line.startswith("UCLA") or line.startswith("#"):
                continue
            l = line.split()
            if line.strip() == "": continue
            if "Outline" in line:
                self.xl = 0
                self.yl = 0
                self.xh = int(l[-2])
                self.yh = int(l[-1])
            elif "NumBlocks" in line:
                self.num_physical_nodes += int(l[-1])
            elif "NumTerminals" in line:
                self.num_terminals = int(l[-1])
                self.num_physical_nodes += int(l[-1])
            else:
                if node_i < self.num_physical_nodes:
                    name = l[0]
                    w = float(l[-2])
                    h = float(l[-1])
                    self.node_names.append(name)
                    self.node_name2id_map[name] = node_i
                    self.node_size_x.append(w)
                    self.node_size_y.append(h)
                    self.node_x.append(0)
                    self.node_y.append(0)
                    self.node_orient.append(0)
                    self.node2pin_map.append([])
                    node_i+=1
                else:
                    name = l[0]
                    x = float(l[-2])
                    y = float(l[-1])
                    self.node_names.append(name)
                    self.node_name2id_map[name] = node_i
                    self.node_size_x.append(1)
                    self.node_size_y.append(1)
                    self.node_x.append(x)
                    self.node_y.append(y)
                    self.node_orient.append(0)
                    self.node2pin_map.append([])
                    node_i+=1

    def read_nets_mcnc(self, params, nets_file):
        tt = time.time()
        logging.info("reading %s" % (nets_file))
        count = 0
        net_name = 0
        pin_id = 0
        with open(nets_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("UCLA") or line.startswith("#"):
                    continue
                if line.startswith("NetDegree"):
                    self.net_names.append(str(net_name))
                    self.net_name2id_map[str(net_name)] = net_name
                    self.net_weights.append(1.0)
                    self.net2pin_map.append([])
                    net_name += 1
                    continue
                #pos = re.search(r"(\w+)\s+(\w+)\s+:\s*\%([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s+\%([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)\s*",line)
                pos = re.search(r"(\w+)\s+(\w+)\s*(:?\s*\%?)([+-]?(\d+(\.\d*)?|\.\d+)?([eE][+-]?\d+)?)\s*(\%([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?))?\s*",line)
                if pos:
                    node_id = self.node_name2id_map[pos.group(1)]
                    if pos.group(9):
                        pin_offset_x = float(pos.group(4))#3
                        pin_offset_y = float(pos.group(9))#7
                    else:
                        pin_offset_x=0.0
                        pin_offset_y=0.0
                    pin_exists = False
                    _pin_id = -1
                    for _pid in self.node2pin_map[node_id]:
                        if self.pin_offset_x[_pid] == pin_offset_x and self.pin_offset_y[_pid] == pin_offset_y:
                            pin_exists = True
                            _pin_id = _pid
                    if pin_exists:
                        assert _pin_id > -1
                        self.net2pin_map[net_name-1].append(_pin_id)
                        self.pin2net_map[_pin_id].append(net_name-1)
                    else:
                        self.net2pin_map[net_name-1].append(pin_id)
                        self.node2pin_map[node_id].append(pin_id)
                        self.pin_offset_x.append(pin_offset_x)
                        self.pin_offset_y.append(pin_offset_y)
                        self.pin2node_map.append(node_id)
                        self.pin2net_map.append([net_name-1])
                        #self.pin2net_map.append(net_name-1)
                        pin_id += 1

        logging.info("read_nets takes %.3f seconds" % (time.time()-tt))

    def write_pl(self, params, pl_file, node_x, node_y):
        """
        @brief write .pl file
        @param pl_file .pl file
        """
        tt = time.time()
        logging.info("writing to %s" % (pl_file))
        content = "UCLA pl 1.0\n"
        str_node_names = np.array(self.node_names).astype(np.str)
        str_node_orient = np.array(self.node_orient).astype(np.str)
        for i in range(self.num_movable_nodes):
            content += "\n%s %g %g : %s" % (
                    str_node_names[i],
                    node_x[i],
                    node_y[i],
                    str_node_orient[i]
                    )
        # use the original fixed cells, because they are expanded if they contain shapes
        fixed_node_indices = list(self.rawdb.fixedNodeIndices())
        for i, node_id in enumerate(fixed_node_indices):
            content += "\n%s %g %g : %s /FIXED" % (
                    str(self.rawdb.nodeName(node_id)),
                    float(self.rawdb.node(node_id).xl()),
                    float(self.rawdb.node(node_id).yl()),
                    "N" # still hard-coded
                    )
        for i in range(self.num_movable_nodes + self.num_terminals, self.num_movable_nodes + self.num_terminals + self.num_terminal_NIs):
            content += "\n%s %g %g : %s /FIXED_NI" % (
                    str_node_names[i],
                    node_x[i],
                    node_y[i],
                    str_node_orient[i]
                    )
        with open(pl_file, "w") as f:
            f.write(content)
        logging.info("write_pl takes %.3f seconds" % (time.time()-tt))

    def write_nets(self, params, net_file):
        """
        @brief write .net file
        @param params parameters
        @param net_file .net file
        """
        tt = time.time()
        logging.info("writing to %s" % (net_file))
        content = "UCLA nets 1.0\n"
        content += "\nNumNets : %d" % (len(self.net2pin_map))
        content += "\nNumPins : %d" % (len(self.pin2net_map))
        content += "\n"

        for net_id in range(len(self.net2pin_map)):
            pins = self.net2pin_map[net_id]
            content += "\nNetDegree : %d %s" % (len(pins), self.net_names[net_id])
            for pin_id in pins:
                content += "\n\t%s %s : %d %d" % (self.node_names[self.pin2node_map[pin_id]], self.pin_direct[pin_id],
                                                  self.pin_offset_x[pin_id]/params.scale_factor, self.pin_offset_y[pin_id]/params.scale_factor)

        with open(net_file, "w") as f:
            f.write(content)
        logging.info("write_nets takes %.3f seconds" % (time.time()-tt))

    def write_pl(self, params, pl_file, node_x, node_y):
        """
        @brief write .pl file
        @param pl_file .pl file
        """
        tt = time.time()
        logging.info("writing to %s" % (pl_file))
        content = "UCLA pl 1.0\n"
        str_node_names = np.array(self.node_names).astype(np.str)
        str_node_orient = np.array(self.node_orient).astype(np.str)
        for i in range(self.num_movable_nodes):
            content += "\n%s %g %g : %s" % (
                    str_node_names[i],
                    node_x[i],
                    node_y[i],
                    str_node_orient[i]
                    )
        # use the original fixed cells, because they are expanded if they contain shapes
        fixed_node_indices = list(self.rawdb.fixedNodeIndices())
        for i, node_id in enumerate(fixed_node_indices):
            content += "\n%s %g %g : %s /FIXED" % (
                    str(self.rawdb.nodeName(node_id)),
                    float(self.rawdb.node(node_id).xl()),
                    float(self.rawdb.node(node_id).yl()),
                    "N" # still hard-coded
                    )
        for i in range(self.num_movable_nodes + self.num_terminals, self.num_movable_nodes + self.num_terminals + self.num_terminal_NIs):
            content += "\n%s %g %g : %s /FIXED_NI" % (
                    str_node_names[i],
                    node_x[i],
                    node_y[i],
                    str_node_orient[i]
                    )
        with open(pl_file, "w") as f:
            f.write(content)
        logging.info("write_pl takes %.3f seconds" % (time.time()-tt))

    def write_nets(self, params, net_file):
        """
        @brief write .net file
        @param params parameters
        @param net_file .net file
        """
        tt = time.time()
        logging.info("writing to %s" % (net_file))
        content = "UCLA nets 1.0\n"
        content += "\nNumNets : %d" % (len(self.net2pin_map))
        content += "\nNumPins : %d" % (len(self.pin2net_map))
        content += "\n"

        for net_id in range(len(self.net2pin_map)):
            pins = self.net2pin_map[net_id]
            content += "\nNetDegree : %d %s" % (len(pins), self.net_names[net_id])
            for pin_id in pins:
                content += "\n\t%s %s : %d %d" % (self.node_names[self.pin2node_map[pin_id]], self.pin_direct[pin_id],
                                                  self.pin_offset_x[pin_id]/params.scale_factor, self.pin_offset_y[pin_id]/params.scale_factor)

        with open(net_file, "w") as f:
            f.write(content)
        logging.info("write_nets takes %.3f seconds" % (time.time()-tt))
