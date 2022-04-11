import cvxpy as cp
from cvxpy import Variable, Constant, Minimize, Problem
from z3 import *
import numpy as np
import placedb
import pylab
import math
import joblib

class Net(object):
    """ A box in a floor packing problem. """
    ASPECT_RATIO = 1.0
    def __init__(self, moduleidxs, offsets, idx):
        self.moduleidxs = moduleidxs
        self.offsets = offsets
        self.idx = idx
        self.U_x = Variable()
        self.L_x = Variable()
        self.U_y = Variable()
        self.L_y = Variable()

class Box(object):
    """ A box in a floor packing problem. """
    ASPECT_RATIO = 1.0
    def __init__(self, width, height, initialx=0.0, initialy=0.0, initialr=0, initialmx=0, initialmy=0, 
                 idx=0, pl=True, r=False, m=False, terminal=False, min_area=None):
        self.min_area = min_area
        self.h = Constant(width)
        self.w = Constant(height)
        self.terminal = terminal
        self.pl = pl
        self.r = r # rotated
        self.mx = m # mirrored
        self.my = m # mirrored
        
        #self.initial_x = Constant(initialx)
        #self.initial_y = Constant(initialy)
        #self.slack = Variable(nonneg=True)
        if pl:
            self.x = Variable(nonneg=True)
            self.y = Variable(nonneg=True)
            self.x.value = initialx
            self.y.value = initialy
        else:
            self.x = Constant(initialx)
            self.y = Constant(initialy)        
        if r:
            self.r = Variable(boolean=True)
            self.r.value = initialr
        else:
            self.r=Constant(initialr)
        if m:
            self.mx = Variable(boolean=True)
            self.mx.value = initialmx
            
            self.my = Variable(boolean=True)
            self.my.value = initialmy
        else:
            self.mx=Constant(initialmx) 
            self.my=Constant(initialmy)
            
        self.xorrmx = Variable(boolean=True) # xor wigit        
        self.rmx    = Variable(boolean=True) # rotated and mirrrored 
        self.nrnmx  = Variable(boolean=True) # not rotated and not mirrored
        self.nrmx   = Variable(boolean=True) # not rotated and mirrored
        self.rnmx   = Variable(boolean=True) # rotated and not mirrored
        
        self.xorrmy = Variable(boolean=True) # xor wigit
        self.rmy    = Variable(boolean=True) # rotated and mirrrored 
        self.nrnmy  = Variable(boolean=True) # not rotated and not mirrored
        self.nrmy   = Variable(boolean=True) # not rotated and mirrored
        self.rnmy   = Variable(boolean=True) # rotated and not mirrored
        
        self.idx = idx
        self.netidxs = []

    @property
    def position(self):
        return (np.round(self.x.value,2), np.round(self.y.value,2))

    @property
    def size(self):
        return (np.round(self.w.value,2), np.round(self.h.value,2))
    
    @property
    def center(self):
        return self.x + self.r*self.h/2 + (1-self.r)*self.w/2, self.y + self.r*self.w/2 + (1-self.r)*self.h/2

    @property
    def left(self):
        return self.x 

    @property
    def right(self):
        return self.x + (1-self.r)*self.w + (self.r)*self.h

    @property
    def bottom(self):
        return self.y

    @property
    def top(self):
        return self.y + (self.r)*self.w + (1-self.r)*self.h

    @property
    def rotation(self):
        return self.r.value
    
    @property
    def mirror(self):
        return self.m.value

"""
http://cc.ee.ntu.edu.tw/~ywchang/Courses/PD_Source/EDA_floorplanning.pdf
"""
class FloorPlan(object):
    MARGIN = 0.1
    ASPECT_RATIO = 5.0
    def __init__(self, boxes, nets, adj, obj=True, norelpairs=None, horizontal_orderings=[], vertical_orderings=[], boundary_W=100, ox=0,oy=0,boundary_H=100, margin=0.5, max_seconds=100, num_cores=1, name=""):
        self.boxes = boxes
        self.nets = nets
        self._adj = adj
        self._obj = obj
        self.norelpairs = norelpairs
        self.adj = Constant(adj)
        self.num_nets = len(nets)
        self.num_nodes = len(boxes)
        self.name=name
        
        self.boundary_W = Constant(boundary_W)
        self.boundary_H = Constant(boundary_H)
        self.ox = Constant(ox)
        self.oy = Constant(oy)
        self.margin=margin
        
        #self.height = Variable(pos=True)
        #self.width = Variable(pos=True)
        
        self.max_x = Variable()
        self.max_x.value = self.boundary_W.value
        self.max_y = Variable()
        self.max_y.value = self.boundary_H.value
        self.min_x = Variable()
        self.min_x.value = 0.0
        self.min_y = Variable()
        self.min_y.value = 0.0
        
        self.height = Constant(boundary_H)
        self.width = Constant(boundary_W)
        
        self.p = cp.Variable(shape=(self.num_nodes,self.num_nodes), boolean=True)
        self.q = cp.Variable(shape=(self.num_nodes,self.num_nodes), boolean=True)
        
        nec = np.where(self._adj > 1)[0].shape[0]
        self.u_p = cp.Variable(shape=(nec,nec), boolean=True)
        self.u_q = cp.Variable(shape=(nec,nec), boolean=True)
        self.u_g = cp.Variable(shape=(nec,nec))
                
        self.horizontal_orderings = horizontal_orderings
        self.vertical_orderings = vertical_orderings
        
        self.max_seconds = max(max_seconds,1)
        self.num_cores = max(num_cores,1)

    @property
    def size(self):
        return (np.round(self.width.value,2), np.round(self.height.value,2))

    # Return constraints for the ordering.
    @staticmethod
    def _order(boxes, horizontal):
        if len(boxes) == 0: return
        constraints = []
        curr = boxes[0]
        for box in boxes[1:]:
            if horizontal:
                constraints.append(curr.right + FloorPlan.MARGIN <= box.left)
            else:
                constraints.append(curr.top + FloorPlan.MARGIN <= box.bottom)
            curr = box
        return constraints

    def layout(self, solve=True):
        constraints = []
        for box in self.boxes:
            
            #slack
            #constraints += [box.slack >= cp.abs(box.x-box.initial_x)+cp.abs(box.y-box.initial_y)]
            
            # xor wigit
            constraints += [box.xorrmx <= box.r + box.mx, 
                            box.xorrmx >= box.r - box.mx,
                            box.xorrmx >= box.mx - box.r,
                            box.xorrmx <= 2 - box.r - box.mx,
                            
                            box.xorrmy <= box.r + box.my, 
                            box.xorrmy >= box.r - box.my,
                            box.xorrmy >= box.my - box.r,
                            box.xorrmy <= 2 - box.r - box.my
                           ]
            
            # rotation & mirroring
            constraints += [box.rmx >= box.r + box.mx - 1,
                            box.rmx <= box.r,
                            box.rmx <= box.mx,
                            
                            box.rmy >= box.r + box.my - 1,
                            box.rmy <= box.r,
                            box.rmy <= box.my
                           ]
            # rotation & not mirroring == (rot xor mir) and rot
            constraints += [box.rnmx >= box.r + box.xorrmx - 1,
                            box.rnmx <= box.r,
                            box.rnmx <= box.xorrmx,
                            
                            box.rnmy >= box.r + box.xorrmy - 1,
                            box.rnmy <= box.r,
                            box.rnmy <= box.xorrmy
                           ]
            # not rotation & mirroring == (rot xor mir) and mir
            constraints += [box.nrmx >= box.xorrmx + box.mx - 1,
                            box.nrmx <= box.xorrmx,
                            box.nrmx <= box.mx,
                            
                            box.nrmy >= box.xorrmy + box.my - 1,
                            box.nrmy <= box.xorrmy,
                            box.nrmy <= box.my
                           ]

            # not rotation & not mirroring
            constraints += [box.nrnmx >= (1-box.r) + (1-box.mx) - 1,
                            box.nrnmx <= (1-box.r),
                            box.nrnmx <= (1-box.mx),
                            
                            box.nrnmy >= (1-box.r) + (1-box.my) - 1,
                            box.nrnmy <= (1-box.r),
                            box.nrnmy <= (1-box.my)
                           ]
            
            if (not box.pl):
                continue
            # Enforce that boxes lie in bounding box. 
            constraints += [self.min_x <= b.x for b in self.boxes if b.pl]
            constraints += [self.min_y <= b.y for b in self.boxes if b.pl]
            constraints += [self.max_x >= b.x+ b.r*b.h + (1-b.r)*b.w for b in self.boxes if b.pl]
            constraints += [self.max_y >= b.y+ b.r*b.w + (1-b.r)*b.h for b in self.boxes if b.pl]
            
            constraints += [self.min_x >= self.ox,
                            self.min_y >= self.oy]
            
            constraints += [self.max_x <= self.ox + self.boundary_W,
                            self.max_y <= self.oy + self.boundary_H]
            
            #constraints += [box.x >= FloorPlan.MARGIN,
            #    box.x + box.r*box.h + (1-box.r)*box.w + FloorPlan.MARGIN <= self.width]
            #constraints += [box.y >= FloorPlan.MARGIN,
            #                box.y + box.r*box.w + (1-box.r)*box.h + FloorPlan.MARGIN <= self.height]
            
            # Enforce aspect ratios.
            #constraints += [(1/box.ASPECT_RATIO)*box.height <= box.width,
            #                box.width <= box.ASPECT_RATIO*box.height]
            # Enforce minimum area
            #constraints += [
            #    geo_mean(vstack([box.width, box.height])) >= math.sqrt(box.min_area)
            #]
            
        # wirelength minimization
        for n_i, net in enumerate(self.nets):
            if len(net.moduleidxs) <= 1: continue
            # no explicit net crossing constraints
            if False:
                for n_j in range(n_i+1,len(self.nets)):
                    net_i = net
                    net_j = self.nets[n_j]
                    constraints += [
                        net_i.L_x + (net_i.U_x-net_i.L_x) <= net_j.L_x + self.boundary_W*(self.u_p[n_i,n_j] + self.u_q[n_i,n_j]) - self.margin + self.u_g[n_i, n_j],
                        net_i.L_y + (net_i.U_y-net_i.L_y) <= net_j.L_x + self.boundary_H*(1 + self.u_p[n_i,n_j] - self.u_q[n_i,n_j]) - self.margin  + self.u_g[n_i, n_j],
                        net_i.L_x - (net_j.U_x-net_j.L_x) >= net_j.L_x - self.boundary_W*(1 - self.u_p[n_i,n_j] + self.u_q[n_i,n_j]) + self.margin  - self.u_g[n_i, n_j],
                        net_i.L_y - (net_j.U_y-net_j.L_y) >= net_j.L_x - self.boundary_H*(2 - self.u_p[n_i,n_j] - self.u_q[n_i,n_j]) + self.margin - self.u_g[n_i, n_j],
                    ]
            mx = []
            my = []
            for pi, moduleidx in enumerate(net.moduleidxs):
                box = self.boxes[moduleidx]
                boxx, boxy = box.center
                boxw = box.w
                boxh = box.h
                
                poff = net.offsets[pi]
                
                # not rotated, not mirrored
                # rotated, mirrored
                # not rotated, mirrored
                # rotated, not mirrored
                
                pinposx = boxx + box.nrnmx*(poff[0]*boxw) + \
                                 box.rmx*(-1*poff[1]*boxh) + \
                                 box.nrmx*(-1*poff[0]*boxw) + \
                                 box.rnmx*(poff[1]*boxh)
                
                pinposy = boxy + box.nrnmy*(poff[1]*boxh) + \
                                 box.rmy*(-1*poff[0]*boxw) + \
                                 box.nrmy*(-1*poff[1]*boxh) + \
                                 box.rnmy*(poff[0]*boxw)
                
                mx.append(pinposx)
                my.append(pinposy)
                                
            #mx, my = list(zip(*modules))
            constraints += [net.L_x <= x for x in mx]
            constraints += [net.U_x >= x for x in mx]
            constraints += [net.L_y <= y for y in my]
            constraints += [net.U_y >= y for y in my]
        
        # nonoverlap constraints
        for i in range(len(self.boxes)):
            for j in range(i+1,len(self.boxes)):
                if self.norelpairs is not None:
                    if [i,j] not in self.norelpairs:
                        continue
                b_i = self.boxes[i]
                b_j = self.boxes[j]
                
                if (not b_i.pl) and (not b_j.pl):
                    continue
                
                x_i, y_i = b_i.x, b_i.y
                w_i, h_i = b_i.w, b_i.h
                r_i  = b_i.r
                
                x_j, y_j = b_j.x, b_j.y
                w_j, h_j = b_j.w, b_j.h
                r_j = b_j.r
                                
                constraints += [
                    x_i + r_i*h_i + (1-r_i)*w_i <= x_j + self.boundary_W*(self.p[i,j] + self.q[i,j]) - self.margin,
                    y_i + r_i*w_i + (1-r_i)*h_i <= y_j + self.boundary_H*(1 + self.p[i,j] - self.q[i,j]) - self.margin,
                    x_i - r_j*h_j - (1-r_j)*w_j >= x_j - self.boundary_W*(1 - self.p[i,j] + self.q[i,j]) + self.margin,
                    y_i - r_j*w_j - (1-r_j)*h_j >= y_j - self.boundary_H*(2 - self.p[i,j] - self.q[i,j]) + self.margin,
                ]
                
                # net crossing constraints
                """
                ci_x, ci_y = b_i.center
                cj_x, cj_y = b_j.center
                
                if self._adj[i,j] >= 1:
                    w_i = cp.abs(cj_x - ci_x)
                    h_i = cp.abs(cj_y - ci_y)
                    constraints += [
                        ci_x + w_i/2 <= cj_x + self.boundary_W*(self.u_p[i,j] + self.u_q[i,j]) - FloorPlan.MARGIN + self.u_g[i, j],
                        ci_y + h_i/2 <= cj_y + self.boundary_H*(1 + self.u_p[i,j] - self.u_q[i,j]) - FloorPlan.MARGIN + self.u_g[i, j],
                        ci_x - w_j/2 >= cj_x - self.boundary_W*(1 - self.u_p[i,j] + self.u_q[i,j]) + FloorPlan.MARGIN - self.u_g[i, j],
                        ci_y - h_j/2 >= cj_y - self.boundary_H*(2 - self.u_p[i,j] - self.u_q[i,j]) + FloorPlan.MARGIN - self.u_g[i, j],
                    ]                    
                    """
        """
        _i = 0
        net_constraints_list = np.where(self._adj > 1)
        
        for i, j in zip(net_constraints_list[0],net_constraints_list[1]):
            _ci_x, _ci_y = self.boxes[i].center
            _cj_x, _cj_y = self.boxes[j].center
            ci_x = ci_x_l = cp.minimum(_ci_x, _cj_x)
            ci_y = ci_y_l = cp.minimum(_ci_y, _cj_y)
            ci_x_u = cp.maximum(_ci_x, _cj_x)
            ci_y_u = cp.maximum(_ci_y, _cj_y)            
            w_i = ci_x_u - ci_x_u
            h_i = ci_y_u - ci_y_l

            _j = 0
            for ii, jj in zip(net_constraints_list[0],net_constraints_list[1]):
                _cii_x, _cii_y = self.boxes[ii].center
                _cjj_x, _cjj_y = self.boxes[jj].center
                cj_x = cj_x_l = cp.minimum(_cii_x, _cjj_x)
                cj_y = cj_y_l = cp.minimum(_cii_y, _cjj_y)
                cj_x_u = cp.maximum(_cii_x, _cjj_x)
                cj_y_u = cp.maximum(_cii_y, _cjj_y)                
                w_j = cj_x_u - cj_x_l
                h_j = cj_y_u - cj_y_l

                constraints += [
                    ci_x + w_i/2 <= cj_x + self.boundary_W*(self.u_p[_i,_j] + self.u_q[_i,_j]) - FloorPlan.MARGIN + self.u_g[_i, _j],
                    ci_y + h_i/2 <= cj_y + self.boundary_H*(1 + self.u_p[_i,_j] - self.u_q[_i,_j]) - FloorPlan.MARGIN + self.u_g[_i, _j],
                    ci_x - w_j/2 >= cj_x - self.boundary_W*(1 - self.u_p[_i,_j] + self.u_q[_i,_j]) + FloorPlan.MARGIN - self.u_g[_i, _j],
                    ci_y - h_j/2 >= cj_y - self.boundary_H*(2 - self.u_p[_i,_j] - self.u_q[_i,_j]) + FloorPlan.MARGIN - self.u_g[_i, _j],
                ]
                _j += 1
            _i +=1
        """            
                        
        # Enforce the relative ordering of the boxes.
        for ordering in self.horizontal_orderings:
            constraints += self._order(ordering, True)
        for ordering in self.vertical_orderings:
            constraints += self._order(ordering, False)
            
        #obj = Minimize(2*(self.height + self.width))
        hpwls = [(net.U_x - net.L_x) + (net.U_y - net.L_y) for net in self.nets]
        hpwl = cp.sum(hpwls)
        self.h = hpwl
        #ce = cp.norm(self.u_g,1)
        
        #initial_placement_slack = cp.sum([box.slack for box in self.boxes])
        
        print('Compiled constraints')
        if self._obj:
            obj = Minimize(hpwl + 1.0*(cp.maximum(*hpwls) - cp.minimum(*hpwls)))
        else:
            obj = Minimize(0)
        p = Problem(obj, constraints)
        
        assert p.is_dcp() or p.is_dgp() or p.is_dqcp()

        p = p.solve(solver=cp.CBC, qcp=True,
                    warm_start=True, verbose=True, maximumSeconds=self.max_seconds, numberThreads=self.num_cores)
        return p, constraints
            

    def verify_constraints(self, constraints):
        return np.array([c.violation() for c in constraints])

    # Show the layout with matplotlib
    def show(self,savefig=False,idx=0):
        pylab.figure(facecolor='w')
        max_x = 0.0
        max_y = 0.0
        min_x = 1e8
        min_y = 1e8
        for k in range(len(self.boxes)):
            box = self.boxes[k]
            x,y = box.position
            if box.rotation:
                h,w = box.size
            else:
                w,h = box.size
                
            if x < min_x: min_x = x
            if y < min_y: min_y = y
            if x+w > max_x: max_x = x+w
            if y+h > max_y: max_y = y+h
                
            pylab.fill([x, x, x + w, x + w],
                       [y, y+h, y+h, y],zorder=-1)
            pylab.text(x+.5*w, y+.5*h, "%d" %(k))
        
        for k in range(len(self.nets)):
            net = self.nets[k]
            
            
            #modules = [[p.value for p in self.boxes[i].center] for i in net.moduleidxs]
            #mx, my = list(zip(*modules))

            mx = []
            my = []
            for pi, moduleidx in enumerate(net.moduleidxs):
                box = self.boxes[moduleidx]
                boxx,boxy = box.center
                boxw = box.w
                boxh = box.h
                                
                poff = net.offsets[pi]
                
                # not rotated, not mirrored
                # rotated, mirrored
                # not rotated, mirrored
                # rotated, not mirrored

                pinposx = boxx + box.nrnmx*(poff[0]*boxw) + \
                                 box.rmx*(-1*poff[1]*boxh) + \
                                 box.nrmx*(-1*poff[0]*boxw) + \
                                 box.rnmx*(poff[1]*boxh)
                
                pinposy = boxy + box.nrnmy*(poff[1]*boxh) + \
                                 box.rmy*(-1*poff[0]*boxw) + \
                                 box.nrmy*(-1*poff[1]*boxh) + \
                                 box.rnmy*(poff[0]*boxw)
                mx.append(pinposx.value)
                my.append(pinposy.value)
            
            pylab.plot(mx,my, color='gray',alpha=0.5)
            pylab.scatter(mx,my, color='black',alpha=0.8,s=5,zorder=1)

        
        pylab.axis([min_x, max_x, min_y, max_y])
        pylab.xticks([])
        pylab.yticks([])

        if savefig:
            pylab.savefig("./tmp2/{}.png".format(idx))
        pylab.show()
        
"""
%%time
boxes = [Box(10, 3), Box(5,2), Box(2, 4), Box(1,8), Box(3,10)]
fp = FloorPlan(boxes)
#fp.horizontal_orderings.append( [boxes[0], boxes[2], boxes[4]] )
#fp.horizontal_orderings.append( [boxes[1], boxes[2]] )
#fp.horizontal_orderings.append( [boxes[3], boxes[4]] )
#fp.vertical_orderings.append( [boxes[1], boxes[0], boxes[3]] )
#fp.vertical_orderings.append( [boxes[2], boxes[3]] )
p, c = fp.layout()
fp.show()
violations = fp.verify_constraints(c)
print(fp.height.value, fp.width.value)
print(2*(fp.height.value + fp.width.value))
assert np.all(violations <= 1e-5)
"""