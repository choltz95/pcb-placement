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
        self.offsets = []
        for offset in offsets:
            self.offsets.append([BitVecVal(str(d),32) for d in offset])       
        self.idx = idx
        self.U_x = BitVec("{}U_x".format(idx),32)
        self.L_x = BitVec("{}L_x".format(idx),32)
        self.U_y = BitVec("{}U_y".format(idx),32)
        self.L_y = BitVec("{}L_y".format(idx),32)

class Box(object):
    """ A box in a floor packing problem. """
    ASPECT_RATIO = 1.0
    def __init__(self, width, height, initialx=0.0, initialy=0.0, initialr=0, initialmx=0, initialmy=0, 
                 idx=0, pl=True, r=False, m=False, terminal=False, min_area=None):
        self.min_area = min_area
        self.w = BitVecVal(height,32)
        self.h = BitVecVal(width,32)
        self.terminal = terminal
        self.pl = pl
        self.r = r # rotated
        self.m = m
        self.mx = m # mirrored
        self.my = m # mirrored
        self.idx = idx
                
        if pl:
            self.x = BitVec("{}x".format(idx),32)
            self.y = BitVec("{}y".format(idx),32)
        else:
            self.x = BitVecVal(initialx,32)
            self.y = BitVecVal(initialy,32)     
        if r:
            self.r = Bool("{}r".format(idx))
        else:
            self.r = BoolVal(initialr)
        if m:
            self.mx = Bool("{}mx".format(idx))
            self.my = Bool("{}my".format(idx))
        else:
            self.mx=BoolVal(initialmx)
            self.my=BoolVal(initialmy)
            
        #self.xorrmx =  # xor wigit        
        self.rmx    = And(self.r,self.mx) # rotated and mirrrored 
        self.nrnmx  = And(Not(self.r), Not(self.mx)) # not rotated and not mirrored
        self.nrmx   = And(Not(self.r),self.mx) # not rotated and mirrored
        self.rnmx   = And(self.r, Not(self.mx)) # rotated and not mirrored
        
        #self.xorrmy =  # xor wigit
        self.rmy    = And(self.r,self.my) # rotated and mirrrored 
        self.nrnmy  = And(Not(self.r), Not(self.my)) # not rotated and not mirrored
        self.nrmy   = And(Not(self.r),self.my) # not rotated and mirrored
        self.rnmy   = And(self.r, Not(self.my)) # rotated and not mirrored
        
        self.netidxs = []

    @property
    def position(self):
        return (np.round(self.x,2), np.round(self.y,2))

    @property
    def size(self):
        return (np.round(self.w.as_long(),2), np.round(self.h.as_long(),2))
    
    @property
    def center(self):
        r  = If(self.r,1,0)
        return self.x + r*self.h/2 + (1-r)*self.w/2, self.y + r*self.w/2 + (1-r)*self.h/2

    @property
    def left(self):
        return self.x 

    @property
    def right(self):
        return self.x + self.w

    @property
    def bottom(self):
        return self.y

    @property
    def top(self):
        return self.y + self.h

    #@property
    #def rotation(self):
    #    return self.r.value
    
    #@property
    #def mirror(self):
    #    return self.m.value

"""
http://cc.ee.ntu.edu.tw/~ywchang/Courses/PD_Source/EDA_floorplanning.pdf
"""
class FloorPlan(object):
    MARGIN = 0.0
    ASPECT_RATIO = 5.0
    def __init__(self, boxes, nets, adj,norelpairs=None, horizontal_orderings=[], vertical_orderings=[], boundary_W=100, boundary_H=100, max_seconds=100, num_cores=1, name=""):

        self.boxes = boxes
        self.nets = nets
        self.num_nodes = len(boxes)
        self.name=name
        
        self.boundary_W = BitVecVal(boundary_W,32)
        self.boundary_H = BitVecVal(boundary_H,32)
        
        self.max_x = BitVec("max_x",32)
        self.max_y = BitVec("max_y",32)
        self.min_x = BitVec("min_x",32)
        self.min_y = BitVec("min_y",32)
        
        self.horizontal_orderings = []
        self.vertical_orderings = []
        
        self.max_seconds = max_seconds
        self.num_cores = num_cores

    # Compute minimum perimeter layout.
    def layout(self, solve=True):
        constraints = []
        for box in self.boxes:
            #mx = If(box.mx,1,0)
            #my = If(box.my,1,0)
            #r = If(box.r,1,0)         

            if (not box.pl):
                continue
            # Enforce that boxes lie in bounding box. 
            constraints += [b.x >= 0 for b in self.boxes if b.pl]
            constraints += [b.y >= 0 for b in self.boxes if b.pl]
            constraints += [b.x+If(b.r,b.h,b.w) <= self.boundary_W for b in self.boxes if b.pl]
            constraints += [b.y+If(b.r,b.w,b.h) <= self.boundary_H for b in self.boxes if b.pl]
            """
            constraints += [self.min_x <= b.x for b in self.boxes if b.pl]
            constraints += [self.min_y <= b.y for b in self.boxes if b.pl]
            
            constraints += [self.max_x >= b.x+If(b.r,b.h,b.w) for b in self.boxes if b.pl]
            constraints += [self.max_y >= b.y+If(b.r,b.w,b.h) for b in self.boxes if b.pl] 
            
            constraints += [self.max_x <= self.boundary_W,
                            self.max_y <=self.boundary_H]
            
            constraints += [self.min_x >= 0,
                            self.min_y >=0]
            """
            
        # wirelength minimization
        for net in self.nets:
            if len(net.moduleidxs) <= 1: continue

            mx = []
            my = []
            for pi, moduleidx in enumerate(net.moduleidxs):
                box = self.boxes[moduleidx]
                boxx, boxy = box.x, box.y
                boxw = box.w
                boxh = box.h
                poff = net.offsets[pi]
                
                # not rotated, not mirrored
                # rotated, mirrored
                # not rotated, mirrored
                # rotated, not mirrored
                pinposx = boxx + If(box.nrnmx, poff[0]*boxw, 0) + \
                                 If(box.rmx, -1*poff[1]*boxh, 0) + \
                                 If(box.nrmx, -1*poff[0]*boxw, 0) + \
                                 If(box.rnmx, poff[1]*boxh, 0)
                
                
                pinposy = boxy + If(box.nrnmy, poff[1]*boxh, 0) + \
                                 If(box.rmy, -1*poff[0]*boxw, 0) + \
                                 If(box.nrmy, -1*poff[1]*boxh, 0) + \
                                 If(box.rnmy, poff[0]*boxw, 0)
                
                mx.append(pinposx)
                my.append(pinposy)
                                     
            constraints += [net.L_x <= x for x in mx]
            constraints += [net.U_x >= x for x in mx]
            constraints += [net.L_y <= y for y in my]
            constraints += [net.U_y >= y for y in my]
                 
        # nonoverlap constraints
        for i in range(len(self.boxes)):
            for j in range(i+1,len(self.boxes)):
                b_i = self.boxes[i]
                b_j = self.boxes[j]
                
                if (not b_i.pl) or (not b_j.pl):
                    continue
                
                x_i, y_i = b_i.x, b_i.y
                w_i, h_i = b_i.w, b_i.h
                r_i = b_i.r
                
                x_j, y_j = b_j.x, b_j.y
                w_j, h_j = b_j.w, b_j.h
                r_j = b_j.r
                
                c1 = x_i + If(r_i,h_i,w_i) <= x_j 
                c2 = y_i + If(r_i,w_i,h_i) <= y_j 
                c3 = x_i - If(r_j,h_j,w_j) >= x_j
                c4 = y_i - If(r_j,w_j,h_j) >= y_j
                
                constraints += [
                    Or(c1,c2,c3,c4)
                ]
                                
        # Enforce the relative ordering of the boxes.
        #for ordering in self.horizontal_orderings:
        #    constraints += self._order(ordering, True)
        #for ordering in self.vertical_orderings:
        #    constraints += self._order(ordering, False)
            
        hpwls = [(net.U_x - net.L_x) + (net.U_y - net.L_y) for net in self.nets]
        #set_option('timeout',1000*self.max_seconds)
        set_option('parallel.enable', True)
        opt = Optimize()
        opt.set("timeout", 1000*self.max_seconds)
        for constraint in constraints:
            opt.add(constraint)
        #opt.minimize(Sum(hpwls))
        obj = opt.minimize(Sum(hpwls))
        return opt, obj
            
    def verify_constraints(self, constraints):
        return np.array([c.violation() for c in constraints])

    # plot the layout
    def show(self, model):
        pylab.figure(facecolor='w')
        max_x = 0.0
        max_y = 0.0
        min_x = 1e8
        min_y = 1e8
        for k in range(len(self.boxes)):
            box = self.boxes[k]
            if box.pl:
                x,y = (float(model[box.x].as_long()), float(model[box.y].as_long()))
            else:
                x,y = (np.round(box.x.as_long(),2), np.round(box.y.as_long(),2))
            if model[box.r]:
                h,w = box.size
            else:
                w,h = box.size
                
            if x < min_x: min_x = x
            if y < min_y: min_y = y
            if x+w > max_x: max_x = x+w
            if y+w > max_y: max_y = y+w
                
            pylab.fill([x, x, x + w, x + w],
                       [y, y+h, y+h, y])
            pylab.text(x+.5*w, y+.5*h, "%d" %(k+1))
        
        for k in range(len(self.nets)):
            net = self.nets[k]

            mx = []
            my = []
            for pi, moduleidx in enumerate(net.moduleidxs):
                box = self.boxes[moduleidx]
                #boxx,boxy = box.center
                if box.pl:
                    boxx = float(model[box.x].as_long()) + bool(model[box.r])*box.h.as_long()/2 + (1-bool(model[box.r]))*box.w.as_long()/2 
                    boxy = float(model[box.y].as_long()) + bool(model[box.r])*box.w.as_long()/2 + (1-bool(model[box.r]))*box.h.as_long()/2
                else:
                    assert False
                    boxx = box.x.as_long() + bool(model[box.r])*box.h.as_long()/2 + (1-bool(model[box.r]))*box.w.as_long()/2 
                    boxy = box.y.as_long() + bool(model[box.r])*box.w.as_long()/2 + (1-bool(model[box.r]))*box.h.as_long()/2

                nrnmx = model.eval(box.nrnmx)
                rmx = model.eval(box.rmx)
                nrmx = model.eval(box.nrmx)
                rnmx = model.eval(box.rnmx)
                
                nrnmy = model.eval(box.nrnmy)
                rmy = model.eval(box.rmy)
                nrmy = model.eval(box.nrmy)
                rnmy = model.eval(box.rnmy)
                
                poff = net.offsets[pi]
                
                boxw = box.w.as_long()
                boxh = box.h.as_long()
                
                # not rotated, not mirrored
                # rotated, mirrored
                # not rotated, mirrored
                # rotated, not mirrored
                pinposx = boxx + bool(nrnmx)*(float(poff[0].as_long())*boxw) + \
                                 bool(rmx)*(-1*float(poff[1].as_long())*boxh) + \
                                 bool(nrmx)*(-1*float(poff[0].as_long())*boxw) + \
                                 bool(rnmx)*(float(poff[1].as_long())*boxh)
                
                pinposy = boxy + bool(nrnmy)*(float(poff[1].as_long())*boxh) + \
                                 bool(rmy)*(-1*float(poff[0].as_long())*boxw) + \
                                 bool(nrmy)*(-1*float(poff[1].as_long())*boxh) + \
                                 bool(rnmy)*(float(poff[0].as_long())*boxw)
                mx.append(pinposx)
                my.append(pinposy)
            
            pylab.plot(mx,my, color='gray',alpha=0.25)
            pylab.scatter(mx,my, color='black',alpha=0.8,s=2)
        
        pylab.axis([min_x, max_x, min_y, max_y])
        pylab.xticks([])
        pylab.yticks([])

        pylab.show()