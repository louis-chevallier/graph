import torch, torchvision
import cv2
import sys
from PIL import Image, ImageDraw, ImageFont
from utillc import EKON
from torch.autograd import Variable
from torch.optim import Adam
from torch import Tensor
import matplotlib.pyplot as plt 
import functools
from typing import Tuple

def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = torchvision.ops.box_area(boxes1)
    area2 = torchvision.ops.box_area(boxes2)

    EKON(boxes1)

    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    EKON(lt.shape)
    EKON(inter)
    EKON(union)
    return inter, union
class Graph :
    boxes = []
    constraints = []
    #font = ImageDraw.getfont()
    font = ImageFont.load_default()
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    image = Image.new('RGB', (512, 512))
    drawo = ImageDraw.Draw(image)
    
    def solve(self) :
        vrs = [ b.x for b in self.boxes]
        optimizer = Adam(vrs, lr=0.01)
        eye = torch.eye(len(vrs))
        z = torch.zeros(len(vrs) * 2)
        for i in range(10000) :
            boxes = [ b.box() for b in self.boxes]
            boxes = torch.vstack(boxes)

            loss = 0
            iou = torchvision.ops.box_iou(boxes, boxes) - eye
            #EKON(iou.shape)
            #EKON(iou)
            loss += iou.sum()             

            centers = torch.vstack([ b.center() for b in self.boxes])
            dist = centers[None,...] - centers[:,None,:]
            dist = dist.norm(dim=2)
            #loss += dist.sum() / 10

            xy =  torch.cat([ b.x for b in self.boxes])
            xy = torch.minimum(xy, z).sum()
            loss += -xy
            #EKON(loss)
            if loss == 0. :
                EKON(i)
                break
            loss.backward()
            optimizer.step()

        for i in range(2) :
            loss = 0
            boxes = [ b.box() for b in self.boxes]
            boxes = torch.vstack(boxes)

            
            _box_inter_union(boxes, boxes)
            iou = torchvision.ops.box_iou(boxes, boxes) - eye
            #EKON(iou.shape)
            #EKON(iou)
            loss += iou.sum()             

            centers = torch.vstack([ b.center() for b in self.boxes])
            dist = centers[None,...] - centers[:,None,:]
            dist = dist.norm(dim=2)
            #loss += dist.sum() / 10

            xy =  torch.cat([ b.x for b in self.boxes])
            xy = torch.minimum(xy, z).sum()
            loss += -xy
            
            loss.backward()
            optimizer.step()
            
        EKON(loss)
        EKON(iou.sum())
        EKON(iou)        
        

    def draw(self) :
        [ b.draw(self.drawo) for b in self.boxes]
        plt.imshow(self.image); plt.show()

class Box :
    def __init__(self, g, text, dep=(0,0)) :
        self.dep = dep
        self.text = text
        self.x = Variable(torch.zeros(2), requires_grad=True)
        g.boxes.append(self)
        text_box = g.font.getmask(text).getbbox()
        (left, top, right, bottom) = text_box
        #EKON(text_box)
        self.ww, self.hh = right - left, bottom - top
        
        #plt.imshow(image); plt.show()
        self.size = torch.tensor([self.ww, self.hh])
        EKON(self.box())

    def draw(self, draw) :
        xx, yy,_,_ = self.box().cpu().detach().numpy()
        draw.text((xx, yy), self.text, font=g.font)

    def box(self) :
        x = self.x + torch.tensor(self.dep)
        return torch.cat([x, x + self.size])

    def center(self) :
        b = self.box()
        return (b[0:2] + b[2:4]) / 2
        
    def connect(self, b) :
        pass
    
g = Graph()
a = Box(g, "A")
b = Box(g, "B", (0, 0))
c = Box(g, "ZZ")
a.connect(b)
g.solve()
g.draw()











