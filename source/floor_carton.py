"""Floor carton reconstructed from the opening shot, in metres."""
import cv2 as cv
import numpy as np

# Clockwise in the photograph, starting at the back left corner.
PHOTO = np.float32([[68,29],[211,56],[217,167],[58,141]])
CORNERS = np.array([[.74447,.19690],[.90363,.14864],[.86665,-.07885],[.69955,-.04084]])
UV = np.float32([[0,1],[1,1],[1,0],[0,0]])
H = cv.getPerspectiveTransform(PHOTO, UV)
BASE = -.77
HEIGHT = .255
LEFT = [(139.2,42),(138.7,54),(137.0,69),(135.7,84),(133.6,104),(131.5,117.0),
        (129.2,121.5),(119.7,125.2),(124.3,128.7),(131.0,131.2),(134.8,141.2),(136.8,150.7),(137.5,155)]
RIGHT = [(140.4,42),(140.3,61),(140.1,78),(140.0,97),(139.9,117),(139.7,135),(139.6,146),(139.5,155)]


def uv(points):
    return cv.perspectiveTransform(np.float32([points]),H)[0]


def lip(v, points):
    edge=uv(points)
    order=np.argsort(edge[:,1])
    return np.interp(v,edge[order,1],edge[order,0])


def side_panel(edge,u,t):
    i,j=[(3,2),(2,1),(1,0),(0,3)][edge]
    tangent=CORNERS[j]-CORNERS[i]
    normal=np.array([tangent[1],-tangent[0]])/np.linalg.norm(tangent)
    base=(1-u)[...,None]*CORNERS[i]+u[...,None]*CORNERS[j]
    bow=np.sin(np.pi*u)*(.0015*np.sin(np.pi*t)-.0011*np.cos(t*7+edge))
    # Rounded compression close to the floor and one gentle kink near the front corner.
    tuck=np.sin(np.pi*u)**.5*(.0028*np.exp(-(t/.13)**2)+.0012*np.exp(-((t-.31)/.04)**2))
    if edge==0:
        tuck+=.0041*np.exp(-((u-.89+.08*t)/.056)**2)*np.sin(np.pi*u)*np.sin(np.pi*t)
    xy=base+(bow-tuck)[...,None]*normal
    xy[:,:,0]-=.022*(1-np.exp(-(1-t)/.32))
    z=BASE+HEIGHT*t+.0014*t*np.sin(np.pi*u)*np.sin(4*u+edge)
    return np.dstack((xy,z)),normal


def xy_at(u,v):
    back=(1-u)[...,None]*CORNERS[0]+u[...,None]*CORNERS[1]
    front=(1-u)[...,None]*CORNERS[3]+u[...,None]*CORNERS[2]
    return (1-v)[...,None]*front+v[...,None]*back


def surface(g,path,points,uvs,material,upward=True):
    # Rows and columns form a regular patch. Smooth normals follow the bent sheet.
    rows,cols=points.shape[:2]
    index=np.arange(rows*cols).reshape(rows,cols)
    faces=np.stack((index[:-1,:-1],index[1:,:-1],index[1:,1:],index[:-1,1:]),axis=-1).reshape(-1,4)
    normals=np.cross(np.gradient(points,axis=0),np.gradient(points,axis=1))
    normals/=np.maximum(np.linalg.norm(normals,axis=2,keepdims=True),1e-12)
    if not upward:
        normals=-normals
        faces=faces[:,::-1]
    ob=g.mesh(path,points.reshape(-1,3),faces,material,normals.reshape(-1,3),uvs.reshape(-1,2))
    ob.CreateDoubleSidedAttr(True)


def floor_carton(g,path):
    for name in ('floorbox_lid','floorbox_face','floorbox_left'):
        g.material(name,(.40,.30,.20),.79,0,f'./assets/surfaces/{name}.png',name,normal_map=False)
    g.material('CartonLiner',(.24,.162,.088),.88)
    g.material('PulledPaper',(.44,.358,.255),.88)
    g.material('FoldShadow',(.038,.022,.012),.96)
    g.group(path)
    front_edge=(CORNERS[2]-CORNERS[3]);front_normal=np.array([front_edge[1],-front_edge[0]])
    front_normal/=np.linalg.norm(front_normal)
    for side in range(4):
        u,t=np.meshgrid(np.linspace(0,1,76),np.linspace(0,1,40),indexing='ij')
        vertices,normal=side_panel(side,u,t)
        mat={0:'floorbox_face',3:'floorbox_left'}.get(side,'floorbox_lid')
        surface(g,path+f'/Side{side}',vertices,np.dstack((u,t)),mat)
        u,s=np.meshgrid(np.linspace(0,1,76),np.linspace(-.5,.5,5),indexing='ij')
        t=.296+.011*np.sin(5*u+side)+.024*s
        vertices,_=side_panel(side,u,t)
        vertices[:,:,:2]+=.00055*normal
        surface(g,path+f'/Band{side}',vertices,np.dstack((u,t)),'PackingBand')
    # Deep backing is visible through the actual opening between the two flaps.
    z=BASE+HEIGHT-.039
    g.mesh(path+'/Inside',np.c_[CORNERS,np.full(4,z)],[[3,2,1,0]],'FoldShadow').CreateDoubleSidedAttr(True)
    for i,edge in enumerate((LEFT,RIGHT)):
        s,v=np.meshgrid(np.linspace(0,1,34),np.linspace(0,1,117),indexing='ij')
        end=lip(v,edge)
        u=s*end if i==0 else 1-s*(1-end)
        xy=xy_at(u,v)
        # The left flap has a single broad diagonal buckle. The right flap holds
        # a shallow transverse printing/packing crease. Both tips are free.
        if i==0:
            d=v-(.67-.25*s)
            ripple=-.0029*np.exp(-(d/.085)**2)+.0013*np.exp(-((d-.095)/.12)**2)
            ripple-=.0019*np.exp(-((v-.27)/.07)**2)*np.sin(np.pi*s)
            tip=-.0042*np.sin(np.pi*v)**1.2+.0013*np.exp(-((v-.27)/.08)**2)
        else:
            d=v-(.43-.045*s)
            ripple=-.0017*np.exp(-(d/.060)**2)+.0008*np.exp(-((d-.06)/.09)**2)
            tip=.0024*np.sin(np.pi*v)
        z=BASE+HEIGHT+ripple*np.sin(np.pi*s)+tip*s**1.4
        z+=.0011*np.sin(3.2*v+1.6*i)*np.sin(np.pi*s)
        verts=np.dstack((xy,z))
        surface(g,path+f'/Lid{i}',verts,np.dstack((u,v)),'floorbox_lid',upward=i==0)
        cut=verts[-1]
        under=cut-np.array([0,0,.0017])
        column=np.stack([cut,under],axis=1)
        surface(g,path+f'/RawCut{i}',column,np.zeros((*column.shape[:2],2)),'CartonLiner')
    # The old adhesive has pulled the printed liner away in two unequal patches.
    for n,(a,b,depth) in enumerate([(.014,.285,.009),(.414,.616,.014)]):
        u,t=np.meshgrid(np.linspace(a,b,54),np.linspace(0,1,5),indexing='ij')
        local=(u-a)/(b-a)
        contour=(np.maximum(0,np.sin(np.pi*local))**.72)*(.62+.23*np.sin(10*local+n)+.13*np.sin(31*local))
        side_t=1-.002-(.0016+depth*contour)*t/HEIGHT
        vertices,normal=side_panel(0,u,side_t)
        vertices[:,:,:2]+=(.00065+.0014*contour*t)[:,:,None]*normal
        surface(g,path+f'/TornFacing{n}',vertices,np.dstack((u,t)),'PulledPaper',upward=False)
