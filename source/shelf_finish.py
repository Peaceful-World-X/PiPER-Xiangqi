"""Workshop timber, folded packing cartons and used tool finishes."""
import numpy as np


def folds(seed):
    base = [
        (.485, .91, .567, .58, 1.0), (.502, .91, .523, .62, .65),
        (.535, .65, .583, .45, .74), (.507, .07, .555, .28, .96),
        (.54, .08, .562, .21, .63), (.707, .91, .750, .60, .90),
        (.724, .22, .755, .52, .85), (.735, .10, .710, .23, .62),
        (.465, .60, .411, .30, .83), (.475, .10, .444, .28, .62),
        (.355, .92, .389, .73, .70), (.312, .19, .345, .43, .66),
        (.825, .87, .857, .65, .73), (.895, .11, .863, .29, .60),
        (.07, .94, .12, .61, .79), (.161, .12, .214, .43, .63),
        (.205, .54, .241, .66, .51)
    ]
    rng = np.random.default_rng(seed * 753)
    for u0, v0, u1, v1, strength in base:
        du = rng.uniform(-.018, .018)
        yield (u0 + du, v0, u1 + du, v1 + rng.uniform(-.055, .055), strength)


def crease_profile(u, v, seed):
    # Coordinates in metres keep the fold width consistent around a tall or short box.
    length, height = (.668, .098) if seed == 13 else (.698, .146)
    depression = np.zeros(np.broadcast_shapes(np.shape(u), np.shape(v)), np.float32)
    for u0, v0, u1, v1, strength in folds(seed):
        dx, dz = (u1-u0)*length, (v1-v0)*height
        x, z = (u-u0)*length, (v-v0)*height
        param = (x*dx + z*dz)/(dx*dx + dz*dz)
        span = np.clip(param, 0, 1)
        bend = .0017*np.sin(span*np.pi*1.55) + .0006*np.sin(span*np.pi*4)
        across = (x*dz - z*dx)/np.hypot(dx,dz) - bend
        fade = (np.maximum(0,np.sin(np.pi*span))**.6)*strength*(.68 if seed == 13 else 1.0)
        core = np.exp(-.5*(across/.0017)**2)
        shoulder = np.exp(-.5*((across-.0036)/.0025)**2)
        trough = np.exp(-.5*(across/.0058)**2)
        depression += fade * (.0028*trough+.0033*core-.0017*shoulder)
    return depression


def finishes(g):
    for i in range(4):
        m=f'rackwood{i}'
        g.material(m,(.21,.16,.10),.59,0,f'./assets/surfaces/{m}.png',m)
    for m, roughness, metal in [("rack_iron", .46, .48), ("worn_orange", .42, .30)]:
        g.material(m, (.21, .15, .10), roughness, metal, f"./assets/surfaces/{m}.png", m)
    for m in ("carton_side_13", "carton_top_13_0", "carton_top_13_1"):
        g.material(m, (.37, .28, .18), .79, 0, f"./assets/surfaces/{m}.png", m, normal_map=False)
    g.material("PackingBand",(.63,.59,.47),.39,.035)
    g.material("TornLiner",(.34,.234,.12),.87,0)
    g.material("InsideCarton",(.052,.039,.027),.97,0)
    g.material("OldTape",(.40,.283,.142),.38,.025)
    g.material("Oxide",(.091,.063,.045),.54,.55)


def carton(g,path,x,y,z,w,d,h):
    seed = 13
    rng=np.random.default_rng(seed)
    origin=np.array([x,y,z]); g.group(path,origin)
    # Rounded corners carry the strap continuously around compressed cardboard.
    w2,d2=w/2,d/2;corner=.011
    perimeter=[]
    for cx,cy,theta in [(w2-corner,d2-corner,0),(-w2+corner,d2-corner,90),
                        (-w2+corner,-d2+corner,180),(w2-corner,-d2+corner,270)]:
        for a in np.deg2rad(theta+np.linspace(0,90,5)):
            perimeter.append([cx+corner*np.cos(a),cy+corner*np.sin(a)])
    # Subdivide long, otherwise flat faces to support small dents in the panels.
    ring=[]
    for i,a in enumerate(perimeter):
        a=np.array(a);b=np.array(perimeter[(i+1)%len(perimeter)])
        n=max(1,round(np.linalg.norm(b-a)/.0028))
        ring.extend(a+(b-a)*j/n for j in range(n))
    ring=np.array(ring); ring=np.vstack((ring,ring[0]))
    # Use surface normals so a diagonal pressure mark recesses into the cardboard.
    tangent=np.roll(ring[:-1],-1,axis=0)-np.roll(ring[:-1],1,axis=0)
    normal=np.column_stack([tangent[:,1],-tangent[:,0]])
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-9)
    normal=np.vstack((normal,normal[0]))
    run=np.r_[0,np.cumsum(np.linalg.norm(np.diff(ring,axis=0),axis=1))]
    phase=seed*.04
    def contour(t):
        squeeze=(.0020*np.exp(-(t/.18)**2)+.0019*np.exp(-((t-.54)/.07)**2)+
                 .0028*np.exp(-((t-.94)/.20)**2)*(1+.8*np.sin(48*run+phase)))
        wobble=.0017*np.sin(37*run+phase)*np.sin(t*np.pi)+.0012*np.sin(78*run-3*t)
        dent=crease_profile(run/run[-1],t,seed)
        corner_crush=.0045*np.exp(-((run/run[-1]-.48)/.039)**2)*np.exp(-((t-.84)/.23)**2)
        xy=ring+normal*(wobble-squeeze-dent-corner_crush)[:,None]
        zz=h*t+.0029*t**3*np.sin(28*run+phase)
        return np.column_stack([xy,zz])
    heights=np.unique(np.r_[np.linspace(0,1,43), .018, .51, .54, .56, .986])
    def surface(name,rings,ts,mat):
        verts=np.vstack(rings);stride=len(ring)
        faces=[[j*stride+i,j*stride+i+1,(j+1)*stride+i+1,(j+1)*stride+i]
               for j in range(len(rings)-1) for i in range(stride-1)]
        uv=np.vstack([np.column_stack([run/run[-1],np.full(stride,t)]) for t in ts])
        points = np.stack(rings)[:, :-1]
        tangent = np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1)
        vertical = np.gradient(points, axis=0)
        normals = np.cross(tangent, vertical)
        normals /= np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-12)
        normals = np.concatenate((normals, normals[:, :1]), axis=1).reshape(-1, 3)
        g.mesh(path+'/'+name,verts,faces,mat,normals,uv)
    surface('Sides',[contour(t) for t in heights],heights,f'carton_side_{seed}')
    rings=[]
    for t in (.514,.52,.548,.554):
        v=contour(t);v[:,:2]+=.00065*normal;rings.append(v)
    surface('Strap',rings,[0,.12,.88,1],'PackingBand')
    g.box(path+'/StrapLap',(w*.21,-d2-.00035,.534*h),(.021,.0019,.0045),'PackingBand',.0005)
    g.box(path+'/DarkInterior',(0,0,h-.024),(w-.013,d-.012,.003),'InsideCarton',.001)
    # Folded under-flaps sit beneath two bowed lips. A narrow dark slit remains between them.
    for i,sgn in enumerate((-1,1)):
        g.box(path+f'/EndFlap{i}',(0,sgn*d*.34,h-.007),(w-.010,d*.29,.002),'TornLiner',.0008)
        ss=np.linspace(0,1,24)
        yy=np.linspace(-d2+.003,d2-.003,62)
        vs=[];uv=[]
        for f in ss:
            for yy0 in yy:
                along=(yy0+d2)/d
                notch=.012*np.exp(-((along-(.32 if i==0 else .62))/.055)**2)
                torn=.0009*np.sin(724*yy0+phase)+.00065*np.sin(1041*yy0+i)
                gap=.0025+.0035*i+.006*np.sin(np.pi*along)**2+notch+torn
                xx=sgn*((w2-.0025)*(1-f)+gap*f)
                bow=(.008*np.cos(22*yy0+phase+i)+.0042*(1-2*i))*f**1.6
                # A curled corner and two transverse creases break the top silhouette.
                bow-=.0055*np.exp(-((along-.16)/.14)**2)*np.exp(-((f-.18)/.20)**2)
                for centre,slope,extent in [(.19,.22,.39),(.73,-.17,.34),(.087,-.19,.56)]:
                    step=along-centre-slope*(f-.5)
                    bow+=(-.0028*np.exp(-(step/.022)**2)+.0014*np.exp(-((step-.028)/.035)**2))*np.exp(-((f-.51)/extent)**2)
                vs.append((xx,yy0,h+bow))
                uv.append((f,along))
        st=len(yy)
        faces=[[j*st+k,j*st+k+1,(j+1)*st+k+1,(j+1)*st+k]
               for j in range(len(ss)-1) for k in range(st-1)]
        if sgn<0:faces=[face[::-1] for face in faces]
        grid=np.reshape(vs,(len(ss),st,3))
        normals=np.cross(np.gradient(grid,axis=1),np.gradient(grid,axis=0))*sgn
        normals/=np.maximum(np.linalg.norm(normals,axis=2,keepdims=True),1e-9)
        g.mesh(path+f'/TopFlap{i}',vs,faces,f'carton_top_{seed}_{i}',normals.reshape(-1,3),uv)
        pts=grid[-1]
        lower=pts-np.array([0,0,.0024])
        # The naturally broken cut exposes the thickness of the corrugated liner.
        rim_faces=[[k,k+1,st+k+1,st+k] for k in range(st-1)]
        if sgn<0:rim_faces=[face[::-1] for face in rim_faces]
        rim=g.mesh(path+f'/CutFace{i}',np.vstack([pts,lower]),rim_faces,'TornLiner')
        rim.CreateDoubleSidedAttr(True)
        g.wire(path+f'/RoughCut{i}',pts,.00062,'TornLiner')
        for nick in (11,29,44):
            off=rng.uniform(.001,.003)
            a=pts[nick];b=pts[nick+2]
            g.wire(path+f'/LiftedFibre{i}_{nick}',[a,(a+b)*.5+[-sgn*off,0,.0018],b],.00043,'PackingBand')
    for j,(xx,yy) in enumerate([(-w*.32,-d*.25),(w*.29,d*.29)]):
        g.box(path+f'/TapeRepair{j}',(xx,yy,h+.0015),(.029,.016,.0009),'OldTape',.0004)
    # Crease just inside a folded front corner.
    g.wire(path+'/CornerCrease',[(-w2+.014,-d2+.0002,.015),(-w2+.009,-d2-.0002,.035),
                                 (-w2+.015,-d2+.0001,.047)],.00065,'TornLiner')


def vessel(g,path,x,y,z,r,h,mat):
    # Turned rolled lip and shoulder catch the window at distinct angles.
    profile=np.array([[.82,0],[.95,.017],[1,.052],[1,.09],[.98,.14],[.98,.77],
                      [1,.83],[.985,.90],[.91,.945],[.69,.99],[.58,1.]])
    count=80;ang=np.linspace(0,2*np.pi,count+1);stride=count+1
    verts=[];norm=[];uv=[]
    for (rad,ht),(dr,dh) in zip(profile,np.gradient(profile,axis=0)):
        rad*=r;ht*=h;dr*=r;dh*=h;den=np.hypot(dr,dh)
        verts.extend(zip(x+rad*np.cos(ang),y+rad*np.sin(ang),np.full(stride,z+ht)))
        norm.extend(zip(np.cos(ang)*dh/den,np.sin(ang)*dh/den,np.full(stride,-dr/den)))
        uv.extend((k/count,ht/h) for k in range(stride))
    faces=[[j*stride+k,j*stride+k+1,(j+1)*stride+k+1,(j+1)*stride+k] for j in range(len(profile)-1) for k in range(count)]
    g.mesh(path+'/Shell',verts,faces,mat,norm,uv)
    g.tube(path+'/Cap',(x,y,z+h-.0008),(x,y,z+h+.0015),r*.60,'Oxide')
