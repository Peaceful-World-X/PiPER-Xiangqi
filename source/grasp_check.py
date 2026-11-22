"""Measure the distance from the animated disc rim to each CAD finger pad."""
import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial import ConvexHull
from scipy.optimize import linprog


class Finger:
    def __init__(self, wrist, name, cache):
        self.body = wrist.GetChild(name)
        self.geom = next(p for p in self.body.GetChildren() if p.IsA(UsdGeom.Mesh))
        geo = UsdGeom.Mesh(self.geom)
        verts = np.asarray(geo.GetPointsAttr().Get())
        self.hull = verts[ConvexHull(verts).vertices]
        self.faces = np.asarray(geo.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
        wrist_inv = np.linalg.inv(cache.GetLocalToWorldTransform(wrist))
        basis = np.asarray(cache.GetLocalToWorldTransform(self.geom)) @ wrist_inv
        tri = (verts @ basis[:3,:3] + basis[3,:3])[self.faces]
        self.slide_op = UsdGeom.Xformable(self.body).GetOrderedXformOps()[-1]
        self.baseline = np.linalg.norm(self.slide_op.Get(0))
        self.side = np.sign(tri[:,:,1].mean())
        # Only triangles spanning the contact patch can meet a ray along the closing axis.
        lo, hi = tri.min(1), tri.max(1)
        self.tri = tri[(lo[:,0] < .002) & (hi[:,0] > -.002) &
                       (lo[:,2] < .1395) & (hi[:,2] > .132)]
        self.a = self.tri[:,0]
        self.b = self.tri[:,1] - self.a
        self.c = self.tri[:,2] - self.a
        self.det = self.b[:,0]*self.c[:,2] - self.b[:,2]*self.c[:,0]
        self.valid = abs(self.det) > 1e-13

    def gap(self, centre, normal, radius, half_height, frame):
        dx, dz = centre[0]-self.a[:,0], centre[2]-self.a[:,2]
        u = np.divide(dx*self.c[:,2]-dz*self.c[:,0], self.det,
                      out=np.full_like(dx,-1), where=self.valid)
        v = np.divide(self.b[:,0]*dz-self.b[:,2]*dx, self.det,
                      out=np.full_like(dx,-1), where=self.valid)
        use = (u>=-1e-6) & (v>=-1e-6) & (u+v<=1+1e-6)
        assert use.any(), ("disc missed finger surface",frame,centre)
        yy = (self.a[:,1]+u*self.b[:,1]+v*self.c[:,1])[use]
        slide = np.linalg.norm(self.slide_op.Get(frame))
        distance = (self.side*(yy-centre[1])).min() + slide-self.baseline
        cosine = abs(normal[1])
        radial = radius / np.sqrt(max(1e-12,1-cosine*cosine))
        assert radial*cosine < half_height-.001, ("disc held on its face",frame)
        return distance-radial

    def planes(self, cache):
        matrix = np.asarray(cache.GetLocalToWorldTransform(self.geom))
        return ConvexHull(self.hull @ matrix[:3,:3]+matrix[3,:3]).equations


def verify_grasps(stage):
    suffix = "/base_link/link1/link2/link3/link4/link5/link6"
    wrists = [stage.GetPrimAtPath("/World/"+s+"PiPER"+suffix) for s in ("Left","Right")]
    cache = UsdGeom.XformCache(0)
    jaws = [[Finger(wrist,n,cache) for n in ("link7","link8")] for wrist in wrists]
    smallest, largest, drift = np.inf, -np.inf, 0.
    count = transfers = 0
    for piece in stage.GetPrimAtPath("/World/Pieces").GetChildren():
        donor,receiver,pick,meet,drop = [piece.GetAttribute("grasp:"+name).Get() for name in
                                       ("donor","receiver","take","handover","drop")]
        vertices = np.asarray(UsdGeom.Mesh(piece.GetChild("Edge")).GetPointsAttr().Get())
        radius = np.linalg.norm(vertices[:,:2],axis=1).max()
        half_height = vertices[:,2].max()
        phases = [(donor,pick,drop)] if meet < 0 else [(donor,pick,meet),(receiver,meet,drop)]
        for owner,begin,end in phases:
            initial = None
            for tick in range(begin*2,end*2+1):
                at=Usd.TimeCode(tick*.5)
                cache.SetTime(at)
                rel = np.asarray(cache.GetLocalToWorldTransform(piece)) @ np.linalg.inv(
                    cache.GetLocalToWorldTransform(wrists[owner]))
                if initial is None:
                    initial = rel
                drift = max(drift,np.linalg.norm(rel[3,:3]-initial[3,:3]))
                for finger in jaws[owner]:
                    gap = finger.gap(rel[3,:3],rel[2,:3],radius,half_height,at)
                    smallest,largest=min(smallest,gap),max(largest,gap)
                    count+=1
        if meet >= 0:
            transfers += 1
            # During shared grip the fingers of the opposite arms must also clear one another.
            for t in [meet-1,meet,meet+1]:
                cache.SetTime(Usd.TimeCode(t))
                for left in jaws[0]:
                    for right in jaws[1]:
                        planes=np.vstack([left.planes(cache),right.planes(cache)])
                        overlap=linprog([0,0,0],A_ub=planes[:,:3],b_ub=-planes[:,3],
                                        bounds=[(None,None)]*3,method="highs")
                        assert overlap.status==2, ("finger hulls overlap",piece.GetName(),t)
    print(f"Measured {count} rim/pad distances, {transfers} exchanges: "
          f"{smallest*1000:.3f} to {largest*1000:.3f} mm, peak local drift {drift*1000:.3f} mm.",flush=True)
    assert smallest > -.00035 and largest < .00035 and drift < .00035
