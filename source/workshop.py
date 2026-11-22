from scipy.spatial.transform import Rotation
import numpy as np
from pxr import Sdf, UsdLux, UsdShade
from shelf_finish import finishes, carton, vessel
from floor_carton import floor_carton


def mount_bridge(g, path):
    # The steel plate in the reference laps over the ply edge. Its underside
    # clears the laminate, with a thicker, turned-down nose beyond the desk.
    profile = np.array([
        (-.3210, -.01355), (-.3205, -.01305), (-.1385, -.01305),
        (-.1385, -.01495), (-.3040, -.01495), (-.3190, -.01880),
        (-.3205, -.01880), (-.3210, -.01830),
    ])
    n = len(profile)
    points = [(x, y, z) for x in (-.426, .426) for y, z in profile]
    points.extend([(x, -.3200, -.0140) for x in (-.426, .426)])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend([(i, i+n, j+n), (i, j+n, j),
                      (2*n, i, j), (2*n+1, j+n, i+n)])
    return g.mesh(path, points, faces, "dark")


def left_shelf(g, workshop, floor):
    path = workshop + "/LeftRack"
    g.group(path, (-1.356, .378, 0), Rotation.from_euler("z", -10.9, degrees=True))
    width, depth = .59, .380
    decks = [-.486, -.137, .225, .592]
    g.material("RackPaint", (.147, .153, .098), .65, .31)
    g.material("RackHardware", (.060, .054, .038), .57, .47)
    g.material("RackBoardEdge", (.147, .107, .076), .84)
    for i, height in enumerate(decks):
        material = f"LeftShelfBoard{i}"
        g.material(material, (.23, .19, .15), .85, 0,
                   f"./assets/surfaces/left_shelf_{i}.jpg")
        g.box(f"{path}/Shelf{i}", (0, 0, height), (width, depth, .015), material, .0015)
        for side in (-1, 1):
            # The exposed narrow side of each shelf sits inside the steel lip.
            g.box(f"{path}/SideRail{i}_{int(side>0)}",
                  (side*(width/2+.0013), 0, height-.0035),
                  (.0024, depth, .018), "RackBoardEdge", .0005)
    top = decks[-1]+.008
    for i, (sx, sy) in enumerate((x,y) for x in (-1,1) for y in (-1,1)):
        # Each upright is an open angle with two 24 mm flanges.
        x, y = sx*(width/2+.0034), sy*(depth/2+.0026)
        profile = [(0,0),(.024,0),(.024,.0022),(.0022,.0022),
                   (.0022,.024),(0,.024)]
        verts = [(x-sx*u, y-sy*v, z) for z in (floor,top) for u,v in profile]
        triangles = [(0,1,3),(1,2,3),(0,3,5),(3,4,5)]
        faces = [tuple(reversed(t)) for t in triangles] + [tuple(j+6 for j in t) for t in triangles]
        for j in range(6):
            k = (j+1)%6
            faces.extend([(j,k,k+6),(j,k+6,j+6)])
        if sx*sy<0:
            faces = [tuple(reversed(t)) for t in faces]
        g.mesh(f"{path}/Post{i}", verts, faces, "RackPaint")
        for j, z in enumerate([floor+.014,*decks]):
            g.tube(f"{path}/Bolt{i}_{j}", (x, y-sy*.012, z),
                   (x+sx*.0015,y-sy*.012,z), .0031, "RackHardware")
    for i, side in enumerate((-1,1)):
        g.box(f"{path}/FloorRail{i}", (side*(width/2+.001),0,floor+.011),
              (.004,depth,.022),"RackPaint",.0008)
    g.box(path+"/BackFloorRail", (0,depth/2,floor+.010),
          (width,.004,.020),"RackPaint",.0007)
    # The side wall catches the shelf shadows in the opening shot.
    g.box(workshop+"/ShelfSideWall", (-1.743,.233,.49), (.044,.93,2.52), "Plaster", .001)


def stool_and_cable(g, workshop, floor):
    path = workshop + "/Stool"
    cx, cy, top = -1.061, -.452, -.400
    radius = .179
    g.material("StoolEdge", (.088, .048, .026), .73, 0, "./assets/surfaces/seat_ply.jpg")
    g.material("Stoolwood", (.40, .32, .22), .72, 0,
               "./assets/surfaces/seat_worn.jpg", "seat_worn")
    g.material("StoolScrew", (.17, .15, .12), .57, .58)
    # The veneer runs to the rounded rim. The dark ply is visible side-on.
    angles = np.arange(96)*2*np.pi/96
    circle = np.column_stack((np.cos(angles),np.sin(angles)))
    edge = [(radius-.003, top-.014), (radius,top-.0115),
            (radius,top-.0025), (radius-.002,top)]
    vertices = [(cx+r*x,cy+r*y,z) for r,z in edge for x,y in circle]
    faces = [(i*96+j,i*96+(j+1)%96,(i+1)*96+(j+1)%96,(i+1)*96+j)
             for i in range(len(edge)-1) for j in range(96)]
    edge_uv = [[(j/96,i/3),((j+1)/96,i/3),
                ((j+1)/96,(i+1)/3),(j/96,(i+1)/3)]
               for i in range(3) for j in range(96)]
    g.mesh(path+"/SeatEdge", vertices, faces, "StoolEdge", uv=edge_uv)
    top_ring = [(cx+x*(radius-.002),cy+y*(radius-.002),top) for x,y in circle]
    triangles = [(0,j+1,(j+1)%96+1) for j in range(96)]
    uv = np.vstack(([.5,.5],.5+.5*circle))
    g.mesh(path+"/Seat",[(cx,cy,top),*top_ring],triangles,"Stoolwood",uv=uv)
    bottom = [(x,y,top-.014) for x,y,_ in top_ring]
    g.mesh(path+"/SeatBottom", [(cx,cy,top-.014),*bottom],
           [tuple(reversed(t)) for t in triangles],"StoolEdge")
    for i,(x,y) in enumerate([(-1.1504,-.4612),(-1.0602,-.3587),
                              (-.9621,-.4416),(-1.0462,-.5458)]):
        g.tube(f"{path}/Screw{i}",(x,y,top-.0002),(x,y,top+.0004),.0034,"StoolScrew")
        g.box(f"{path}/Slot{i}",(x,y,top+.00045),(.0042,.0007,.00013),"ink",.00005,
              Rotation.from_euler("z",13+29*i,degrees=True))
    for i, angle in enumerate(np.arange(4)*np.pi/2+.38):
        dx,dy=np.cos(angle),np.sin(angle)
        t=np.linspace(0,1,25)
        r=.095+.092*(1-(1-t)**2.6)
        z=top-.014+(floor+.008-top+.014)*t
        g.wire(f"{path}/Leg{i}", np.c_[cx+r*dx,cy+r*dy,z],.0075,"ink")
        x,y=cx+r[-1]*dx,cy+r[-1]*dy
        g.tube(f"{path}/Foot{i}",(x,y,floor),(x,y,floor+.014),.0098,"dark")
    angle=np.linspace(0,2*np.pi,97)
    g.wire(path+"/Footrest",np.c_[cx+.184*np.cos(angle),cy+.184*np.sin(angle),
                                 np.full_like(angle,floor+.079)],.0037,"ink")
    t=np.linspace(0,1,171)
    a=t*2*np.pi*1.87+.37
    r=.89+.095*t+.025*np.sin(3*a+.4)
    g.wire(workshop+"/FloorCable",
           np.c_[-.727+.131*r*np.cos(a)+.013*np.sin(a),
                 -.258+.099*r*np.sin(a)+.041*np.cos(a),
                 floor+.0029+.0016*t+.0013*t*np.cos(2*a)**4],.0028,"ink")


def furnish(g, h, bases):
    finishes(g)
    P = "/World/Workshop"
    g.group(P)
    mats = {
        "ink": ((0.013, 0.015, 0.016), 0.54, 0.16),
        "steel": ((0.34, 0.36, 0.35), 0.34, 0.76),
        "frame": ((0.43, 0.44, 0.42), 0.44, 0.46),
        "tape": ((0.57, 0.46, 0.24), 0.67, 0.03),
        "crate": ((0.20, 0.048, 0.043), 0.5, 0.14),
        "white": ((0.72, 0.72, 0.68), 0.4, 0.12),
        "matEdge": ((0.127, 0.092, 0.052), 0.82, 0),
        "dark": ((0.034, 0.037, 0.034), 0.59, 0.14),
    }
    for m, (col, r, k) in mats.items():
        g.material(m, col, r, k)
    for name, tex, r, met in [
        ("Plaster", "wall", 0.96, 0),
        ("Concrete", "floor", 0.91, 0),
        ("Racking", "shelf", 0.81, 0.13),
    ]:
        g.material(name, (0.4, 0.4, 0.4), r, met, "./assets/surfaces/" + tex + ".jpg")

    g.material(
        "Green",
        (0.04, 0.08, 0.06),
        0.56,
        0,
        "./assets/surfaces/satin_laminate_color.png",
        "satin_laminate",
        coat=0.055,
        coat_roughness=0.43,
    )
    g.material(
        "Board",
        (0.47, 0.30, 0.16),
        0.41,
        0,
        "./assets/surfaces/chessboard_color.png",
        "chessboard",
        coat=0.17,
        coat_roughness=0.30,
    )
    g.material(
        "VeneerEdge",
        (0.26, 0.14, 0.06),
        0.50,
        0,
        "./assets/surfaces/board_edge.png",
        "endgrain",
    )
    def box(n, c, s, m, r=0.001, angle=0):
        return g.box(
            P + "/" + n,
            c,
            s,
            m,
            r,
            Rotation.from_euler("z", angle, degrees=True) if angle else None,
        )

    def pole(n, a, b, r, m):
        return g.tube(P + "/" + n, a, b, r, m)

    def wire(n, pts, r, m):
        return g.wire(P + "/" + n, pts, r, m)

    floor = -0.77
    box("Floor", (0, 0, floor - 0.015), (6, 5, 0.03), "Concrete")
    for i in range(-3, 4):
        box(
            "Plaster_" + str(i + 3),
            (i * 0.91 - 0.39, 0.725, 0.54),
            (0.908, 0.055, 2.65),
            "Plaster",
            0.001,
        )
    box("Baseboard", (0, 0.686, -0.68), (6, 0.045, 0.105), "Racking")
    # A green laminated worktop with a narrow exposed ply edge.
    box("Workbench/Edge", (0, 0.103, -0.0345), (1.24, 0.81, 0.027), "matEdge", 0.001)
    surface = box(
        "Workbench/GreenSurface",
        (0, 0.103, -0.018),
        (1.241, 0.811, 0.006),
        "Green",
        0.001,
    )
    # The cut edge has a satin finish. Wrapping the glossy tabletop normal
    # map onto the 6 mm lip creates a moving highlight only a pixel wide.
    g.material("LaminateCutEdge", (.036, .065, .049), .66, 0)
    points = np.asarray(surface.GetPointsAttr().Get())
    quads = np.asarray(surface.GetFaceVertexIndicesAttr().Get()).reshape(-1, 4)
    corners = points[quads]
    normals = np.cross(corners[:, 1]-corners[:, 0], corners[:, 3]-corners[:, 0])
    front = np.flatnonzero(normals[:, 1] < -.05*np.linalg.norm(normals, axis=1))
    binding = UsdShade.MaterialBindingAPI(surface)
    lip = binding.CreateMaterialBindSubset("FrontLip", front.tolist())
    UsdShade.MaterialBindingAPI.Apply(lip.GetPrim()).Bind(g.materials["LaminateCutEdge"])
    for i, x in enumerate([-0.548, 0.548]):
        for j, y in enumerate([-0.244, 0.442]):
            box(
                f"Workbench/Leg_{i}{j}",
                (x, y, -0.405),
                (0.045, 0.046, 0.73),
                "frame",
                0.0018,
            )
    for i, y in enumerate([-0.245, 0.444]):
        box("Workbench/Rail" + str(i), (0, y, -0.075), (1.14, 0.037, 0.073), "frame")
    box("Workbench/RearBrace", (0, 0.445, -0.601), (1.10, 0.036, 0.037), "frame")
    mount_bridge(g, P + "/Mount/Bridge")
    for i, (x, base_y, _) in enumerate(bases):
        offset_y = base_y + .202
        box(f"Mount/Plate{i}", (x, -0.198 + offset_y, -.005275), (.140, .150, .01555), "dark")
        for j, dx in enumerate([-0.052, 0.052]):
            box(
                f"Mount/Clamp{i}{j}",
                (x + dx, -0.284 + offset_y, -0.065),
                (0.022, 0.034, 0.108),
                "ink",
            )
        p = np.array(
            [
                [x, -0.225, 0.041],
                [x, -0.262, 0.028],
                [x, -0.294, 0.010],
                [x + 0.002, -0.317, 0.001],
                [x + 0.004, -0.328, -0.011],
                [x + 0.006, -0.332, -0.034],
                [x + 0.009, -0.328, -0.098],
                [x + 0.022, -0.331, -0.163],
                [x + 0.022, -0.322, -0.201],
                [x + 0.015, -0.298, -0.204],
                [x - 0.003, -0.275, -0.159],
                [x - 0.008, -0.25, -0.084],
            ]
        )
        p[:, 1] += offset_y
        from scipy.interpolate import make_interp_spline

        pole(f"Mount/Ferrule{i}", (x, -.251 + offset_y, .032), (x, -.262 + offset_y, .028), .0092, "steel")
        pole(f"Mount/StrainRelief{i}", (x, -.262 + offset_y, .028), (x, -.276 + offset_y, .0205), .0068, "ink")
        wire(
            f"Mount/SignalCable{i}",
            make_interp_spline(np.arange(len(p)), p, k=2)(
                np.linspace(0, len(p) - 1, 65)
            ),
            0.0045,
            "ink",
        )
    # Worn laminate fibres are in the colour and roughness maps.
    for j, an in enumerate([-25, 65]):
        box(
            "Workbench/Mark" + str(j),
            (-0.515, 0.258, -0.0144 + j * 0.0002),
            (0.065, 0.009, 0.00035),
            "tape",
            0.0001,
            an,
        )
    box(
        "Board/Substrate",
        (0, 0, -0.0076),
        (0.451, h + 0.001, 0.015),
        "VeneerEdge",
        0.002,
    )
    g.mesh(
        P + "/Board/PrintedWood",
        [
            [-0.225, -h / 2, 0.00005],
            [0.225, -h / 2, 0.00005],
            [0.225, h / 2, 0.00005],
            [-0.225, h / 2, 0.00005],
        ],
        [[0, 1, 2, 3]],
        "Board",
        None,
        [[0, 0], [1, 0], [1, 1], [0, 1]],
    )

    def shelving(name, x, y, w, d, levels, frame, deck):
        for i, (dx, dy) in enumerate(
            [(v, z) for v in (-w / 2, w / 2) for z in (-d / 2, d / 2)]
        ):
            top = levels[-1] + 0.052
            box(
                f"{name}/Post{i}",
                (x + dx, y + dy, (top + floor) / 2),
                (0.016, 0.017, top - floor),
                frame,
            )
        for i, z in enumerate(levels):
            box(f"{name}/Shelf{i}", (x, y, z), (w, d, 0.015), f"rackwood{i%4}" if name=="RightRack" else deck, 0.0018)
        for i, dx in enumerate([-w / 2, w / 2]):
            box(
                f"{name}/TopBar{i}",
                (x + dx, y, levels[-1] + 0.05),
                (0.016, d, 0.014),
                frame,
            )

    left_shelf(g, P, floor)
    shelving(
        "RightRack",
        1.047,
        0.529,
        0.55,
        0.285,
        [-0.59, -0.190, -0.024, 0.128],
        "rack_iron",
        "rackwood0",
    )
    for i in range(4):
        box(
            "RightRack/Drawer" + str(i),
            (1.047, 0.529, -0.535 + i * 0.095),
            (0.54, 0.277, 0.089),
            f"rackwood{i}",
            0.0025,
        )

    carton(g,P+"/ParcelOnShelf",1.091,.489,-.1825,.186,.148,.098)
    floor_carton(g,P+"/ParcelOnFloor")
    case=g.group(P+"/ToolCase",(.81455,-.19642,floor),Rotation.from_euler("z",-12.3,degrees=True))
    case_path=str(case.GetPath())
    from case_mesh import floor_tool_case
    floor_tool_case(g,case_path)

    stool_and_cable(g, P, floor)
    from workshop_props import scanned_props
    scanned_props(g, P)
    z = -.01495
    # Silver caliper, including both pairs of jaws.
    box(
        "Caliper/Beam",
        (-0.013, 0.35, z + 0.003),
        (0.184, 0.010, 0.005),
        "steel",
        0.001,
        -4,
    )
    box(
        "Caliper/Slide",
        (-0.049, 0.352, z + 0.006),
        (0.047, 0.024, 0.010),
        "frame",
        0.001,
        -4,
    )
    for j, x in enumerate([-0.091, -0.053]):
        # Ground steel jaws taper from their sliding blocks to a fine contact edge.
        for sign, length in [(-1, 0.032), (1, 0.016)]:
            root_y = 0.353 + 0.003 * j
            outline = np.array(
                [
                    [x - 0.004, root_y],
                    [x + 0.004, root_y],
                    [x + 0.004, root_y + sign * length],
                    [x + 0.0007, root_y + sign * (length - 0.003)],
                    [x - 0.004, root_y + sign * length * 0.45],
                ]
            )
            if sign < 0:
                outline = outline[::-1]
            points = [(xx, yy, z + zz) for zz in (0.001, 0.005) for xx, yy in outline]
            name = f"Caliper/Jaw{j}_{int(sign > 0)}"
            g.mesh(P + "/" + name + "Top", points, [list(range(5, 10))], "frame")
            g.mesh(
                P + "/" + name + "Edges",
                points,
                [[k, (k + 1) % 5, (k + 1) % 5 + 5, k + 5] for k in range(5)],
                "steel",
            )
    for k in range(56):
        x = -0.024 + 0.0019 * k
        y = 0.350 - 0.070 * (x + 0.013)
        box(
            f"Caliper/Scale{k}",
            (x, y + 0.0008, z + 0.00564),
            (0.00020, 0.0032 if k % 5 == 0 else 0.0017, 0.00010),
            "ink",
            0.00002,
            -4,
        )
    box("Caliper/DepthRod", (.094, .3426, z+.002), (.032, .0015, .0016), "steel", .00025, -4)
    for item,offset_y in (("Canister",.134),("Screwdriver",.140)):
        g.group(P+"/"+item,(-.024,offset_y,0))
    vessel(g,P+"/Canister",.964,.336,-.0165,.032,.086,"worn_orange")
    pole("Screwdriver/Grip",(1.049,.299,.003),(1.102,.29,.003),.008,"crate")
    pole("Screwdriver/Blade",(1.102,.29,.003),(1.16,.280,.003),.0022,"steel")
    g.material("OilLabel", (.5,.5,.5), .64, 0, "./assets/surfaces/oil_label.jpg")

    def label(n, x, y, z, w, h, mat):
        g.mesh(
            P + "/" + n,
            [
                (x - w / 2, y, z - h / 2),
                (x + w / 2, y, z - h / 2),
                (x + w / 2, y, z + h / 2),
                (x - w / 2, y, z + h / 2),
            ],
            [[0, 1, 2, 3]],
            mat,
            None,
            [[0, 0], [1, 0], [1, 1], [0, 1]],
        )

    pole("Canister/Band", (.964,.336,.043), (.964,.336,.054), .0323, "ink")
    label("Canister/Brand", .964, .3035, .018, 0.034, 0.012, "OilLabel")
    for i, (x, base_y, _) in enumerate(bases):
        offset_y = base_y + .202
        for j, (dx, dy) in enumerate(
            [(x, y) for x in (-0.050, 0.050) for y in (-0.060, 0.060)]
        ):
            pole(
                f"Mount/Bolt{i}_{j}",
                (x + dx, -0.198 + dy, 0.001),
                (x + dx, -0.198 + dy, 0.003),
                0.003,
                "steel",
            )


def reference_surfaces(g):
    # The opening shot shows the rear-right corner at the end of the cabinet.
    # Close the room there so the cabinet can occlude indirect wall light.
    g.box("/World/Workshop/CabinetSideWall", (1.768, .193, .54),
          (.044, 1.01, 2.65), "Plaster")
    g.box("/World/Workshop/CabinetSkirting", (1.728, .189, -.68),
          (.036, 1.01, .105), "Racking")
    # The Go2 is the only model bound to the supplier's "gray" finish.
    g.material("gray", (.202, .227, .256), .45, .12)
    # Retain the scan's normal, metallic and roughness maps on the cup only.
    cup_path = "/World/Workshop/Cup/Model"
    silver = UsdShade.Material.Define(g.s, "/World/Materials/PenCupSilver")
    silver.GetPrim().GetReferences().AddInternalReference(cup_path + "/_materials/stationery_supplies")
    image = UsdShade.Shader.Get(g.s, silver.GetPath().AppendChild("Image_Texture"))
    silver_bias = (.205, .220, .225)
    image.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((.62, .62, .62, 1))
    image.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((*silver_bias, 0))
    # Use the scan's USD Preview Surface for this local override. Keeping its
    # normal/roughness connections also avoids compiling a second MaterialX graph.
    silver.GetPrim().GetAttribute("outputs:mtlx:surface").SetConnections([])
    mesh = g.s.GetPrimAtPath(cup_path + "/stationery_supplies_pencilcup/Cylinder_100")
    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(silver)

    def tint(material, rgb):
        image = UsdShade.Shader.Get(g.s, "/World/Materials/" + material + "/Image")
        image.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((*rgb, 1))

    # The mounting bridge and feet are coated steel with little specular return.
    g.material("MountCoating", (.023, .025, .024), .69, .08)
    for name in ("Bridge", "Plate0", "Plate1"):
        prim = g.s.GetPrimAtPath("/World/Workshop/Mount/" + name)
        g.bind(prim, "MountCoating")
    board = UsdShade.Shader.Get(g.s, "/World/Materials/Board/Surface")
    board.GetInput("clearcoat").Set(.09)
    # Restore the low grain values while keeping the pale fibres legible.
    grain = UsdShade.Shader.Get(g.s, "/World/Materials/Board/Image")
    grain.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((1.045, 1.045, 1.06, 1))
    grain.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-.022, -.018, -.011, 0))
    tint("Concrete", (1.25, 1.27, 1.44))
    for i in range(4):
        tint(f"rackwood{i}", (.83, .77, .70))


def lights(g, reference=False):
    s = g.s
    d = UsdLux.DomeLight.Define(s, "/World/Light/DiffuseSky")
    d.CreateIntensityAttr(70 if reference else 60)
    d.CreateColorAttr((0.91, 0.94, 1.0))

    def rect(name, pos, at, size, intensity, color, diffuse=1.0):
        p = UsdLux.RectLight.Define(s, "/World/Light/" + name)
        p.CreateWidthAttr(size[0])
        p.CreateHeightAttr(size[1])
        p.CreateIntensityAttr(intensity)
        p.CreateColorAttr(color)
        p.CreateDiffuseAttr(diffuse)
        forward = np.array(at) - pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        g.transform(
            p,
            pos,
            Rotation.from_matrix(
                np.stack([right, np.cross(right, forward), -forward], axis=1)
            ),
        )
        return p

    if reference:
        # The pen cup and rear shelf posts both cast a soft shadow towards +X.
        # Light from the front opening also reaches the floor below the bench.
        window = rect("FrontWindow", np.array([-1.29, -2.16, 1.48]), [-.65, .30, -.16],
                      (1.06, 1.01), 5210, (1, .965, .895))
        UsdLux.ShapingAPI.Apply(window.GetPrim()).CreateShapingFocusAttr(2.0)
    rect(
        "LargeWindow",
        np.array([.30, -.51, 1.68]),
        [0, 0.06, -0.1],
        (.77, .35),
        1675 if reference else 7900,
        (1, 0.975, 0.925),
    )
    rect(
        "OverheadBounce",
        np.array([0.89, 0.16, 2.05]),
        [0, 0.05, -0.2],
        (1.35, 1.10),
        500 if reference else 505,
        (0.90, 0.94, 1.0),
    )
    rect(
        "WindowReturn",
        np.array([-0.78, 0.37, 1.08]),
        [-0.19, 0.25, 0.0],
        (0.35, 0.88),
        140 if reference else 420,
        (0.91, 0.95, 1.0),
    )
    # Broad front wall return reads in the metal sleeves and shaded shelf faces.
    rect("FrontWallReturn",np.array([-.68,-1.60,.82]),[.1,.15,.12],(.85,1.25),170 if reference else 300,(1,.965,.905),.48)
    # Light returned by the pale wall in front of the open shelving reaches the
    # carton under the second shelf and brings out the soft sides of each crease.
    rect("ShelfWallReturn",np.array([1.32,-1.11,.29]),[1.05,.36,-.22],
         (1.04,1.22),245 if reference else 745,(1,.955,.885),.86)
