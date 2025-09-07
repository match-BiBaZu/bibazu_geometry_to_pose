#!/usr/bin/env python3
"""
check_outer_symmetry_step.py

Goal:
- Find the largest cylindrical surface (radius R0) and its axis.
- Look for geometry whose radial extent goes *outside* that base circle and test if these outer features
  are rotationally symmetric around the axis.

Heuristic (no CAD kernel required):
- Use CYLINDRICAL_SURFACE to define axis and R0 (largest radius).
- Consider as "outer" candidates:
  1) CIRCLE entities: compute center distance r_c to axis. If r_c + circle_radius > R0 + eps → outer.
  2) LINE entities whose direction is ~parallel to axis: take their base point distance r_p to axis.
     If r_p > R0 + eps → outer.
- Project candidate anchor points onto a plane ⟂ axis and compute polar angles around axis.
- Detect k-fold rotational symmetry (k ∈ {2..12}) by checking if all angles fall close (±tol) to a lattice of 360/k.

Limitations:
- Heuristic; does not analyze arbitrary faces or freeform edges.
- Works best if protrusions are represented by circular/linear geometry (holes, bosses, slots with circular edges,
  ribs with vertical lines, etc.).
- Units: same as STEP file (often mm).

Usage:
    python check_outer_symmetry_step.py part.STEP

Output: JSON with symmetry verdict and detected k (if any).
"""
import sys, re, math, json, os
from typing import List, Tuple, Dict

def parse_list(s: str):
    out, buf, depth = [], "", 0
    for ch in s:
        if ch == '(':
            depth += 1; buf += ch
        elif ch == ')':
            depth -= 1; buf += ch
        elif ch == ',' and depth == 0:
            out.append(buf.strip()); buf = ""
        else:
            buf += ch
    if buf.strip():
        out.append(buf.strip())
    return out

num_re = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-\+]?\d+)?")

def num_in(tok):
    m = num_re.search(tok or "")
    return float(m.group(0)) if m else None

def load_entities(path):
    txt = open(path, "r", errors="ignore").read()
    ent_re = re.compile(r"#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*?)\)\s*;", re.IGNORECASE | re.DOTALL)
    ents={}
    for m in ent_re.finditer(txt):
        ents[int(m.group(1))]=(m.group(2).upper(), m.group(3).strip())
    return ents

def get_point(ents, pid):
    e = ents.get(pid)
    if not e or e[0]!="CARTESIAN_POINT": return None
    parts = parse_list(e[1])
    nums = [float(x) for x in num_re.findall(parts[1] if len(parts)>=2 else "")[:3]]
    return tuple(nums) if len(nums)==3 else None

def get_dir(ents, did):
    e = ents.get(did)
    if not e or e[0]!="DIRECTION": return None
    parts = parse_list(e[1])
    nums = [float(x) for x in num_re.findall(parts[1] if len(parts)>=2 else "")[:3]]
    if len(nums)!=3: return None
    x,y,z = nums; n = math.sqrt(x*x+y*y+z*z) or 1.0
    return (x/n, y/n, z/n)

def get_axis2(ents, axid):
    e = ents.get(axid)
    if not e or e[0]!="AXIS2_PLACEMENT_3D": return None
    parts = parse_list(e[1])
    pt = get_point(ents, int(parts[1][1:])) if len(parts)>1 and parts[1].startswith("#") else (0.0,0.0,0.0)
    dz = get_dir(ents, int(parts[2][1:])) if len(parts)>2 and parts[2].startswith("#") else (0.0,0.0,1.0)
    dx = get_dir(ents, int(parts[3][1:])) if len(parts)>3 and parts[3].startswith("#") else None
    return {"origin": pt, "axis": dz, "ref": dx}

def vector_sub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def norm(a): return math.sqrt(max(0.0, dot(a,a)))
def normalized(a):
    n = norm(a)
    return (a[0]/n, a[1]/n, a[2]/n) if n>0 else (0.0,0.0,0.0)

def dist_point_to_axis(p, a_origin, a_dir):
    # distance from point p to infinite line a_origin + t*a_dir
    v = vector_sub(p, a_origin)
    # component orthogonal to a_dir
    a_dir = normalized(a_dir)
    proj = dot(v, a_dir)
    perp = vector_sub(v, (a_dir[0]*proj, a_dir[1]*proj, a_dir[2]*proj))
    return norm(perp)

def angle_on_plane(p, a_origin, ax_dir, ref_dir=None):
    # compute polar angle of the projection of p onto the plane ⟂ ax_dir, relative to ref_dir (or arbitrary)
    ax = normalized(ax_dir)
    # pick ref basis (u,v) on plane
    if ref_dir is None or abs(abs(dot(ax, normalized(ref_dir))) - 1.0) < 1e-6:
        # choose any orthonormal basis
        tmp = (1.0,0.0,0.0) if abs(ax[0]) < 0.9 else (0.0,1.0,0.0)
        u = normalized(cross(ax, tmp))
    else:
        u = normalized(cross(ax, cross(ref_dir, ax)))
        if norm(u) < 1e-9:
            tmp = (1.0,0.0,0.0) if abs(ax[0]) < 0.9 else (0.0,1.0,0.0)
            u = normalized(cross(ax, tmp))
    v = cross(ax, u)
    w = vector_sub(p, a_origin)
    x = dot(w, u); y = dot(w, v)
    ang = math.degrees(math.atan2(y, x)) % 360.0
    return ang

def detect_k_fold(angles: List[float], tol_deg=5.0, k_max=12):
    if not angles: return None
    angles = sorted(a % 360.0 for a in angles)
    # Try k from high to low preference (higher-fold symmetry first)
    for k in range(k_max, 1, -1):
        step = 360.0 / k
        # test all possible phase offsets using the first angle as reference
        ok = True
        for a in angles:
            mod = (a % step)
            delta = min(mod, step - mod)
            if delta > tol_deg:
                ok = False; break
        if ok:
            return k
    return None

def main(path, tol_axis_deg=3.0, eps=1e-6):
    ents = load_entities(path)

    # Find largest cylinder
    R0, axis_origin, axis_dir = None, None, None
    for eid,(t,args) in ents.items():
        if t == "CYLINDRICAL_SURFACE":
            parts = parse_list(args)
            axid = int(parts[1][1:]) if len(parts)>=2 and parts[1].startswith("#") else None
            radius = num_in(parts[2] if len(parts)>=3 else None)
            if axid and radius is not None:
                ax = get_axis2(ents, axid)
                if ax and (R0 is None or radius > R0):
                    R0, axis_origin, axis_dir = radius, ax["origin"], ax["axis"]

    if R0 is None or axis_dir is None:
        print(json.dumps({"status":"no_axis"}, indent=2)); return

    # Collect outer features (anchor points & angles)
    outer_points = []  # points used to measure angle
    outer_descr = []   # descriptions for reporting

    # Circles
    for eid,(t,args) in ents.items():
        if t == "CIRCLE":
            parts = parse_list(args)
            axid = int(parts[1][1:]) if len(parts)>=2 and parts[1].startswith("#") else None
            radius = num_in(parts[2] if len(parts)>=3 else None)
            if axid and radius is not None:
                ax = get_axis2(ents, axid)
                center = ax["origin"] if ax else None
                if center:
                    rc = dist_point_to_axis(center, axis_origin, axis_dir)
                    # Effective max radial extent ~ center offset + own circle radius if plane ⟂ axis
                    # We don't check circle plane normal; approximate upper bound:
                    radial_extent = rc + radius
                    if radial_extent > R0 + 1e-6:
                        outer_points.append(center)
                        outer_descr.append(("CIRCLE", eid, radial_extent))

    # Lines with direction ~parallel to axis: use their base point
    for eid,(t,args) in ents.items():
        if t == "LINE":
            parts = parse_list(args)
            if len(parts)>=2 and parts[0].startswith("#") and parts[1].startswith("#"):
                p0 = get_point(ents, int(parts[0][1:]))
                vec = ents.get(int(parts[1][1:]))
                if p0 and vec and vec[0]=="VECTOR":
                    vparts = parse_list(vec[1])
                    d = get_dir(ents, int(vparts[0][1:])) if len(vparts)>=1 and vparts[0].startswith("#") else None
                    if d:
                        # check parallelism with axis
                        cosang = abs(dot(normalized(d), normalized(axis_dir)))
                        ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
                        if ang <= tol_axis_deg:  # ~parallel
                            rp = dist_point_to_axis(p0, axis_origin, axis_dir)
                            if rp > R0 + 1e-6:
                                outer_points.append(p0)
                                outer_descr.append(("LINE", eid, rp))

    # Compute angles of outer points
    angles = [angle_on_plane(p, axis_origin, axis_dir) for p in outer_points]

    k = detect_k_fold(angles, tol_deg=5.0, k_max=12)
    out = {
        "status": "ok",
        "base_radius": R0,
        "axis_origin": axis_origin,
        "axis_dir": axis_dir,
        "outer_feature_count": len(outer_points),
        "detected_k_fold": k,
        "is_outer_rotationally_symmetric": bool(k) if outer_points else True,  # no outer features => vacuously symmetric
        "notes": "Heuristic based on CIRCLE centers and vertical LINE anchors; freeform/planar features not fully checked."
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_outer_symmetry_step.py part.STEP", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
