"""
Usage:
    python find_largest_cylinder_step.py part.STEP [output.csv]

- Reads STEP file and finds largest cylinder (or circle). 
- Writes result as CSV:
    First line = categories
    Second line = values

Note: please center the co-ordinates of part around the geometric centroid first! I could not be bothered so I just added a hacky offset for my circle parts and Rl1a
"""

import sys, re, math, csv, os

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

def num_in(token):
    if not token: return None
    m = re.search(r"[-+]?\d*\.?\d+(?:[Ee][-\+]?\d+)?", token)
    return float(m.group(0)) if m else None

def get_point(entities, eid):
    ent = entities.get(eid)
    if not ent or ent[0] != "CARTESIAN_POINT": return (0,0,0)
    parts = parse_list(ent[1])
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-\+]?\d+)?", parts[1] if len(parts)>=2 else "")
    return tuple(map(float, nums[:3])) if len(nums)>=3 else (0,0,0)

def get_direction(entities, eid):
    ent = entities.get(eid)
    if not ent or ent[0] != "DIRECTION": return (0,0,1)
    parts = parse_list(ent[1])
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-\+]?\d+)?", parts[1] if len(parts)>=2 else "")
    if len(nums)>=3:
        x,y,z = map(float, nums[:3])
        n = (x*x+y*y+z*z)**0.5 or 1.0
        return (x/n, y/n, z/n)
    return (0,0,1)

def get_axis2(entities, eid):
    ent = entities.get(eid)
    if not ent or ent[0] != "AXIS2_PLACEMENT_3D":
        return {"origin":(0,0,0),"axis":(0,0,1)}
    parts = parse_list(ent[1])
    pt_id = int(parts[1][1:]) if len(parts)>1 and parts[1].startswith("#") else None
    axis_id = int(parts[2][1:]) if len(parts)>2 and parts[2].startswith("#") else None
    return {
        "origin": get_point(entities, pt_id) if pt_id else (0,0,0),
        "axis": get_direction(entities, axis_id) if axis_id else (0,0,1)
    }

def step_find_all_cylinders(stepfile, outfile=None):
    text = open(stepfile,"r",errors="ignore").read()
    entity_re = re.compile(r"#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*?)\)\s*;", re.IGNORECASE | re.DOTALL)
    entities = {}
    for m in entity_re.finditer(text):
        eid = int(m.group(1)); etype = m.group(2).upper(); args = m.group(3).strip()
        entities[eid] = (etype, args)

    cyls, circles = [], []

    for eid,(etype,args) in entities.items():
        if etype == "CYLINDRICAL_SURFACE":
            parts = parse_list(args)
            ax_id = int(parts[1][1:]) if len(parts)>=2 and parts[1].startswith("#") else None
            radius = num_in(parts[2] if len(parts)>=3 else None)
            if radius is not None:
                ax = get_axis2(entities, ax_id)
                cyls.append({
                    "Type":"cylinder","EntityID":eid,"Radius":radius,
                    "AxisOriginX":ax["origin"][0],"AxisOriginY":ax["origin"][1],"AxisOriginZ":ax["origin"][2],
                    "AxisDirX":ax["axis"][0],"AxisDirY":ax["axis"][1],"AxisDirZ":ax["axis"][2]
                })
        elif etype == "CIRCLE":
            parts = parse_list(args)
            ax_id = int(parts[1][1:]) if len(parts)>=2 and parts[1].startswith("#") else None
            radius = num_in(parts[2] if len(parts)>=3 else None)
            if radius is not None:
                ax = get_axis2(entities, ax_id)
                circles.append({
                    "Type":"circle_edge","EntityID":eid,"Radius":radius,
                    "AxisOriginX":ax["origin"][0],"AxisOriginY":ax["origin"][1],"AxisOriginZ":ax["origin"][2],
                    "AxisDirX":ax["axis"][0],"AxisDirY":ax["axis"][1],"AxisDirZ":ax["axis"][2]
                })

    rows = sorted(cyls, key=lambda r: r["Radius"], reverse=True)
    if not rows:
        rows = sorted(circles, key=lambda r: r["Radius"], reverse=True)
    if not rows:
        rows = [{"Type":"none"}]

    if not outfile:
        base = os.path.splitext(stepfile)[0]
        outfile = f"{base}_cylinders.csv"

    # unify fieldnames across cases
    fieldnames = ["Type","EntityID","Radius","AxisOriginX","AxisOriginY","AxisOriginZ","AxisDirX","AxisDirY","AxisDirZ"]
    with open(outfile,"w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Result saved in: {outfile}")
