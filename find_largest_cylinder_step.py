#!/usr/bin/env python3
"""
find_largest_cylinder_step.py

Usage:
    python find_largest_cylinder_step.py /path/to/file.STEP

Funktion:
- Lädt eine STEP-Datei (AP203/AP214).
- Sucht alle CYLINDRICAL_SURFACE und CIRCLE Entitäten.
- Gibt den größten gefundenen Radius aus und die Achse (Ursprung + Richtung).

Hinweis:
- Werte sind in den Einheiten der STEP-Datei (meist mm).
- Funktioniert ohne OpenCascade, nur Text-Parsing.
"""

import sys, re, math, json

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

def main(stepfile):
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
                cyls.append({"eid":eid,"radius":radius,"axis_origin":ax["origin"],"axis_dir":ax["axis"]})
        elif etype == "CIRCLE":
            parts = parse_list(args)
            ax_id = int(parts[1][1:]) if len(parts)>=2 and parts[1].startswith("#") else None
            radius = num_in(parts[2] if len(parts)>=3 else None)
            if radius is not None:
                ax = get_axis2(entities, ax_id)
                circles.append({"eid":eid,"radius":radius,"center":ax["origin"],"normal":ax["axis"]})

    result = {}
    if cyls:
        largest = max(cyls, key=lambda c: c["radius"])
        result = {"type":"cylinder", **largest}
    elif circles:
        largest = max(circles, key=lambda c: c["radius"])
        result = {"type":"circle_edge", **largest}
    else:
        result = {"type":"none"}

    print(json.dumps({"status":"ok","result":result}, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_largest_cylinder_step.py part.STEP", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
