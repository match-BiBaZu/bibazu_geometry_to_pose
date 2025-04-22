import numpy as np
from scipy.spatial.transform import Rotation as R

# Initial camera rotation: 45° X then 45° Z
r_initial = R.from_euler('xz', [30, 45], degrees=True)

# Stand rotated 19° around X
r_stand = R.from_euler('x', 21, degrees=True)

# Desired camera rotation to compensate
r_target = r_stand.inv() * r_initial

# Convert to axis-angle
axis_angle = r_target.as_rotvec()
print('Axis-angle:', axis_angle)
print("Axis:", axis_angle / np.linalg.norm(axis_angle))
print("Angle (deg):", np.rad2deg(np.linalg.norm(axis_angle)))

