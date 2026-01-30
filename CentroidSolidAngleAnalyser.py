import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from PoseFinder import PoseFinder
from matplotlib.path import Path
import matplotlib.pyplot as plt

class CentroidSolidAngleAnalyser(PoseFinder):
    def __init__(self, poses = None, convex_hull_obj_file=None, obj_file=None, pose_types=None, pose_cylinder_radius=None, pose_cylinder_axis_origin=None, pose_cylinder_axis_direction=None, tolerance=1e-5):
        super().__init__(convex_hull_obj_file, obj_file, tolerance)
        self.poses = poses
        self.pose_types = pose_types
        self.pose_cylinder_radius = pose_cylinder_radius
        self.pose_cylinder_axis_origin = pose_cylinder_axis_origin
        self.pose_cylinder_axis_direction = pose_cylinder_axis_direction
        self.csa_values = []
        self.stability_values = []
        self.crsa_values = []

        self.csa_scores = []
        self.stability_scores = []
        self.crsa_scores = []

    def solid_angle_triangle(self, a, b, c, p):
        a = a - p
        b = b - p
        c = c - p

        la = np.linalg.norm(a)
        lb = np.linalg.norm(b)
        lc = np.linalg.norm(c)

        num = abs(np.linalg.det(np.stack([a, b, c])))
        den = (
            la * lb * lc
            + np.dot(a, b) * lc
            + np.dot(b, c) * la
            + np.dot(c, a) * lb
        )

        return 2.0 * np.arctan2(num, den)

    def solid_angle_polygon(self,hull_points, point):
        """
        Solid angle (steradians) of a planar polygon seen from `point`.

        Parameters
        ----------
        hull_points : (N+1, 3) array
            Closed polygon (last point == first point)
        point : (3,) array
            Observation point (COM or critical point)

        Returns
        -------
        omega : float
            Solid angle in steradians
        """
        # hull_points: (N+1,3), closed polygon on plane
        verts = hull_points[:-1]
        v0 = verts[0]

        omega = 0.0
        for i in range(1, len(verts) - 1):
            omega += self.solid_angle_triangle(v0, verts[i], verts[i+1], point)
        
        return omega
    
    
    import numpy as np


    def cylinder_side_solid_angle_continuous(self,center, axis, R, L, com,
                                            g_dir=np.array([0.0, 0.0, -1.0]),
                                            n_theta=720, n_z=240):
        """
        Solid angle of the *lower half* of the lateral surface of a cylinder
        as seen from `com`, computed by continuous surface integral.

        Assumes the cylinder is resting on a plane under gravity direction g_dir.
        """
        a = np.linalg.norm(np.asarray(axis, float))
        print('Axis direction:', a)
        c0 = np.asarray(center, float)
        p = np.asarray(com, float)
        g = np.linalg.norm(np.asarray(g_dir, float))  # down

        # Build radial basis e1,e2 in plane perpendicular to axis
        # e1 points toward "down" projected into the cross-section plane.
        g_perp = g - np.dot(g, a) * a
        if np.linalg.norm(g_perp) < 1e-9:
            raise ValueError("Axis parallel to gravity; 'side resting' not defined.")
        e1 = np.linalg.norm((g_perp))          # points toward down in cross-section
        e2 = np.linalg.norm((np.cross(a, e1))) # completes right-handed basis

        # Parameter grids
        thetas = np.linspace(-np.pi/2, np.pi/2, n_theta)  # lower half: radial dot g >= 0
        zs = np.linspace(-L/2, L/2, n_z)

        dtheta = thetas[1] - thetas[0]
        dz = zs[1] - zs[0]

        Omega = 0.0
        for z in zs:
            for th in thetas:
                n = np.cos(th) * e1 + np.sin(th) * e2          # outward normal on lateral surface
                x = c0 + R * n + z * a                         # point on surface
                r = x - p
                rn = np.dot(r, n)
                rnorm = np.linalg.norm(r)
                # dA = R dtheta dz
                Omega += (rn / (rnorm**3)) * (R * dtheta * dz)

        # Optional: "height" to tangent plane at lowest point (for CSA score Omega/h)
        up = -g
        p0 = c0 - R * e1  # lowest point on cylinder (along down direction)
        h = abs(np.dot(p - p0, up))

        return Omega, h, Omega / (h + 1e-12)

    
    def compute_scores(self):
        """
        calculates stability scores based on centroid solid angle method, stability score method and critical solid angle method from:
        centroid solid angle from: B. K. A. NGOI , L. E. N. LIM & S. S. G. LEE (1995) Analysing the natural resting aspects of a complex part, International Journal of Production Research, 33:11, 3163-3172,
        stability score method from:  Chua, P. S. K., and Tay, M. L. (August 1, 1998). "Modelling the Natural Resting Aspect of Small Regular Shaped Parts." ASME. J. Manuf. Sci. Eng. August 1998; 120(3): 540–546.
        Critical solid angle method from: Ngoi, K.A., Lye, S.W. & Chen, J. Analysing the natural resting aspect of a prism on a hard surface for automated assembly. Int J Adv Manuf Technol 11, 406–412 (1996).


        alpha_tilt – CCW tilt (deg) around X-axis (front-back)
        """

        non_symmetrical_csa_values_sum = 0.0
        non_symmetrical_stability_values_sum = 0.0
        non_symmetrical_crsa_values_sum = 0.0
        
        seen_indicies = []

        for index, face_id, edge_id, quat in self.poses:

            # Step 1: Apply pose rotation
            rotated_mesh = self.mesh.copy()
            rotation = R.from_quat(quat)
            T_pose = np.eye(4)
            T_pose[:3, :3] = rotation.as_matrix()
            rotated_mesh.apply_transform(T_pose)

            # Step 2: Identify contact vertices at bottom plane (min-Z)
            verts = rotated_mesh.vertices
            z_coords = verts[:, 2]
            min_z = np.min(z_coords)
            contact_idx = np.where(np.abs(z_coords - min_z) < self.tolerance)[0]
            contact_vertices = verts[contact_idx]

            if len(contact_vertices) < 3:
                self.csa_values.append(0.0)

            # Step 3: Tilt the system so that the slide plane becomes horizontal
            #alpha = np.radians(alpha_tilt)
            #beta = np.radians(beta_tilt)

            R_tilt = R.from_euler('xy', [0, 0]).as_matrix()  # inverse tilt
            contact_tilted = contact_vertices @ R_tilt.T
            com_tilted = rotated_mesh.center_mass @ R_tilt.T

            # contact_tilted: (M,3)
            contact_xy = np.unique(contact_tilted[:, :2], axis=0)

            if contact_xy.shape[0] < 3:
                self.csa_values.append(0.0)

            # convex hull in 2D
            hull = ConvexHull(contact_xy)
            hull_xy = contact_xy[hull.vertices]        # (N,2)

            # lift polygon back to 3D (resting plane)
            hull_3d = np.column_stack([
                hull_xy[:, 0],
                hull_xy[:, 1],
                np.full(len(hull_xy), min_z)
            ])

            # close polygon
            hull_3d = np.vstack([hull_3d, hull_3d[0]])  # (N+1,3)

            # compute surface area of the polygonal face
            # points: (N,2), ordered, not closed
            #if self.pose_types[index] == 3:
            #    y_min, y_max = np.min(hull_3d[:, 1]), np.max(hull_3d[:, 1])
            #    length = y_max - y_min
            #    area = np.pi * self.pose_cylinder_radius[index] * length
            #else:

            x = hull_3d[:, 0]
            y = hull_3d[:, 1]
            area =  0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

            #if self.pose_types[index] == 3:
            #    print('Computing solid angle of cylinder side face')
            #    y_min, y_max = np.min(hull_3d[:, 1]), np.max(hull_3d[:, 1])
            #    y_center = 0.5 * (y_min + y_max)
            #    Q = self.cylinder_side_solid_angle_continuous([self.pose_cylinder_axis_origin[index][0], y_center, self.pose_cylinder_axis_origin[index][2]],
            #                                                   self.pose_cylinder_axis_direction[index],
            #                                                   self.pose_cylinder_radius[index],
            #                                                   y_max - y_min,
            #                                                   com_tilted,
            #                                                   g_dir=np.array([0,0,-1]))  # if world z is up)
            #else:
                
                
            Q = self.solid_angle_polygon(hull_3d, com_tilted)

            # critical solid angle (Qcrit)
            delta_sum = 0.0

            N = len(hull_3d) - 1  # last point duplicates the first

            for i in range(N):
                v0 = hull_3d[i]
                v1 = hull_3d[i + 1]

                # midpoint of the edge
                edge_mid = 0.5 * (v0 + v1)

                z_crit = len(com_tilted - edge_mid) # distance from COM to edge midpoint in same plane projection
                print(z_crit)
                Q_crit = self.solid_angle_polygon(hull_3d, [com_tilted[0],com_tilted[1],z_crit]) # compute tipping solid angle for one edge

                delta = Q - Q_crit

                delta_sum += max(0.0, delta)
                '''
                plt.figure()
                plt.plot(hull_3d[:, 0], hull_3d[:, 1], 'k--', lw=1, label='Support Polygon')
                plt.plot(edge_mid[0], edge_mid[1], 'go', label='Center of Mass')
                plt.gca().set_aspect('equal')
                plt.legend()
                plt.text(0.95, 0.95, 'Q = {:.4f} sr\nStability Area = {:.4f} \nQcrit Sum = {:.4f} sr\n'.format(Q, area, delta_sum), fontsize=8, transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                plt.savefig(f"csa_check_{index:03d}.png", dpi=150)
                plt.close()
                '''

            # height (h)
            h = abs(com_tilted[2] - min_z)

            ''' Debug plot
            plt.figure()
            plt.plot(hull_3d[:, 0], hull_3d[:, 1], 'k--', lw=1, label='Support Polygon')
            plt.plot(com_tilted[0], com_tilted[1], 'go', label='Center of Mass')
            plt.gca().set_aspect('equal')
            plt.legend()
            plt.text(0.95, 0.95, 'Q = {:.4f} sr\nStability Area = {:.4f} \nQcrit Sum = {:.4f} sr'.format(Q, area, delta_sum), fontsize=8, transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.savefig(f"csa_check_{index:03d}.png", dpi=150)
            plt.close()
            '''

            self.csa_values.append(Q / h)  # compute centroid solid angle for one pose
            self.stability_values.append(area / h)  # compute stability score for one pose
            self.crsa_values.append(delta_sum / h)  # compute critical solid angle score for one pose

            if index not in seen_indicies:
                seen_indicies.append(index)
                non_symmetrical_csa_values_sum = Q / h + non_symmetrical_csa_values_sum
                non_symmetrical_stability_values_sum = area / h + non_symmetrical_stability_values_sum
                non_symmetrical_crsa_values_sum = delta_sum / h + non_symmetrical_crsa_values_sum
            else:
                continue

        self.csa_scores = (np.array(self.csa_values) / non_symmetrical_csa_values_sum) * 100
        self.stability_scores = (np.array(self.stability_values) / non_symmetrical_stability_values_sum)*100
        self.crsa_scores = (np.array(self.crsa_values) / non_symmetrical_crsa_values_sum)*100