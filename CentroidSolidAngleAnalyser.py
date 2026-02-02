import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from PoseEliminator import PoseEliminator
from matplotlib.path import Path
import matplotlib.pyplot as plt

class CentroidSolidAngleAnalyser(PoseEliminator):
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
    
    def critical_solid_angle(self,hull_3d, com_tilted, Q, x_edge_verticies):
        # critical solid angle (Qcrit)
        delta_sum = 0.0

        N = len(hull_3d) - 1  # last point duplicates the first

        #x_edge = hull_3d[:, 0].min()
        #i_back_edge = np.where(np.isin(hull_idx_in_verts, edge_idx))[0]

        i_back_edge = [i for i, v in enumerate(hull_3d[:-1,:2]) if any(np.all(v == e) for e in x_edge_verticies[:,:2])]

        for i in range(N):
            if i not in i_back_edge:# skip edge poses as they dont allow tipping
                v0 = hull_3d[i]
                v1 = hull_3d[i + 1]

                # midpoint of the edge
                edge_mid = 0.5 * (v0 + v1)

                z_crit = np.linalg.norm(com_tilted - edge_mid) # distance from COM to edge midpoint in same plane projection
                print(z_crit)
                Q_crit = self.solid_angle_polygon(hull_3d, [com_tilted[0],com_tilted[1],z_crit]) # compute tipping solid angle for one edge

                delta = Q - Q_crit
                
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
            else:
                delta = Q # maximise non tipping poses Q crit is 0 as the edge poses dont tip

            delta_sum += max(0.0, delta)

        return delta_sum
    
    def fix_contact_edge_alignment_axis(
        self,
        contact_polygon,
        edge_origin,
        edge_direction,
        direction='z',
        target_val=0.0,
        tolerance=1e-6,
    ):
        """
        Snaps vertex coordinates of a contact polygon to a plane if the edge they're on
        is facing toward the intersecting direction (e.g. -x or -z), and is near the
        infinite edge line defined by (edge_origin, edge_direction).

        Parameters
        ----------
        contact_polygon : (N, 3) np.ndarray
            The polygon vertices in 3D (should be ordered/cyclic).
        edge_origin : (3,) np.ndarray
            A point on the shared edge line (e.g., [min_x, 0, min_z]).
        edge_direction : (3,) np.ndarray
            Unit direction vector of the edge (e.g., [0, 1, 0]).
        direction : str
            Which axis to check normal against: 'x' or 'z'.
        target_val : float
            Value to assign to the target axis (typically min_x or min_z).
        tolerance : float
            Max distance from the edge axis to snap a point.

        Returns
        -------
        corrected : (N, 3) np.ndarray
            Adjusted polygon vertices.
        """
        corrected = contact_polygon.copy()
        N = len(corrected)

        axis_map = {'z': (1, 2), 'x': (0, 1)}  # 2D projection indices
        target_idx = {'z': 2, 'x': 0}          # axis to overwrite

        if direction not in axis_map:
            raise ValueError("direction must be 'x' or 'z'")

        i1, i2 = axis_map[direction]
        t_idx = target_idx[direction]

        for i in range(N):
            v0 = corrected[i]
            v1 = corrected[(i + 1) % N]

            # Edge vector in local 2D projection
            edge_vec = v1[[i1, i2]] - v0[[i1, i2]]
            norm = np.linalg.norm(edge_vec)
            if norm < 1e-9:
                continue
            edge_unit = edge_vec / norm

            # 2D normal (CCW rotation)
            normal = np.array([-edge_unit[1], edge_unit[0]])

            # Check if normal points in negative direction (e.g. -z or -x)
            if normal[1] < -0.5:
                for vi in [i, (i + 1) % N]:
                    p = corrected[vi]

                    # Distance from point to axis (in 3D)
                    v = p - edge_origin
                    proj = np.dot(v, edge_direction) * edge_direction
                    dist = np.linalg.norm(v - proj)

                    if dist < tolerance:
                        corrected[vi, t_idx] = target_val
        return corrected

        

    def compute_scores(self, alpha_tilt, beta_tilt):
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


            # Step 3: Indentify contact verticies at base and back plane
            verts = rotated_mesh.vertices
            com = rotated_mesh.center_mass 
            min_z = np.min(verts[:, 2])  # base plane
            min_x = np.min(verts[:, 0])  # back plane

            edge_origin = np.array([min_x, 0.0, min_z])
            edge_direction = np.array([0.0, 1.0, 0.0])  # intersection of x=min_x and z=min_z
            
            # Step 3: Tilt the system so that the center of mass becomes offset by the tilt angle
            alpha = np.radians(alpha_tilt)
            beta = np.radians(beta_tilt)

            R_tilt = R.from_euler('xy', [alpha, -beta]).as_matrix()  # inverse tilt

            verts_tilted = verts @ R_tilt.T
            com_tilted   = com @ R_tilt.T

            #find the location of the verticies which are touching the back and base planes before the tilt is applied
            base_idx = np.where(np.abs(verts[:, 2] - min_z) < self.tolerance)[0]
            back_idx = np.where(np.abs(verts[:, 0] - min_x) < self.tolerance)[0]

            
            base_contact = verts[base_idx]
            back_contact = verts[back_idx]

            # 3. Apply snapping to edge before rotation, this is a modification of the resting surface to account for simulataneous resting on back contact
            base_contact_corrected = self.fix_contact_edge_alignment_axis(
                contact_polygon=base_contact,
                edge_origin=edge_origin,
                edge_direction=edge_direction,
                direction='x',
                target_val=min_x,
                tolerance=self.tolerance,
            )

            back_contact_corrected = self.fix_contact_edge_alignment_axis(
                contact_polygon=back_contact,
                edge_origin=edge_origin,
                edge_direction=edge_direction,
                direction='z',
                target_val=min_z,
                tolerance=self.tolerance,
            )

            
            #find the location of the verticies which are touching the edge before the tilt is applied
            edge_idx = np.intersect1d(base_idx, back_idx)

            # use the location of these verticies to extract the tilted polygons
            base_contact_tilt = base_contact_corrected @ R_tilt.T
            back_contact_tilt = back_contact_corrected @ R_tilt.T

            edge_contact_tilt = verts_tilted[edge_idx]
            

            if len(np.unique(base_contact_tilt,axis= 0)) >= 3:
                # create convex hull projection of tilted base convex hull onto the base xy plane
                hull_base = super().find_contact_polygon(base_contact_tilt[:, :2], min_z)

                path_base = Path(hull_base[:, :2])

                inside_base_polygon = path_base.contains_point(com_tilted, radius=self.stability_tolerance)

                if  inside_base_polygon:
                    x = hull_base[:, 0]
                    y = hull_base[:, 1]

                    area_base =  0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

                    Q_base = self.solid_angle_polygon(hull_base, com_tilted)

                    # critical solid angle
                    delta_base = self.critical_solid_angle(hull_base, com_tilted, Q_base, edge_contact_tilt)

                    # height (h)
                    A_base = np.c_[base_contact_tilt[:,0], base_contact_tilt[:,1], np.ones(len(base_contact_tilt))]

                    #solve least squares to find the location on plane directly below com_tilted
                    a_base, b_base, c_base = np.linalg.lstsq(A_base, base_contact_tilt[:,2], rcond=None)[0]

                    # z of plane directly under COM (same x,y)
                    #z_base = a_base*com_tilted[0] + b_base*com_tilted[1] + c_base

                    h = abs(com[2] - min_z)*np.cos(alpha)*np.cos(beta)

                    csa_base = Q_base/h
                    crsa_base = delta_base/h
                    stability_base = area_base/h
                else:
                    csa_base = 0.0
                    crsa_base = 0.0
                    stability_base = 0.0
            else:
                csa_base = 0.0
                crsa_base = 0.0
                stability_base = 0.0

            # calculate back stability values when COM shifts to over the back plane
            if (beta_tilt > 0.0) and (len(np.unique(back_contact_tilt,axis=0)) >= 3):
                
                # create convex hull projection of tilted back convex hull onto the base xy plane
                hull_back = super().find_contact_polygon(back_contact_tilt[:, :2], min_z)

                path_back = Path(hull_back[:, :2])

                inside_back_polygon = path_back.contains_point(com_tilted, radius=self.stability_tolerance)

                if inside_back_polygon:
                    x_back = hull_back[:, 0]
                    y_back = hull_back[:, 1]

                    area_back =  0.5 * np.abs(np.dot(x_back, np.roll(y_back, -1)) - np.dot(y_back, np.roll(x_back, -1)))

                    Q_back = self.solid_angle_polygon(hull_back, com_tilted)

                    # critical solid angle (Qcrit)
                    delta_back = self.critical_solid_angle(hull_back, com_tilted, Q_back, edge_contact_tilt)

                    # depth (d)
                    A_back = np.c_[back_contact_tilt[:,0], back_contact_tilt[:,1], np.ones(len(back_contact_tilt))]

                    #solve least squares to find the location on plane directly below com_tilted
                    a_back, b_back, c_back = np.linalg.lstsq(A_back, back_contact_tilt[:,2], rcond=None)[0]

                    # z of plane directly under COM (same x,y)
                    #z_back = a_back*com_tilted[0] + b_back*com_tilted[1] + c_back

                    d = abs(com[0] - min_x)*np.cos(alpha)*np.sin(beta)

                    csa_back = Q_back/d
                    stability_back = area_back/d
                    crsa_back = delta_back/d
                else:
                    csa_back = 0.0
                    stability_back = 0.0
                    crsa_back = 0.0
            else:
                csa_back = 0.0
                stability_back = 0.0
                crsa_back = 0.0
            
            csa_value = csa_base + csa_back
            stability_value = stability_base + stability_back
            crsa_value = max(crsa_base, crsa_back)

            # Debug plot
            plt.figure()
            ax = plt.gca()

            # Plot base polygon
            if hull_base is not None:
                plt.plot(hull_base[:, 0], hull_base[:, 1], 'k--', lw=1.5, label='Base Support Polygon')
                plt.plot(com_tilted[0], com_tilted[1], 'go', label='CoM over Base')

            # Plot back polygon (only if it exists and is distinct)
            if 'hull_back' in locals() and hull_back is not None:
                plt.plot(hull_back[:, 0], hull_back[:, 1], 'b--', lw=1.5, label='Back Support Polygon')  # YZ projection
                plt.plot(com_tilted[0], com_tilted[1], 'co', label='CoM over Back')

            # Plot settings
            ax.set_aspect('equal')
            plt.legend(loc='best')

            plt.text(
                0.95, 0.95,
                'csa = {:.4f} sr\nStability = {:.4f}\ncrsa = {:.4f} sr'.format(csa_value, stability_value, crsa_value),
                fontsize=8,
                transform=ax.transAxes,
                ha='right',
                va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

            plt.tight_layout()
            plt.savefig(f"csa_debug_{index:03d}.png", dpi=150)
            plt.close()

            self.csa_values.append(csa_value)  # compute centroid solid angle for one pose for both back and base
            self.stability_values.append(stability_value)  # compute stability score for one pose for both back and base
            self.crsa_values.append(crsa_value) # compute critical solid angle and pick bigger value (not sure if 100% right)

            if index not in seen_indicies:
                seen_indicies.append(index)
                non_symmetrical_csa_values_sum = csa_value + non_symmetrical_csa_values_sum
                non_symmetrical_stability_values_sum = stability_value + non_symmetrical_stability_values_sum
                non_symmetrical_crsa_values_sum = crsa_value + non_symmetrical_crsa_values_sum
            else:
                continue

        self.csa_scores = (np.array(self.csa_values) / non_symmetrical_csa_values_sum) * 100 
        self.stability_scores = (np.array(self.stability_values) / non_symmetrical_stability_values_sum)*100 
        self.crsa_scores = (np.array(self.crsa_values) / non_symmetrical_crsa_values_sum)*100 