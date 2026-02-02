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

    def safe_normalize_to_percent(values, total, eps=1e-12):
        values = np.asarray(values, dtype=float)
        if total <= eps:
            return np.zeros_like(values)
        return (values / (total + eps)) * 100.0    
        

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
            min_z = np.min(verts[:, 2])  # base plane
            min_x = np.min(verts[:, 0])  # back plane
            
            # Step 3: Tilt the system so that the center of mass becomes offset by the tilt angle
            alpha = np.radians(alpha_tilt)
            beta = np.radians(beta_tilt)

            if self.pose_types == 3:
                R_tilt = R.from_euler('xy', [alpha, 0]).as_matrix()  # inverse tilt
            else:
                R_tilt = R.from_euler('xy', [alpha, -beta]).as_matrix()  # inverse tilt

            verts_tilted = verts @ R_tilt.T
            com_tilted   = rotated_mesh.center_mass @ R_tilt.T

            #find the location of the verticies which are touching the back and base planes before the tilt is applied
            base_idx = np.where(np.abs(verts[:, 2] - min_z) < self.tolerance)[0]
            back_idx = np.where(np.abs(verts[:, 0] - min_x) < self.tolerance)[0]
            
            #find the location of the verticies which are touching the edge before the tilt is applied
            edge_idx = np.intersect1d(base_idx, back_idx)

            # use the location of these verticies to extract the tilted polygons
            base_contact_tilt = verts_tilted[base_idx]
            back_contact_tilt = verts_tilted[back_idx]

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
                    z_base = a_base*com_tilted[0] + b_base*com_tilted[1] + c_base

                    h = abs(com_tilted[2] - z_base)

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
                    z_back = a_back*com_tilted[0] + b_back*com_tilted[1] + c_back

                    d = abs(com_tilted[0] - z_back)

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

            ''' Debug plot
            plt.figure()
            plt.plot(hull_base[:, 0], hull_base[:, 1], 'k--', lw=1, label='Support Polygon')
            plt.plot(com_tilted[0], com_tilted[1], 'go', label='Center of Mass')
            plt.gca().set_aspect('equal')
            plt.legend()
            plt.text(0.95, 0.95, 'Q = {:.4f} sr\nStability Area = {:.4f} \nQcrit Sum = {:.4f} sr'.format(Q, area, delta_sum), fontsize=8, transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.savefig(f"csa_check_{index:03d}.png", dpi=150)
            plt.close()
            '''
            csa_value = csa_base + csa_back
            stability_value = stability_base + stability_back
            crsa_value = max(crsa_base, crsa_back)

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