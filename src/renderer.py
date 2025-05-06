"""
PyTorch implementation of Gaussian Splat Rasterizer.

The implementation is based on torch-splatting: https://github.com/hbb1/torch-splatting
"""

from jaxtyping import Bool, Float, jaxtyped
import torch
from typeguard import typechecked

from .camera import Camera
from .scene import Scene
from .sh import eval_sh


class GSRasterizer(object):
    """
    Gaussian Splat Rasterizer.
    """

    def __init__(self):

        self.sh_degree = 3
        self.white_bkgd = True
        self.tile_size = 25

    def render_scene(self, scene: Scene, camera: Camera):

        # Retrieve Gaussian parameters
        mean_3d = scene.mean_3d
        scales = scene.scales
        rotations = scene.rotations
        shs = scene.shs
        opacities = scene.opacities
        
        # ============================================================================
        # Process camera parameters
        # NOTE: We transpose both camera extrinsic and projection matrices
        # assuming that these transforms are applied to points in row vector format.
        # NOTE: Do NOT modify this block.
        # Retrieve camera pose (extrinsic)
        R = camera.camera_to_world[:3, :3]  # 3 x 3
        T = camera.camera_to_world[:3, 3:4]  # 3 x 1
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=R.device, dtype=R.dtype))
        R = R @ R_edit
        R_inv = R.T
        T_inv = -R_inv @ T
        world_to_camera = torch.eye(4, device=R.device, dtype=R.dtype)
        world_to_camera[:3, :3] = R_inv
        world_to_camera[:3, 3:4] = T_inv
        world_to_camera = world_to_camera.permute(1, 0)

        # Retrieve camera intrinsic
        proj_mat = camera.proj_mat.permute(1, 0)
        world_to_camera = world_to_camera.to(mean_3d.device)
        proj_mat = proj_mat.to(mean_3d.device)
        # ============================================================================

        # Project Gaussian center positions to NDC
        mean_ndc, mean_view, in_mask = self.project_ndc(
            mean_3d, world_to_camera, proj_mat, camera.near,
        )
        mean_ndc = mean_ndc[in_mask]
        mean_view = mean_view[in_mask]
        assert mean_ndc.shape[0] > 0, "No points in the frustum"
        assert mean_view.shape[0] > 0, "No points in the frustum"
        depths = mean_view[:, 2]

        # Compute RGB from spherical harmonics
        color = self.get_rgb_from_sh(mean_3d, shs, camera)

        # Compute 3D covariance matrix
        cov_3d = self.compute_cov_3d(scales, rotations)

        # Project covariance matrices to 2D
        cov_2d = self.compute_cov_2d(
            mean_3d=mean_3d, 
            cov_3d=cov_3d, 
            w2c=world_to_camera,
            f_x=camera.f_x, 
            f_y=camera.f_y,
        )
        
        # Compute pixel space coordinates of the projected Gaussian centers
        mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
        mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
        mean_2d = torch.stack([mean_coord_x, mean_coord_y], dim=-1)

        color = self.render(
            camera=camera, 
            mean_2d=mean_2d,
            cov_2d=cov_2d,
            color=color,
            opacities=opacities, 
            depths=depths,
        )
        color = color.reshape(-1, 3)

        return color

    @torch.no_grad()
    def get_rgb_from_sh(self, mean_3d, shs, camera):
        rays_o = camera.cam_center        
        rays_d = mean_3d - rays_o
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        color = eval_sh(self.sh_degree, shs.permute(0, 2, 1), rays_d)
        color = torch.clamp_min(color + 0.5, 0.0)
        return color
    
    @jaxtyped(typechecker=typechecked)
    @torch.no_grad()
    def project_ndc(
        self,
        points: Float[torch.Tensor, "N 3"],
        w2c: Float[torch.Tensor, "4 4"],
        proj_mat: Float[torch.Tensor, "4 4"],
        z_near: float,
    ) -> tuple[
        Float[torch.Tensor, "N 4"],
        Float[torch.Tensor, "N 4"],
        Bool[torch.Tensor, "N"],
    ]:
        """
        Projects points to NDC space.
        
        Args:
        - points: 3D points in object space.
        - w2c: World-to-camera matrix.
        - proj_mat: Projection matrix.
        - z_near: Near plane distance.

        Returns:
        - p_ndc: NDC coordinates.
        - p_view: View space coordinates.
        - in_mask: Mask of points that are in the frustum.
        """
        # ========================================================
        # TODO: Implement the projection to NDC space
        # 1) Make homogeneous (Nx4)
        p_world_h = homogenize(points)          # [N, 4]

        # 2) World to camera space
        p_view = p_world_h @ w2c                # [N, 4]

        # 3) Camera to clip space
        p_clip = p_view @ proj_mat              # [N, 4]

        # 4) Perspective divide to get NDC
        w = p_clip[:, 3:4]                      # [N, 1]
        # Avoid division by zero
        eps = torch.tensor(1e-6, device=w.device, dtype=w.dtype)
        w_safe = torch.where(w == 0, eps, w)
        p_ndc = p_clip / w_safe                  # [N, 4]

        # TODO: Cull points that are close or behind the camera
        # 5) Mask out points that are behind the near plane
        # Camera looks along +z, so z_view > z_near is in front of the camera
        in_mask = p_view[:, 2] > z_near         # [N]
        # ========================================================

        return p_ndc, p_view, in_mask

    @torch.no_grad()
    def compute_cov_3d(self, s, r):
        L = build_scaling_rotation(s, r)
        cov3d = L @ L.transpose(1, 2)
        return cov3d

    @jaxtyped(typechecker=typechecked)
    @torch.no_grad()
    def compute_cov_2d(
        self,
        mean_3d: Float[torch.Tensor, "N 3"],
        cov_3d: Float[torch.Tensor, "N 3 3"],
        w2c: Float[torch.Tensor, "4 4"],
        f_x: Float[torch.Tensor, ""],
        f_y: Float[torch.Tensor, ""],
    ) -> Float[torch.Tensor, "N 2 2"]:
        """
        Projects 3D covariances to 2D image plane.

        Args:
        - mean_3d: Coordinates of center of 3D Gaussians.       # [Nx3]
        - cov_3d: 3D covariance matrix.                         # [Nx3x3]
        - w2c: World-to-camera matrix.                          # [4x4]
        - f_x: Focal length along x-axis.
        - f_y: Focal length along y-axis.

        Returns:
        - cov_2d: 2D covariance matrix.                         # [Nx2x2]
        """ 
        # ========================================================
        # TODO: Transform 3D mean coordinates to camera space
        # ========================================================

        # Transpose the rigid transformation part of the world-to-camera matrix

        # 1) Compute Jacobian J for each Gaussian
        # Make a batch-homogeneous view-space point
        p_view_h = homogenize(mean_3d) @ w2c                                # [N, 4]
        t_x, t_y, t_z = p_view_h[:, 0], p_view_h[:, 1], p_view_h[:, 2]      # [N]

        # Initialize the Jacobian J = zero [Nx3x3]
        J = torch.zeros(mean_3d.shape[0], 3, 3, device=mean_3d.device, dtype=mean_3d.dtype)

        # fill in according to
        #   J = [[ f_x/t_z,        0,  -f_x t_x / t_z^2 ],
        #        [      0,   f_y/t_z,  -f_y t_y / t_z^2 ],
        #        [      0,        0,               0   ]]
        J[:, 0, 0] = f_x / t_z
        J[:, 0, 2] = -f_x * t_x / (t_z * t_z)
        J[:, 1, 1] = f_y / t_z
        J[:, 1, 2] = -f_y * t_y / (t_z * t_z)
        # The rest of the Jacobian is zero

        # 2) Rotate the world-covariance to camera space, then project to 2D
        # Extract the world-to-camera rotation part
        W = w2c[:3, :3].T                       # [3x3]

        # Σ_camera = W Σ_3D Wᵀ   →  [N×3×3]
        cov_cam = W @ cov_3d                    # broadcast W [3×3] over [N×3×3] → [N×3×3]
        cov_cam = cov_cam @ W.T                 # [N×3×3] @ [3×3] → [N×3×3]

        # ========================================================
        # TODO: Compute Jacobian of view transform and projection
        # Σ_2D = J Σ_camera Jᵀ  →  [N×3×3]
        cov_2d = J @ cov_cam @ J.permute(0, 2, 1)
        # ========================================================

        # add low pass filter here according to E.q. 32
        filt = torch.eye(2, device=cov_2d.device, dtype=cov_2d.dtype) * 0.3
        return cov_2d[:, :2, :2] + filt[None]

    @jaxtyped(typechecker=typechecked)
    @torch.no_grad()
    def render(
        self,
        camera: Camera,
        mean_2d: Float[torch.Tensor, "N 2"],
        cov_2d: Float[torch.Tensor, "N 2 2"],
        color: Float[torch.Tensor, "N 3"],
        opacities: Float[torch.Tensor, "N 1"],
        depths: Float[torch.Tensor, "N"],
    ) -> Float[torch.Tensor, "H W 3"]:
        radii = get_radius(cov_2d)
        rect = get_rect(mean_2d, radii, width=camera.image_width, height=camera.image_height)

        pix_coord = torch.stack(
            torch.meshgrid(torch.arange(camera.image_height), torch.arange(camera.image_width), indexing='xy'),
            dim=-1,
        ).to(mean_2d.device)
        
        render_color = torch.ones(*pix_coord.shape[:2], 3).to(mean_2d.device)

        assert camera.image_height % self.tile_size == 0, "Image height must be divisible by the tile_size."
        assert camera.image_width % self.tile_size == 0, "Image width must be divisible by the tile_size."
        for h in range(0, camera.image_height, self.tile_size):
            for w in range(0, camera.image_width, self.tile_size):
                # check if the rectangle penetrate the tile
                over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
                over_br = rect[1][..., 0].clip(max=w+self.tile_size-1), rect[1][..., 1].clip(max=h+self.tile_size-1)
                
                # a binary mask indicating projected Gaussians that lie in the current tile
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])
                if not in_mask.sum() > 0:
                    continue

                # ========================================================
                # TODO: Sort the projected Gaussians that lie in the current tile by their depths, in ascending order
                # ========================================================
                # 1) Sort the overlapping Gaussians by ascending depth
                idxs = torch.nonzero(in_mask, as_tuple=False).squeeze(-1)           # [M]
                depths_tile = depths[idxs]                                          # [M]
                _, order = torch.sort(depths_tile, descending=False)
                idxs = idxs[order]                                                  # [M]

                # ========================================================
                # TODO: Compute the displacement vector from the 2D mean coordinates to the pixel coordinates
                # ========================================================
                # 2) Build displacement vectors d_{i,j} for each pixel in the tile
                tile_pix = pix_coord[h:h+self.tile_size, w:w+self.tile_size]       # [tile_size, tile_size, 2]
                tile_flat = tile_pix.reshape(-1, 2)                                # [T², 2]
                # Broadcast the center coordinates of the Gaussians to the tile
                d_ij = tile_flat[None, :, :] - mean_2d[idxs][:, None, :]            # [M, T², 2]

                # ========================================================
                # TODO: Compute the Gaussian weight for each pixel in the tile
                # ========================================================
                # 3) Compute Gaussian weights w_{i,j}
                cov_inv = torch.inverse(cov_2d[idxs])                           # [M, 2, 2]
                # Mahalanobis (d @ Σ⁻¹ @ dᵀ)
                u = cov_inv.unsqueeze(1) @ d_ij.unsqueeze(-1)                   # [M, T², 2, 1]
                quad = (d_ij.unsqueeze(-1) * u).sum(dim=2).squeeze(-1)          # [M, T²]
                w_ij = torch.exp(-0.5 * quad)                                   # [M, T²]
                # Modulate by opacity α_j
                alpha_tilde = w_ij * opacities[idxs].squeeze(-1)[:, None]             # [M, T²]

                # ========================================================
                # TODO: Perform alpha blending
                # 4) Front-to-back alpha blending
                T_acc = torch.ones(tile_flat.shape[0], device=color.device, dtype=color.dtype)           # [T²]
                tile_color = torch.zeros(tile_flat.shape[0], 3, device=color.device, dtype=color.dtype)  # [T²×3]
                for m, j in enumerate(idxs):
                    a = alpha_tilde[m]                                                                   # [T²]
                    c_j = color[j]                                                                       # [3]
                    contrib = (a * T_acc).unsqueeze(-1) * c_j                                            # [T²×3]
                    tile_color += contrib
                    T_acc = T_acc * (1.0 - a)
                # ========================================================

                render_color[h:h+self.tile_size,
                             w:w+self.tile_size
                ] = tile_color.reshape(self.tile_size, self.tile_size, 3)

        return render_color

@torch.no_grad()
def homogenize(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

@torch.no_grad()
def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

@torch.no_grad()
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max
