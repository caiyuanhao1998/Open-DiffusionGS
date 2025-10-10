from typing import Optional
import torch
import torch.nn.functional as F
import numpy as np
import trimesh
from trimesh.visual import create_visual

def dot(x, y, dim=-1):
    return torch.sum(x * y, dim=dim, keepdim=True)

class Mesh:
    def __init__(self, v_pos:torch.Tensor, t_pos_idx:torch.Tensor, **kwargs):
        self.v_pos = v_pos
        self.t_pos_idx = t_pos_idx
        self._v_nrm = None
        self._v_tng = None
        self._v_tex = None
        self._t_tex_idx = None
        self._v_tex_map = None
        self._edges = None

        assert v_pos.device == t_pos_idx.device, \
            f'v_pos is on {v_pos.device} but t_pos_idx is on {t_pos_idx.device}'
        self.device = v_pos.device
    
    def to(self, device):
        for key in [
            'v_pos',
            't_pos_idx',
            '_v_nrm',
            '_v_tng',
            '_v_tex',
            '_t_tex_idx',
            '_v_tex_map',
            '_edges',
        ]:
            value = getattr(self, key)
            if isinstance(value, torch.Tensor) and value.device != torch.device(device):
                setattr(self, key, value.to(device))
        self.device = torch.device(device)
        return self
    
    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh):
        v_pos = torch.as_tensor(mesh.vertices, dtype=torch.float32)
        t_pos_idx = torch.as_tensor(mesh.faces, dtype=torch.float32)
        return Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx)
    
    def to_trimesh(self) -> trimesh.Trimesh:
        mesh = trimesh.Trimesh(
            vertices=self.v_pos.detach().cpu().numpy(), 
            faces=self.t_pos_idx.cpu().numpy(), 
            process=True, validate=True,
        )
        return mesh

    def remove_outlier(self, outlier_n_faces_threshold):
        if self.requires_grad:
            print("Mesh is differentiable, not removing outliers")
            return self

        # use trimesh to first split the mesh into connected components
        # then remove the components with less than n_face_threshold faces
        import trimesh

        # construct a trimesh object
        mesh = trimesh.Trimesh(
            vertices=self.v_pos.detach().cpu().numpy(),
            faces=self.t_pos_idx.detach().cpu().numpy(),
        )

        # split the mesh into connected components
        components = mesh.split(only_watertight=False)
        # log the number of faces in each component
        print(
            "Mesh has {} components, with faces: {}".format(
                len(components), [c.faces.shape[0] for c in components]
            )
        )

        n_faces_threshold: int
        if isinstance(outlier_n_faces_threshold, float):
            # set the threshold to the number of faces in the largest component multiplied by outlier_n_faces_threshold
            n_faces_threshold = int(
                max([c.faces.shape[0] for c in components]) * outlier_n_faces_threshold
            )
        else:
            # set the threshold directly to outlier_n_faces_threshold
            n_faces_threshold = outlier_n_faces_threshold

        # log the threshold
        print(
            "Removing components with less than {} faces".format(n_faces_threshold)
        )

        # remove the components with less than n_face_threshold faces
        components = [c for c in components if c.faces.shape[0] >= n_faces_threshold]

        # log the number of faces in each component after removing outliers
        print(
            "Mesh has {} components after removing outliers, with faces: {}".format(
                len(components), [c.faces.shape[0] for c in components]
            )
        )
        # merge the components
        mesh = trimesh.util.concatenate(components)

        # convert back to our mesh format
        v_pos = torch.from_numpy(mesh.vertices).to(self.v_pos)
        t_pos_idx = torch.from_numpy(mesh.faces).to(self.t_pos_idx)

        clean_mesh = Mesh(v_pos, t_pos_idx)
        # keep the extras unchanged

        if len(self.extras) > 0:
            clean_mesh.extras = self.extras
            print(
                f"The following extra attributes are inherited from the original mesh unchanged: {list(self.extras.keys())}"
            )
        return clean_mesh

    def blender_to_threejs(self):
        v_pos = self.v_pos[:, [0, 2, 1]]
        v_pos[:, 2] = -v_pos[:, 2]
        self.v_pos = v_pos

    @property
    def v_nrm(self):
        if self._v_nrm is None:
            self._v_nrm = self._compute_vertex_normal()
        return self._v_nrm

    @property
    def v_tng(self):
        if self._v_tng is None:
            self._v_tng = self._compute_vertex_tangent()
        return self._v_tng

    # @property
    # def v_tex(self):
    #     if self._v_tex is None:
    #         self._v_tex_map, self._t_tex_idx, self._v_tex = self._unwrap_uv()
    #     return self._v_tex

    # @property
    # def t_tex_idx(self):
    #     if self._t_tex_idx is None:
    #         self._v_tex_map, self._t_tex_idx, self._v_tex = self._unwrap_uv()
    #     return self._t_tex_idx
    
    # @property
    # def v_tex_map(self):
    #     if self._v_tex_map is None:
    #         self._v_tex_map, self._t_tex_idx, self._v_tex = self._unwrap_uv()
    #     return self._v_tex_map

    @property
    def edges(self):
        if self._edges is None:
            self._edges = self._compute_edges()
        return self._edges

    def _compute_vertex_normal(self):
        i0 = self.t_pos_idx[:, 0].long()
        i1 = self.t_pos_idx[:, 1].long()
        i2 = self.t_pos_idx[:, 2].long()

        v0 = self.v_pos[i0, :]
        v1 = self.v_pos[i1, :]
        v2 = self.v_pos[i2, :]

        face_normals = torch.linalg.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        v_nrm = torch.zeros_like(self.v_pos)
        v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        v_nrm = torch.where(
            dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
        )
        v_nrm = F.normalize(v_nrm, dim=1)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_nrm))

        return v_nrm

    def _compute_vertex_tangent(self):
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0, 3):
            pos[i] = self.v_pos[self.t_pos_idx[:, i]]
            tex[i] = self.v_tex[self.t_tex_idx[:, i]]
            # t_nrm_idx is always the same as t_pos_idx
            vn_idx[i] = self.t_pos_idx[:, i]

        tangents = torch.zeros_like(self.v_nrm)
        tansum = torch.zeros_like(self.v_nrm)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1 = pos[1] - pos[0]
        pe2 = pos[2] - pos[0]

        nom = pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2]
        denom = uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1]

        # Avoid division by zero for degenerated texture coordinates
        tang = nom / torch.where(
            denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6)
        )

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(
                0, idx, torch.ones_like(tang)
            )  # tansum[n_i] = tansum[n_i] + 1
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = F.normalize(tangents, dim=1)
        tangents = F.normalize(tangents - dot(tangents, self.v_nrm) * self.v_nrm)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))

        return tangents

    def _unwrap_uv(
        self, xatlas_chart_options: dict = {}, xatlas_pack_options: dict = {}
    ):
        xatlas_chart_options = {
            'max_chart_area': 0.0,
            'max_boundary_length': 0.0,
            'normal_deviation_weight': 2.0,
            'roundness_weight': 0.01,
            'straightness_weight': 6.0,
            'normal_seam_weight': 4.0,
            'texture_seam_weight': 0.5,
            'max_cost': 16.0,  # avoid small charts
            'max_iterations': 1,
            'use_input_mesh_uvs': False,
            'fix_winding': False,
        }
        xatlas_pack_options = {
            'max_chart_size': 0,
            'padding': 4,  # avoid adjoint
            'texels_per_unit': 0.0,
            'resolution': 2048,
            'bilinear': True,
            'blockAlign': False,
            'bruteForce': False,
            'create_image': False,
            'rotate_charts_to_axis': True,
            'rotate_charts': True,
        }
        print("Using xatlas to perform UV unwrapping, may take a while ...")

        import xatlas

        atlas = xatlas.Atlas()
        atlas.add_mesh(
            self.v_pos.detach().cpu().numpy(),
            self.t_pos_idx.cpu().numpy(),
        )
        co = xatlas.ChartOptions()
        po = xatlas.PackOptions()
        for k, v in xatlas_chart_options.items():
            setattr(co, k, v)
        for k, v in xatlas_pack_options.items():
            setattr(po, k, v)
        
        atlas.generate(co, po, verbose=True)
        vmapping, indices, uvs = atlas.get_mesh(0)
        vmapping = (
            torch.from_numpy(
                vmapping.astype(np.uint64, casting="same_kind").view(np.int64)
            )
            .to(self.v_pos.device)
            .long()
        )
        uvs = torch.from_numpy(uvs).to(self.v_pos.device).float()
        indices = (
            torch.from_numpy(
                indices.astype(np.uint64, casting="same_kind").view(np.int64)
            )
            .to(self.v_pos.device)
            .long()
        )
        return vmapping, indices, uvs

    def unwrap_uv(
        self, xatlas_chart_options: dict = {}, xatlas_pack_options: dict = {}
    ):
        self._v_tex_map, self._t_tex_idx, self._v_tex = self._unwrap_uv(
            xatlas_chart_options, xatlas_pack_options
        )

    def _compute_edges(self):
        # Compute edges
        edges = torch.cat(
            [
                self.t_pos_idx[:, [0, 1]],
                self.t_pos_idx[:, [1, 2]],
                self.t_pos_idx[:, [2, 0]],
            ],
            dim=0,
        )
        edges = edges.sort()[0]
        edges = torch.unique(edges, dim=0)
        return edges

    def normal_consistency(self):
        edge_nrm = self.v_nrm[self.edges]
        nc = (
            1.0 - torch.cosine_similarity(edge_nrm[:, 0], edge_nrm[:, 1], dim=-1)
        ).mean()
        return nc

    def _laplacian_uniform(self):
        # from stable-dreamfusion
        # https://github.com/ashawkey/stable-dreamfusion/blob/8fb3613e9e4cd1ded1066b46e80ca801dfb9fd06/nerf/renderer.py#L224
        verts, faces = self.v_pos, self.t_pos_idx

        V = verts.shape[0]
        F = faces.shape[0]

        # Neighbor indices
        ii = faces[:, [1, 2, 0]].flatten()
        jj = faces[:, [2, 0, 1]].flatten()
        adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(
            dim=1
        )
        adj_values = torch.ones(adj.shape[1]).to(verts)

        # Diagonal indices
        diag_idx = adj[0]

        # Build the sparse matrix
        idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
        values = torch.cat((-adj_values, adj_values))

        # The coalesce operation sums the duplicate indices, resulting in the
        # correct diagonal
        return torch.sparse_coo_tensor(idx, values, (V, V)).coalesce()

    def laplacian(self):
        with torch.no_grad():
            L = self._laplacian_uniform()
        loss = L.mm(self.v_pos)
        loss = loss.norm(dim=1)
        loss = loss.mean()
        return loss


class Texture:
    def __init__(
        self, 
        mesh:Mesh, 
        v_rgb:Optional[torch.Tensor]=None, 
        map_Kd:Optional[torch.Tensor]=None,
        map_Ks:Optional[torch.Tensor]=None,
        **kwargs,
    ) -> None:
        self.mesh: Mesh = mesh
        self.v_rgb = v_rgb
        self.map_Kd = map_Kd
        self.map_Ks = map_Ks

        assert v_rgb is None or mesh.device == v_rgb.device, \
            f'mesh is on {mesh.device} but v_rgb is on {v_rgb.device}'
        assert map_Kd is None or mesh.device == map_Kd.device, \
            f'mesh is on {mesh.device} but map_Kd is on {map_Kd.device}'
        assert map_Ks is None or mesh.device == map_Ks.device, \
            f'mesh is on {mesh.device} but map_Ks is on {map_Ks.device}'
        self.device = mesh.device
    
    def to(self, device):
        for key in [
            'mesh',
        ]:
            value = getattr(self, key)
            if value is not None and value.device != torch.device(device):
                setattr(self, key, value.to(device))
        for key in [
            'v_rgb',
            'map_Kd',
            'map_Ks',
        ]:
            value = getattr(self, key)
            if isinstance(value, torch.Tensor) and value.device != torch.device(device):
                setattr(self, key, value.to(device))
        self.device = torch.device(device)
        return self

    def to_trimesh(self) -> trimesh.Trimesh:
        mesh = self.mesh.to_trimesh()
        if self.v_rgb is not None:
            mesh.visual = create_visual(
                face_colors = None,
                vertex_colors = self.v_rgb.detach().cpu().numpy(),
                mesh = mesh,
            )
        return mesh
    
    @classmethod
    def from_trimesh(cls, mesh: trimesh.Trimesh):
        if mesh.visual is not None:
            v_rgb = torch.as_tensor(mesh.visual.vertex_colors, dtype=torch.float32)
        else:
            v_rgb = None
        mesh = Mesh.from_trimesh(mesh)
        return Texture(mesh=mesh, v_rgb=v_rgb, map_Kd=None, map_Ks=None)

