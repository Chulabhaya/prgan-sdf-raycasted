from pathlib import Path

import numpy as np
import torch
from skimage import measure
from torchvision.utils import save_image


def save_imgs(path, imgs, init_idx=0):
    for i in range(imgs.size(0)):
        save_image(
            imgs[i, 0, :, :].unsqueeze(0).unsqueeze(1).data,
            Path(path, "img{}.png".format(str(i + init_idx).zfill(4))),
            nrow=1,
            normalize=False,
            padding=0,
        )


def save_meshes(path, value_grids, level=0):
    value_grids = value_grids.detach().cpu().numpy()
    for i in range(value_grids.shape[0]):
        volume = value_grids[i, :, :, :]

        volume_max = np.amax(volume)

        if volume_max > level:
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                volume, level, step_size=1
            )
            faces = faces + 1
            verts[:, [1, 2]] = verts[
                :, [2, 1]
            ]  # Swap vertices to fix incorrect normals

            # Re-center and re-scale the mesh
            verts_normalized = normalize_mesh(verts)

            Path(path).mkdir(parents=True, exist_ok=True)
            mesh_file_path = Path(path, "mesh{}.obj".format(i))
            mesh_file = open(mesh_file_path, "w")
            for item in verts_normalized:
                mesh_file.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))

            for item in faces:
                mesh_file.write(
                    "f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0], item[1], item[2])
                )

            for item in normals:
                mesh_file.write("vn {0} {1} {2}\n".format(item[0], item[1], item[2]))

            mesh_file.close()


def save_points(path, coordinates, sdf_grids, occ_grids):
    coordinates = coordinates.detach().cpu().numpy()
    sdf_grids = sdf_grids.detach().cpu().numpy()
    occ_grids = occ_grids.detach().cpu().numpy()

    sdf_grids = np.reshape(
        sdf_grids,
        (
            sdf_grids.shape[0],
            sdf_grids.shape[1] * sdf_grids.shape[2] * sdf_grids.shape[3],
        ),
    )
    occ_grids = np.reshape(
        occ_grids,
        (
            occ_grids.shape[0],
            occ_grids.shape[1] * occ_grids.shape[2] * occ_grids.shape[3],
        ),
    )

    for i in range(coordinates.shape[0]):
        coordinates_current = coordinates[i, :, :]
        sdf_grids_current = sdf_grids[i, :]
        occ_grids_current = occ_grids[i, :]

        points_file_path = Path(path, "points{}.ply".format(i))
        points_file = open(points_file_path, "w")

        # Header
        points_file.write("ply\n")
        points_file.write("format ascii 1.0\n")
        points_file.write("element vertex " + str(coordinates.shape[1]) + "\n")
        points_file.write("property float32 x\n")
        points_file.write("property float32 y\n")
        points_file.write("property float32 z\n")
        points_file.write("property float32 confidence\n")
        points_file.write("element face 0\n")
        points_file.write("property list uint8 int32 vertex_indices\n")
        points_file.write("end_header\n")

        # Values
        for j in range(coordinates.shape[1]):
            points_file.write(
                "{0} {1} {2} {3}\n".format(
                    coordinates_current[j, 0],
                    coordinates_current[j, 1],
                    coordinates_current[j, 2],
                    occ_grids_current[j],
                )
            )

        points_file.close()


def normalize_mesh(vertices):
    # Calculate max and min values for each axis to
    # get existing bounding box
    current_bb_max = [
        np.max(vertices[:, 0]),
        np.max(vertices[:, 1]),
        np.max(vertices[:, 2]),
    ]
    current_bb_min = [
        np.min(vertices[:, 0]),
        np.min(vertices[:, 1]),
        np.min(vertices[:, 2]),
    ]

    # Calculate scales
    x_scale = current_bb_max[0] - current_bb_max[0]
    y_scale = current_bb_max[1] - current_bb_max[1]
    z_scale = current_bb_max[2] - current_bb_max[2]
    max_scale_dim = np.argmax([x_scale, y_scale, z_scale])

    # Calculate normalized vertices
    x_norm = (vertices[:, 0] - current_bb_min[max_scale_dim]) / (
        current_bb_max[max_scale_dim] - current_bb_min[max_scale_dim]
    )
    y_norm = (vertices[:, 1] - current_bb_min[max_scale_dim]) / (
        current_bb_max[max_scale_dim] - current_bb_min[max_scale_dim]
    )
    z_norm = (vertices[:, 2] - current_bb_min[max_scale_dim]) / (
        current_bb_max[max_scale_dim] - current_bb_min[max_scale_dim]
    )

    # Scale to new bounding box
    x_norm = x_norm * 1.0 - 0.5
    y_norm = y_norm * 1.0 - 0.5
    z_norm = z_norm * 1.0 - 0.5

    normalized_vertices = np.column_stack((x_norm, y_norm, z_norm))

    return normalized_vertices
