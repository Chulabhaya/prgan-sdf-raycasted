import numpy as np
import torch


def get_camera_to_world_batch(batch_size):
    view_angles = torch.tensor(
        [
            0,
            np.pi / 4.0,
            np.pi / 2.0,
            3 * np.pi / 4.0,
            np.pi,
            5 * np.pi / 4.0,
            3 * np.pi / 2.0,
            7 * np.pi / 4.0,
        ]
    )
    rotation_picks = torch.randint(0, 8, (batch_size,))  # Random views
    # rotation_picks = (
    #     0 * torch.ones(batch_size).long()
    # )  # For debugging with a single view angle
    random_rotation_angles = view_angles[rotation_picks]

    camera_rotation = torch.zeros(batch_size, 4, 4)
    camera_rotation[:, 0, 0] = torch.cos(random_rotation_angles)
    camera_rotation[:, 0, 2] = torch.sin(random_rotation_angles)
    camera_rotation[:, 1, 1] = 1.0
    camera_rotation[:, 2, 0] = -torch.sin(random_rotation_angles)
    camera_rotation[:, 2, 2] = torch.cos(random_rotation_angles)
    camera_rotation[:, 3, 3] = 1

    x_translation = 0
    y_translation = 0
    z_translation = 0
    camera_translation = torch.zeros(batch_size, 4, 4)
    camera_translation[:, 0, 0] = 1
    camera_translation[:, 0, 3] = x_translation
    camera_translation[:, 1, 1] = 1
    camera_translation[:, 1, 3] = y_translation
    camera_translation[:, 2, 2] = 1
    camera_translation[:, 2, 3] = z_translation
    camera_translation[:, 3, 3] = 1

    camera_to_world = torch.bmm(camera_rotation, camera_translation)

    return camera_to_world


def get_camera_space_pixels_batch(
    silhouette_height, silhouette_width, scale, img_aspect_ratio, batch_size
):
    # Initialize pixels in raster space
    pixel_y, pixel_x = torch.meshgrid(
        [
            torch.arange(0, silhouette_height, dtype=torch.float),
            torch.arange(0, silhouette_width, dtype=torch.float),
        ]
    )
    pixel_y = pixel_y.reshape((silhouette_height * silhouette_width, -1))
    pixel_x = pixel_x.reshape((silhouette_height * silhouette_width, -1))

    # Convert raster space to camera space
    pixel_camera_y = (1 - 2 * (pixel_y + 0.5) / silhouette_height) * scale
    pixel_camera_x = (
        (2 * (pixel_x + 0.5) / silhouette_width - 1) * img_aspect_ratio * scale
    )

    pixel_camera_y = pixel_camera_y.unsqueeze(0).repeat(int(batch_size), 1, 1)
    pixel_camera_x = pixel_camera_x.unsqueeze(0).repeat(int(batch_size), 1, 1)

    return pixel_camera_y, pixel_camera_x


def generate_ray(pixel_camera_x, pixel_camera_y, camera_to_world):
    # Default camera space point locations
    camera_origin_z = torch.zeros(pixel_camera_y.size(0), pixel_camera_y.size(1), 1)
    pixel_camera_z = -1.0 * torch.ones(
        pixel_camera_y.size(0), pixel_camera_y.size(1), 1
    )
    homogeneous_coordinate_value = torch.ones(
        pixel_camera_y.size(0), pixel_camera_y.size(1), 1
    )

    camera_origin = torch.cat(
        [pixel_camera_x, pixel_camera_y, camera_origin_z, homogeneous_coordinate_value],
        2,
    )
    pixel_camera_point = torch.cat(
        [pixel_camera_x, pixel_camera_y, pixel_camera_z, homogeneous_coordinate_value],
        2,
    )

    # Convert from camera to world space
    # Normally you do Rotation @ Column Vector, but backwards here Row Vector @ Rotation.T
    # so matrix operations work out nicely
    camera_origin_world = torch.bmm(camera_origin, camera_to_world.transpose(1, 2))[
        :, :, 0:3
    ]
    pixel_world_point = torch.bmm(pixel_camera_point, camera_to_world.transpose(1, 2))[
        :, :, 0:3
    ]

    # Normalize the direction vectors
    raw_ray_direction = pixel_world_point - camera_origin_world
    ray_direction_length = torch.norm(raw_ray_direction, dim=2).unsqueeze(2)
    normalized_ray_direction = raw_ray_direction / ray_direction_length

    return camera_origin_world, normalized_ray_direction


def cast_ray(camera_origin_world, ray_direction, sdf, z, threshold, steps):
    point = camera_origin_world

    distance_iterations = []
    for step in range(steps):
        distance = sdf(point, z)
        # dist_test = distance.detach().cpu().numpy()
        # print(
        #     "Step: {}, Min: {}, Max: {}".format(
        #         step, np.min(dist_test), np.max(dist_test)
        #     )
        # )

        distance_iterations.append(distance)

        point = torch.where(
            (torch.abs(distance) < threshold) | (torch.abs(distance) > 100),
            point,
            point + 0.01 * (distance * ray_direction),
        )

    return distance, distance_iterations
