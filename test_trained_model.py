from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from generator import DeepSDF
from raycasting_utils import (
    get_camera_to_world_batch,
    get_camera_space_pixels_batch,
    generate_ray,
    cast_ray,
)

torch.manual_seed(0)
np.random.seed(0)


def main():
    # Set testing device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize generator
    generator = DeepSDF(latent_channels=256, hidden_channels=256, num_layers=8)
    generator.load_state_dict(torch.load(Path("current_trained_generator.pth")))
    # generator = OccNet(dim=3, latent_dim=256, hidden_size=256, leaky=True)
    # generator.load_state_dict(torch.load(Path('occ_net_no_ray_marching_no_tanh.pth')))
    generator = generator.to(device)
    generator.eval()

    # Batch size
    batch_size = 1

    # Dimensions of rendered 2D image
    silhouette_height = 32
    silhouette_width = silhouette_height

    # Field-of-view scaling
    fov_angle = 90
    scale = np.tan(np.deg2rad(fov_angle * 0.5))
    img_aspect_ratio = silhouette_width / silhouette_height

    # Camera-to-world matrix
    camera_to_world = get_camera_to_world_batch(batch_size)

    # ------------
    #  Ray-casting
    # ------------
    # Initialize raster space pixel coordinates
    pixel_camera_y, pixel_camera_x = get_camera_space_pixels_batch(
        silhouette_height, silhouette_width, scale, img_aspect_ratio, batch_size
    )

    # Generate camera origin points and ray direction vectors
    camera_world_origin, ray_direction = generate_ray(
        pixel_camera_x, pixel_camera_y, camera_to_world
    )
    camera_world_origin = camera_world_origin.to(device)
    ray_direction = ray_direction.to(device)

    # Cast rays
    with torch.no_grad():
        z = torch.randn(batch_size, 256, device=device)
        sdf_value, distance_iterations = cast_ray(
            camera_world_origin, ray_direction, generator, z, threshold=0.01, steps=10
        )

        # for i in range(len(distance_iterations)):
        #     print("Saving figure {}".format(i))
        #     sdf_slice_current = distance_iterations[i]

        #     silhouette_final = sdf_slice_current.view(
        #         -1, silhouette_height, silhouette_width
        #     ).squeeze()
        #     silhouette_np = silhouette_final.cpu().numpy()

        #     plt.figure()
        #     plt.imshow(silhouette_np)
        #     plt.colorbar()
        #     plt.savefig("plots/slice_{}.png".format(i))
        #     plt.close()

        sdf_slices = torch.stack(distance_iterations, dim=3)
        sdf_slices = torch.abs(sdf_slices)
        sdf_slices = torch.min(sdf_slices, 3).values

        # Display silhouette
        silhouette_final = sdf_slices.view(
            -1, silhouette_height, silhouette_width
        ).squeeze()
        tau = 1
        silhouette = torch.exp(-tau * silhouette_final)
        silhouette_np = silhouette.cpu().numpy()
        plt.figure()
        plt.imshow(silhouette_np)
        plt.colorbar()
        plt.show()

        print("test")


if __name__ == "__main__":
    main()
