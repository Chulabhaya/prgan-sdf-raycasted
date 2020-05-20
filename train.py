from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import distributions as dist
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

from discriminator import Discriminator
from generator import DeepSDF
from raycasting_utils import (
    cast_ray,
    generate_ray,
    get_camera_space_pixels_batch,
    get_camera_to_world_batch,
)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Train longer, test sphere
def main():
    # Especially important parameters
    num_train_epochs = 50
    batch_size = 32
    generator_lr = 0.0001  # Higher and lower values don't work
    discriminator_lr = 0.00001
    adam_beta_1 = 0.5
    adam_beta_2 = 0.999
    latent_dim = 256
    img_size = 32
    img_channels = 1
    discriminator_channels = 64
    sample_interval = 1000
    n_critic = 1
    batch_norm = False  # True doesn't work
    dataset_path = "data/airplane"

    # Set testing device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Loss function
    adversarial_loss = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss(reduction="sum")

    # Initialize generator and discriminator
    generator = DeepSDF(latent_channels=latent_dim, hidden_channels=256, num_layers=8)
    discriminator = Discriminator(
        in_channels=img_channels,
        d_channels=discriminator_channels,
        batch_norm=batch_norm,
        img_size=img_size,
    )

    # Send models to GPU
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Initialize weights
    # generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Configure data loader
    Path(dataset_path).mkdir(parents=True, exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            dataset_path,
            transform=transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=generator_lr, betas=(adam_beta_1, adam_beta_2)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=discriminator_lr,
        betas=(adam_beta_1, adam_beta_2),
    )

    # Dimensions of rendered 2D image
    silhouette_height = img_size
    silhouette_width = img_size

    # Field-of-view scaling
    fov_angle = 90
    scale = np.tan(np.deg2rad(fov_angle * 0.5))
    img_aspect_ratio = silhouette_width / silhouette_height

    # ------------
    #  Training
    # ------------
    for epoch in range(num_train_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            # Adversarial ground truths
            valid = torch.ones(imgs.shape[0], 1, device=device, requires_grad=False)
            fake = torch.zeros(imgs.shape[0], 1, device=device, requires_grad=False)
            z = torch.randn(imgs.shape[0], latent_dim, device=device)

            # Configure input
            real_imgs = imgs.to(device)

            # ------------
            #  Ray-casting
            # ------------
            # Camera-to-world matrix
            camera_to_world = get_camera_to_world_batch(imgs.shape[0])

            # Initialize raster space pixel coordinates
            pixel_camera_y, pixel_camera_x = get_camera_space_pixels_batch(
                silhouette_height,
                silhouette_width,
                scale,
                img_aspect_ratio,
                imgs.shape[0],
            )

            # Generate camera origin points and ray direction vectors
            camera_world_origin, ray_direction = generate_ray(
                pixel_camera_x, pixel_camera_y, camera_to_world
            )
            camera_world_origin = camera_world_origin.to(device)
            ray_direction = ray_direction.to(device)

            sdf_value, distance_iterations = cast_ray(
                camera_world_origin,
                ray_direction,
                generator,
                z,
                threshold=0.01,
                steps=10,
            )

            # change network to output occpuancy (0-1)
            # take fixed ray marching steps, e.g. 1/32, accumulate sum
            # NERF paper jittering

            distance_iterations = torch.stack(distance_iterations, dim=3)
            distance_iterations = torch.abs(distance_iterations)
            distance_iterations = torch.min(distance_iterations, 3).values

            gen_imgs = distance_iterations.view(-1, silhouette_height, silhouette_width)
            tau = 1
            gen_imgs = torch.exp(-tau * gen_imgs)
            gen_imgs = gen_imgs.unsqueeze(1)

            logits_real, prob_real, features_real = discriminator(real_imgs)
            logits_fake, prob_fake, features_fake = discriminator(gen_imgs)
            real_mean = features_real.mean(dim=0)
            real_std = features_real.std(dim=0)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_d.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(logits_real, valid).mean()
            fake_loss = adversarial_loss(
                discriminator(gen_imgs.detach())[0], fake
            ).mean()
            d_loss = (real_loss + fake_loss) / 2.0

            d_loss.backward(retain_graph=True)

            d_acc_real = prob_real.mean()
            d_acc_fake = torch.mean(1.0 - prob_fake)
            d_acc = (d_acc_real + d_acc_fake) / 2.0

            if d_acc < 0.75:
                optimizer_d.step()

            # Train the generator every n_critic iterations
            if i % n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_g.zero_grad()

                fake_mean = features_fake.mean(dim=0)
                fake_std = features_fake.std(dim=0)

                # Loss measures generator's ability to fool the discriminator
                # g_loss = adversarial_loss(discriminator(gen_imgs), valid).mean()
                g_loss = mse(fake_mean, real_mean)  # + mse(fake_std, real_std)

                g_loss.backward()
                optimizer_g.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D acc: %f]"
                % (
                    epoch,
                    num_train_epochs,
                    i,
                    len(dataloader),
                    d_loss.item(),
                    g_loss.item(),
                    d_acc,
                )
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0 and batches_done > 0:
                results_imgs_folder = Path("results/images/{}".format("current"))
                Path(results_imgs_folder).mkdir(parents=True, exist_ok=True)

                # Save generated images
                results_imgs_path = Path(
                    results_imgs_folder, "{}.png".format(batches_done)
                )
                save_image(
                    gen_imgs[:, 0, :, :].unsqueeze(1).data[:25],
                    results_imgs_path,
                    nrow=5,
                    normalize=False,
                )

                # Save ground truth images
                results_imgs_gt_path = Path(
                    results_imgs_folder, "{}gt.png".format(batches_done)
                )
                save_image(
                    real_imgs[:, 0, :, :].unsqueeze(1).data[:25],
                    results_imgs_gt_path,
                    nrow=5,
                    normalize=False,
                )

    # Save trained generator
    torch.save(generator.state_dict(), "{}_trained_generator.pth".format("current"))


if __name__ == "__main__":
    main()
