import os
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from .modeling_lpips import LPIPS
from .modeling_discriminator import NLayerDiscriminator, NLayerDiscriminator3D, weights_init


class AdaptiveLossWeight:
    def __init__(self, timestep_range=[0, 1], buckets=300, weight_range=[1e-7, 1e7]):
        self.bucket_ranges = torch.linspace(timestep_range[0], timestep_range[1], buckets-1)
        self.bucket_losses = torch.ones(buckets)
        self.weight_range = weight_range

    def weight(self, timestep):
        indices = torch.searchsorted(self.bucket_ranges.to(timestep.device), timestep)
        return (1/self.bucket_losses.to(timestep.device)[indices]).clamp(*self.weight_range)

    def update_buckets(self, timestep, loss, beta=0.99):
        indices = torch.searchsorted(self.bucket_ranges.to(timestep.device), timestep).cpu()
        self.bucket_losses[indices] = self.bucket_losses[indices]*beta + loss.detach().cpu() * (1-beta)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        perceptual_weight=1.0,
        lpips_ckpt='/home/jinyang06/models/vae/video_vae_baseline/vgg_lpips.pth',
        # --- Discriminator Loss ---
        disc_num_layers=4,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=0.5,
        disc_loss="hinge",
        add_discriminator=True,
        using_3d_discriminator=False,
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS(lpips_ckpt_path=lpips_ckpt).eval()
        self.perceptual_weight = perceptual_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        if add_discriminator:
            disc_cls = NLayerDiscriminator3D if using_3d_discriminator else NLayerDiscriminator
            self.discriminator = disc_cls(
                input_nc=disc_in_channels, n_layers=disc_num_layers,
            ).apply(weights_init)
        else:
            self.discriminator = None
    
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.using_3d_discriminator = using_3d_discriminator

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        split="train",
        last_layer=None,
    ):
        t = reconstructions.shape[2]
        inputs = rearrange(inputs, "b c t h w -> (b t) c h w").contiguous()
        reconstructions = rearrange(reconstructions, "b c t h w -> (b t) c h w").contiguous()
    
        if optimizer_idx == 0:
            # rec_loss = torch.mean(torch.abs(inputs - reconstructions), dim=(1,2,3), keepdim=True)
            rec_loss = torch.mean(F.mse_loss(inputs, reconstructions, reduction='none'), dim=(1,2,3), keepdim=True)

            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs, reconstructions)
                nll_loss = self.pixel_weight * rec_loss + self.perceptual_weight * p_loss

            nll_loss = nll_loss / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

            kl_loss = posteriors.kl()
            kl_loss = torch.mean(kl_loss)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            if disc_factor > 0.0:
                if self.using_3d_discriminator:
                    reconstructions = rearrange(reconstructions, '(b t) c h w -> b c t h w', t=t)

                logits_fake = self.discriminator(reconstructions.contiguous())
                g_loss = -torch.mean(logits_fake)
                try:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0)

            
            loss = (
                weighted_nll_loss
                + self.kl_weight * kl_loss
                + d_weight * disc_factor * g_loss
            )
            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/logvar".format(split): self.logvar.detach(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/perception_loss".format(split): p_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            return loss, log

        if optimizer_idx == 1:
            if self.using_3d_discriminator:
                inputs = rearrange(inputs, '(b t) c h w -> b c t h w', t=t)
                reconstructions = rearrange(reconstructions, '(b t) c h w -> b c t h w', t=t)

            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log
