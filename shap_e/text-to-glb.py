import torch
import trimesh
import numpy as np
import tempfile

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on device (cpu - bad, cuda - good):", device)
print("Downloading or loading transmitter")
xm = load_model('transmitter', device=device)
print("transmitter ready")

print("Downloading or loading text300M")
model = load_model('text300M', device=device)
print("text300M loaded")

print("Downloading or loading diffusion")
diffusion = diffusion_from_config(load_config('diffusion'))
print("diffusion loaded")

batch_size = 4
guidance_scale = 15.0


def generate(prompt, name):
  latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
  )

  from shap_e.util.notebooks import decode_latent_mesh

  for i, latent in enumerate(latents):
    """t = decode_latent_mesh(xm, latent).tri_mesh()
    ply_path = f'output/{name}_{i}.ply'
    with open(ply_path, 'wb') as f:
      t.write_ply(f)"""

    ply_path = tempfile.NamedTemporaryFile(suffix='.ply',
                                               delete=True,
                                               mode='w+b')
    decode_latent_mesh(xm, latent).tri_mesh().write_ply(ply_path.file)
    ply_path.seek(0)
    ply_mesh = trimesh.load(ply_path.name)
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    ply_mesh = ply_mesh.apply_transform(rot)
    rot = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
    ply_mesh = ply_mesh.apply_transform(rot)

    mesh_path = f'output/{name}_{i}.glb'
    ply_mesh.export(mesh_path, file_type='glb')
    ply_path.close()


while(True):
  prompt = input("Prompt:")
  name = input("File name:")
  generate(prompt, name)
  print("\n")