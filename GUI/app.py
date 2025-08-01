# -*- coding: utf-8 -*-
import os
import inspect
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
from torch import autocast
import cv2
import gradio as gr

# -----------------------------------------------------------------------------
# 1. REQUIREMENTS & SETUP
# -----------------------------------------------------------------------------
# To set up the environment for this script, create a file named 'requirements.txt'
# with the following content and run 'pip install -r requirements.txt':
#
# torch
# torchvision
# diffusers
# transformers
# accelerate
# gradio
# opencv-python-headless
# -----------------------------------------------------------------------------

# --- Automatic Device Detection ---
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print("-------------------------------------------------")
print(f"INFO: Using device: {torch_device.upper()}")
if torch_device == "cpu":
    print("WARNING: CUDA (GPU) not detected. The script will run on the CPU.")
    print("         This will be extremely slow. For better performance,")
    print("         please ensure you have an NVIDIA GPU and the correct")
    print("         PyTorch version with CUDA support installed.")
print("-------------------------------------------------")


# --- Load the Model ---
print("Loading Stable Diffusion model... This may take a moment.")
try:
    # Load the pipeline and move it to the detected device
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
    pipe.to(torch_device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check your internet connection and ensure the model name is correct.")
    exit()

# -----------------------------------------------------------------------------
# Helper Functions (slerp, diffuse)
# -----------------------------------------------------------------------------

@torch.no_grad()
def diffuse(
        pipe, cond_embeddings, cond_latents, num_inference_steps, guidance_scale, eta, device
    ):
    # The 'device' is now passed explicitly to this function
    max_length = cond_embeddings.shape[1]
    uncond_input = pipe.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
    # Use the passed 'device' variable for all tensor placement
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

    if "LMS" in pipe.scheduler.__class__.__name__:
        cond_latents = cond_latents * pipe.scheduler.sigmas[0]

    accepts_offset = "offset" in set(inspect.signature(pipe.scheduler.set_timesteps).parameters.keys())
    extra_set_kwargs = {}
    if accepts_offset:
        extra_set_kwargs["offset"] = 1
    pipe.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

    accepts_eta = "eta" in set(inspect.signature(pipe.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    for i, t in enumerate(pipe.scheduler.timesteps):
        latent_model_input = torch.cat([cond_latents] * 2)
        if "LMS" in pipe.scheduler.__class__.__name__:
            sigma = pipe.scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        # predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        cond_latents = pipe.scheduler.step(noise_pred, t, cond_latents, **extra_step_kwargs)["prev_sample"]

    cond_latents = 1 / 0.18215 * cond_latents
    image = pipe.vae.decode(cond_latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).astype(np.uint8)
    return image

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    # This function is device-agnostic
    inputs_are_torch = isinstance(v0, torch.Tensor)
    if inputs_are_torch:
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()
    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1
    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)
    return v2

# -----------------------------------------------------------------------------
# Main Generator Function for Gradio
# -----------------------------------------------------------------------------
def generate_dream_video(
    prompt_1, prompt_2, seed_1, seed_2,
    width, height, num_steps, guidance_scale,
    num_inference_steps, eta, name
):
    # --- 1. SETUP ---
    yield {
        status_text: "Status: Preparing prompts and latents...",
        live_frame: None,
        output_video: None,
    }
    prompts = [prompt_1, prompt_2]
    seeds = [int(seed_1), int(seed_2)]
    rootdir = './dreams'
    outdir = os.path.join(rootdir, name)
    os.makedirs(outdir, exist_ok=True)
    
    # --- 2. EMBEDDINGS AND LATENTS ---
    prompt_embeddings = []
    for prompt in prompts:
        text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embed = pipe.text_encoder(text_input.input_ids.to(torch_device))[0]
        prompt_embeddings.append(embed)

    prompt_embedding_a, prompt_embedding_b = prompt_embeddings
    
    generator_a = torch.Generator(device=torch_device).manual_seed(seeds[0])
    generator_b = torch.Generator(device=torch_device).manual_seed(seeds[1])

    init_a = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8), device=torch_device, generator=generator_a)
    init_b = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8), device=torch_device, generator=generator_b)

    # --- 3. GENERATION LOOP ---
    frame_paths = []
    for i, t in enumerate(np.linspace(0, 1, num_steps)):
        yield {
            status_text: f"Status: Generating frame {i + 1} of {num_steps} on {torch_device.upper()}...",
            live_frame: None,
            output_video: None,
        }
        
        cond_embedding = slerp(float(t), prompt_embedding_a, prompt_embedding_b)
        init = slerp(float(t), init_a, init_b)

        # Use autocast only if on CUDA
        with autocast(torch_device) if torch_device == "cuda" else open(os.devnull, 'w') as f:
            # Pass the torch_device explicitly to the diffuse function
            image = diffuse(pipe, cond_embedding, init, num_inference_steps, guidance_scale, eta, torch_device)

        im = Image.fromarray(image)
        outpath = os.path.join(outdir, f'frame{i:06d}.jpg')
        im.save(outpath)
        frame_paths.append(outpath)
        
        yield { live_frame: im }

    # --- 4. VIDEO COMPILATION ---
    yield { status_text: "Status: Compiling video from frames..." }
    
    video_path = os.path.join(outdir, f"{name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 15, (width, height))
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
    video_writer.release()
    
    print(f"Video saved to {video_path}")
    yield {
        status_text: f"Status: Done! Video saved to {video_path}",
        output_video: video_path
    }

# -----------------------------------------------------------------------------
# Gradio UI (Unchanged)
# -----------------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(), css="footer {display: none !important}") as demo:
    gr.Markdown("# 🎥 Stable Diffusion Video Interpolation")
    gr.Markdown("Create smooth transition videos between two concepts. Configure the prompts and settings below, then click Generate.")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Accordion("1. Core Prompts & Seeds", open=True):
                prompt_1 = gr.Textbox(lines=2, label="Starting Prompt", value="ultrarealistic steam punk neural network machine in the shape of a brain, placed on a pedestal, covered with neurons made of gears.")
                seed_1 = gr.Number(label="Seed 1", value=243, precision=0, info="A specific number to control the starting noise pattern.")
                prompt_2 = gr.Textbox(lines=2, label="Ending Prompt", value="A bioluminescent, glowing jellyfish floating in a dark, deep abyss, surrounded by sparkling plankton.")
                seed_2 = gr.Number(label="Seed 2", value=523, precision=0, info="A specific number to control the ending noise pattern.")
                name = gr.Textbox(label="Output File Name", value="my_dream_video", info="The name for the output folder and .mp4 file.")
            
            with gr.Accordion("2. Generation Parameters", open=True):
                with gr.Row():
                    width = gr.Slider(label="Width", minimum=256, maximum=1024, value=512, step=64)
                    height = gr.Slider(label="Height", minimum=256, maximum=1024, value=512, step=64)
                num_steps = gr.Slider(label="Interpolation Frames", minimum=10, maximum=500, value=120, step=1, info="How many frames the final video will have. More frames = smoother video.")

            with gr.Accordion("3. Advanced Diffusion Settings", open=False):
                 num_inference_steps = gr.Slider(label="Inference Steps per Frame", minimum=10, maximum=100, value=40, step=1, info="More steps can improve quality but will be much slower.")
                 guidance_scale = gr.Slider(label="Guidance Scale (CFG)", minimum=1, maximum=20, value=7.5, step=0.5, info="How strongly the prompt guides the image generation.")
                 eta = gr.Slider(label="ETA (for DDIM Scheduler)", minimum=0.0, maximum=1.0, value=0.0, step=0.1, info="A parameter for noise scheduling. 0.0 is deterministic.")

            run_button = gr.Button("Generate Video", variant="primary")

        with gr.Column(scale=3):
            status_text = gr.Textbox(label="Status", value="Ready", interactive=False)
            live_frame = gr.Image(label="Live Preview", type="pil")
            output_video = gr.Video(label="Final Video")

    run_button.click(
        fn=generate_dream_video,
        inputs=[
            prompt_1, prompt_2, seed_1, seed_2,
            width, height, num_steps, guidance_scale,
            num_inference_steps, eta, name
        ],
        outputs=[status_text, live_frame, output_video]
    )

# --- Launch the App ---
if __name__ == "__main__":
    demo.launch(share=True, debug=True)
