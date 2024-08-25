
# Diffusion Model Video Generation

## Introduction
In this project, we explore the power of diffusion models for image generation and their application in creating videos. Diffusion models are a class of generative models that have recently gained prominence due to their ability to generate high-quality, realistic images. By leveraging the capabilities of these models, this project focuses on generating a series of images and compiling them into a coherent video. The entire process is implemented in Python using a diffusion model from Hugging Face, bypassing the need for API calls, thus enabling more direct and customizable control over the model.

## Objective
The primary objective of this project is to harness the potential of diffusion models to create a dynamic and visually appealing video from a sequence of generated images. This involves:

- Generating a specified number of images using a pre-trained diffusion model.
- Ensuring the images are thematically and aesthetically coherent to form a fluid sequence.
- Compiling the generated images into a video, showcasing the capabilities of diffusion models in creative content generation.

## Applications
The application of this project extends across various domains, including but not limited to:

- **Creative Content Creation**: Artists and designers can use this tool to generate unique visuals and animations for digital media.
- **Advertising**: Marketers can create engaging video content that captures the viewer's attention through innovative visuals.
- **Film and Animation**: Filmmakers and animators can experiment with AI-generated sequences to inspire new ideas or enhance existing projects.
- **Education**: This project can be used as a learning tool to demonstrate the principles of diffusion models and their practical applications.

## Technology Used
The project leverages the following technologies and tools:

- **Python**: The primary programming language used to implement the model and manage the workflow.
- **Diffusion Models**: A pre-trained diffusion model is utilized to generate images. This model is accessed directly from Hugging Face without relying on the API, allowing for greater customization and control.
- **Hugging Face Transformers Library**: Used to load and interact with the diffusion model in a Python environment.
- **FFmpeg**: A powerful multimedia framework used to compile the sequence of images into a video file.
- **NumPy and Matplotlib**: These libraries assist in handling image data and visualizing intermediate results during the image generation process.
## Results 
- **Prompt-1**:"blueberry spaghetti","strawberry spaghetti".



https://github.com/user-attachments/assets/eae9761a-4618-420c-a9be-e77b5222daae





- **Prompt-2**:"ultrarealistic steam punk neural network machine in the shape of a brain, placed on a pedestal, covered with neurons made of gears.","dramatic lighting"


https://github.com/user-attachments/assets/dd7d5ae9-ef57-4764-ade3-2778749c96bb


- **Prompt-3**:"ultrarealistic steam punk neural network machine in the shape of a brain, placed on a pedestal, covered with neurons made of gears.", "A meticulously detailed steampunk-style brain-shaped neural network device, set upon a pedestal and featuring neurons constructed from intricately designed gears."

  

https://github.com/user-attachments/assets/51a7188b-3be8-4995-8698-ae122ef047ae



## Credits 
- The code gist was inspired by Andrej karpathy [gist](https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355).
- This code have been updated because there were some parts not working and used different diffusion model **stabilityai/stable-diffusion-2-1-base**.The original gist have different diffusion model.This code was edited and run by [Akarshan Gupta](https://github.com/AkarshanGupta/AkarshanGupta).

## How to Run 
#### Steps
```bash
Open Google Collab
pip install fire diffusers. If other libraries are not install them 
Connect with a GPU Runtime
Enter the code from diffuser.py
Folder will be created in the folder section of Collab.
```












  
