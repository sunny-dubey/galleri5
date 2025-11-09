# Ranveer Singh Image Editing with LoRA Training

This project implements a two-stage LoRA (Low-Rank Adaptation) training pipeline for fine-tuning the Qwen-Image-Edit-2509 model to perform person replacement with Ranveer Singh using the DiffSynth-Studio framework.

## Framework

- **Framework**: [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- **Base Model**: Qwen/Qwen-Image-Edit-2509
- **Training Method**: Two-stage LoRA fine-tuning

## Dataset

The training dataset consists of images of Ranveer Singh with corresponding text descriptions stored in `metadata.json`. The dataset includes:
- Training images: Multiple images of Ranveer Singh in various poses, outfits, and settings
- Metadata format: JSON file with `image` and `prompt` fields
- Dataset location: `data/ranveer_singh/`

## Generated Images

The trained model has been used to generate the following images:
- `ranveer_singh_in_lab.jpg`
- `ranveer_singh_in_mahabharat.jpg`
- `ranveer_singh_in_office.jpg`

## Edit Images

The following images were used as input for person replacement:
- `sunny.jpeg`
- `rahul.jpeg`
- `ranbir.jpg`
- `sunny_face.jpeg`
- `sunny_full.jpeg`

## Training Process

### Stage 1: Data Processing

The first stage processes the dataset and prepares it for training:

```bash
accelerate launch DiffSynth-Studio/examples/qwen_image/model_training/train.py \
  --dataset_base_path data/ranveer_singh \
  --dataset_metadata_path data/ranveer_singh/metadata.json \
  --data_file_keys "image" \
  --max_pixels 1048576 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2509:text_encoder/model*.safetensors,Qwen/Qwen-Image-Edit-2509:vae/diffusion_pytorch_model.safetensors" \
  --output_path "./models/ranveer_cache" \
  --task data_process \
  --dataset_num_workers 8
```

**Configuration:**
- Max pixels: 1,048,576
- Task: `data_process`
- Dataset workers: 8

### Stage 2: LoRA Fine-tuning

The second stage performs the actual LoRA training:

```bash
accelerate launch DiffSynth-Studio/examples/qwen_image/model_training/train.py \
  --dataset_base_path models/ranveer_cache \
  --max_pixels 786432 \
  --dataset_repeat 15 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/ranveer_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 64 \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --gradient_accumulation_steps 8 \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --enable_fp8_training \
  --task sft \
  --save_steps 2000
```

**LoRA Configuration:**
- **LoRA Rank**: 64
- **LoRA Base Model**: `dit`
- **LoRA Target Modules**: `to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1`
- **Learning Rate**: 1e-4
- **Epochs**: 5
- **Dataset Repeat**: 15
- **Max Pixels**: 786,432
- **Gradient Accumulation Steps**: 8
- **Save Steps**: 2000
- **Optimizations**:
  - Gradient checkpointing enabled
  - Gradient checkpointing offload enabled
  - FP8 training enabled
  - Find unused parameters enabled

## Inference

The inference pipeline loads the trained LoRA weights and performs image editing:

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from PIL import Image
import torch

# Load the base model
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="tokenizer/"),
    processor_config=None
)

# Load processor
from transformers import Qwen2VLProcessor
pipe.processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)

# Load trained LoRA
pipe.load_lora(pipe.dit, "./models/train/ranveer_lora/step-100.safetensors")

# Edit an image
edit_image = Image.open("images.jpg")
output = pipe(
    prompt="Replace the person with Ranveer Singh, maintaining the exact same pose, neutral contemplative expression with leftward gaze, dusty pink crew-neck t-shirt, outdoor setting with blurred green foliage background, and soft natural lighting, while ensuring Ranveer Singh has his characteristic full thick beard covering chin and jawline, thick mustache connecting to the beard, strong jawline, prominent cheekbones, dark expressive eyes, prominent eyebrows, medium to tan skin tone, and confident masculine features as seen in training data (metadata.json images 1-103).",
    edit_image=edit_image,
    seed=42,
    num_inference_steps=60,
    cfg_scale=9.0
)

output.save("output.jpg")
```

**Inference Parameters:**
- **Num Inference Steps**: 60
- **CFG Scale**: 9.0
- **Seed**: 42
- **Device**: CUDA
- **Precision**: bfloat16

## Setup

1. Clone DiffSynth-Studio:
```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

2. Install dependencies:
```bash
pip install accelerate
pip install hf_transfer
```

3. Prepare your dataset in the format:
```
data/ranveer_singh/
├── metadata.json
├── image1.jpg
├── image2.jpg
└── ...
```

## Results

The trained LoRA model successfully replaces persons in images with Ranveer Singh while maintaining:
- Original pose and composition
- Background and lighting
- Clothing and scene context
- Natural integration of Ranveer Singh's distinctive features (full beard, facial structure, etc.)

## References

- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)

