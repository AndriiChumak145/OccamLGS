import numpy as np
import torch
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch.nn.functional as F
from pathlib import Path

import sys
from pathlib import Path

# Resolve both the workspace root and the NAF directory
workspace_root = Path(__file__).resolve().parents[2]
naf_root = workspace_root / "NAF"

# 1. Allows your code to look inside 'NAF' as a clean namespace
if str(workspace_root) not in sys.path:
    sys.path.append(str(workspace_root))

# 2. Unblocks NAF's internal code when it looks for 'src' absolute folders
if str(naf_root) not in sys.path:
    sys.path.append(str(naf_root))

# Both your import and NAF's internal cascade will now resolve cleanly
from NAF.src.model.naf import NAF


class OnlineSiglipExtractor:
    def __init__(self, model_id="google/siglip2-giant-opt-patch16-384", device="cuda", upsample_method="bilinear"):
        print(f"\nInitializing Online SigLIP Extractor: {model_id}...")
        print(f"Upsampling Strategy: {upsample_method.upper()}")
        
        self.device = device
        self.upsample_method = upsample_method.lower()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        self.model.eval()
        self.patch_size = 16
        
        if self.upsample_method == "naf":
            # OFFICIAL REPO METHOD: This correctly pulls the pre-trained weights!
            self.naf = torch.hub.load("valeoai/NAF", "naf", pretrained=True, device=self.device)
            self.naf.eval()

    def extract(self, view):
        """
        Extracts dense 1536-D features directly from a 3DGS camera view in VRAM.
        Returns: (feature_tensor, mask_tensor) perfectly formatted for OccamLGS.
        """
        with torch.no_grad():
            # view.original_image is a torch Tensor [3, H, W] in range [0, 1]
            orig_tensor = view.original_image
            C, orig_H, orig_W = orig_tensor.shape
            
            # Convert to PIL for the HuggingFace processor
            img_np = (orig_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            inputs = self.processor(images=img_pil, return_tensors="pt")
            
            # Move inputs to GPU and cast to float16 to match the model
            pixel_values = inputs['pixel_values'].to(self.device, dtype=torch.float16)

            outputs = self.model.vision_model(pixel_values=pixel_values)
            
            patch_embeds = outputs.last_hidden_state 
            actual_dim = patch_embeds.shape[-1]
            
            grid_h = pixel_values.shape[-2] // self.patch_size
            grid_w = pixel_values.shape[-1] // self.patch_size
            
            spatial_features = patch_embeds.permute(0, 2, 1).reshape(1, actual_dim, grid_h, grid_w)
            
            # Feature upscaling to match the original image resolution
            if self.upsample_method == "naf":
                # NAF requires the HR RGB image to guide the edges. Format expected: [1, 3, H, W]
                hr_image = orig_tensor.unsqueeze(0).to(self.device)
                
                # Chunking purely to prevent VRAM overflow. 
                # 256 is at the limit of what RTX5090 can handle with giant SigLIP
                chunk_size = 256 
                dense_chunks = []
                
                for i in range(0, actual_dim, chunk_size):
                    chunk = spatial_features[:, i:i+chunk_size, :, :]
                    
                    # NAF Official Forward Pass: naf(image, lr_features, target_size)
                    dense_chunk = self.naf(hr_image, chunk.float(), (orig_H, orig_W)).squeeze(0)
                    
                    # Move to CPU to save VRAM
                    dense_chunks.append(dense_chunk.cpu())
                    
                dense_features = torch.cat(dense_chunks, dim=0).to(self.device)
                
            else:
                # Default Bilinear Interpolation
                dense_features = F.interpolate(
                    spatial_features.float(), 
                    size=(orig_H, orig_W), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)

            # NOTE: For now, we want features everywhere, so mask is all 1s.
            # Shape expected by OccamLGS: [1, H, W]
            weight_map = torch.ones((1, orig_H, orig_W), dtype=torch.bool, device=self.device)
            
            torch.cuda.empty_cache()
            
            return dense_features.to(self.device), weight_map