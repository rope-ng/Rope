import os
import cv2
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from skimage import transform as trans
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from math import ceil
import onnxruntime

from rope.Models import Models


class FaceSwapProcessor:
    def __init__(self):
        self.models = Models()
        self.device = self.models.device
        self.arcface_dst = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
    
    def base64_to_image(self, base64_string):
        """Convert base64 string to numpy image array (RGB)"""
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        image = image.convert('RGB')
        return np.array(image)
    
    def image_to_base64(self, image_array):
        """Convert numpy image array (RGB) to base64 string"""
        if isinstance(image_array, torch.Tensor):
            image_array = image_array.detach().cpu().numpy()
            if image_array.ndim == 3 and image_array.shape[0] == 3:
                image_array = image_array.transpose(1, 2, 0)
        
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        image = Image.fromarray(image_array)
        
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return image_base64
    
    def detect_faces(self, image, detect_mode='Retinaface', max_num=10, score=0.5):
        """Detect faces in image and return keypoints"""
        return self.models.run_detect(image, detect_mode=detect_mode, max_num=max_num, score=score)
    
    def get_face_embeddings(self, image, face_keypoints):
        """Get embeddings for detected faces"""
        embeddings = []
        cropped_faces = []
        
        for kps in face_keypoints:
            embedding, cropped = self.models.run_recognize(image, kps)
            embeddings.append(embedding)
            cropped_faces.append(cropped)
            
        return embeddings, cropped_faces
    
    def get_default_parameters(self):
        """Get default parameters for face swapping"""
        return {
            'SwapperTypeTextSel': '128',
            'FaceAdjSwitch': False,
            'KPSXSlider': 0,
            'KPSYSlider': 0,
            'KPSScaleSlider': 0,
            'FaceScaleSlider': 0,
            'StrengthSwitch': False,
            'StrengthSlider': 100,
            'DiffSlider': 0,
            'DiffSwitch': False,
            'MouthParserSwitch': False,
            'FaceParserSwitch': False,
            'OccluderSwitch': False,
            'BlurSwitch': False,
            'BorderSwitch': False,
            'ColorSwitch': False,
            'RestorerSwitch': False,
            'UpscaleSwitch': False
        }
    
    def swap_face_core(self, img, target_kps, source_embedding, parameters=None):
        """Core face swapping function - adapted from VideoManager.swap_core"""
        if parameters is None:
            parameters = self.get_default_parameters()
        
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float().to(self.device)
            if img.ndim == 3 and img.shape[2] == 3:
                img = img.permute(2, 0, 1)
        
        dst = self.arcface_dst * 4.0
        dst[:,0] += 32.0
        
        if parameters.get('FaceAdjSwitch', False):
            dst[:,0] += parameters.get('KPSXSlider', 0)
            dst[:,1] += parameters.get('KPSYSlider', 0)
            dst[:,0] -= 255
            dst[:,0] *= (1 + parameters.get('KPSScaleSlider', 0) / 100)
            dst[:,0] += 255
            dst[:,1] -= 255
            dst[:,1] *= (1 + parameters.get('KPSScaleSlider', 0) / 100)
            dst[:,1] += 255
        
        tform = trans.SimilarityTransform()
        tform.estimate(target_kps, dst)
        
        t512 = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t256 = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t128 = v2.Resize((128, 128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        
        rotation_deg = float(tform.rotation * 57.2958)
        translation = [float(tform.translation[0]), float(tform.translation[1])]
        scale_val = float(tform.scale)
        original_face_512 = v2.functional.affine(img, rotation_deg, translation, scale_val, [0], center=[0, 0], interpolation=v2.InterpolationMode.BILINEAR)
        original_face_512 = v2.functional.crop(original_face_512, 0, 0, 512, 512)
        original_face_256 = t256(original_face_512)
        original_face_128 = t128(original_face_256)
        
        latent = torch.from_numpy(self.models.calc_swapper_latent(source_embedding)).float().to(self.device)
        
        dim = 1
        swapper_type = parameters.get('SwapperTypeTextSel', '128')
        if swapper_type == '128':
            dim = 1
            input_face_affined = original_face_128
        elif swapper_type == '256':
            dim = 2
            input_face_affined = original_face_256
        else:
            dim = 4
            input_face_affined = original_face_512
        
        if parameters.get('FaceAdjSwitch', False):
            scale_factor = 1 + parameters.get('FaceScaleSlider', 0) / 100
            center_val = [float(dim*128-1), float(dim*128-1)]
            input_face_affined = v2.functional.affine(input_face_affined, 0, [0.0, 0.0], scale_factor, [0], center=center_val, interpolation=v2.InterpolationMode.BILINEAR)
        
        itex = 1
        if parameters.get('StrengthSwitch', False):
            itex = ceil(parameters.get('StrengthSlider', 100) / 100.)
        
        output_size = int(128 * dim)
        output = torch.zeros((output_size, output_size, 3), dtype=torch.float32, device=self.device)
        input_face_affined = input_face_affined.permute(1, 2, 0)
        input_face_affined = torch.div(input_face_affined, 255.0)
        
        prev_face = None
        for k in range(itex):
            for j in range(dim):
                for i in range(dim):
                    input_face_disc = input_face_affined[j::dim, i::dim]
                    input_face_disc = input_face_disc.permute(2, 0, 1)
                    input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                    
                    swapper_output = torch.empty((1,3,128,128), dtype=torch.float32, device=self.device).contiguous()
                    self.models.run_swapper(input_face_disc, latent, swapper_output)
                    
                    swapper_output = torch.squeeze(swapper_output)
                    swapper_output = swapper_output.permute(1, 2, 0)
                    
                    output[j::dim, i::dim] = swapper_output.clone()
            
            prev_face = input_face_affined.clone()
            input_face_affined = output.clone()
            output = torch.mul(output, 255)
            output = torch.clamp(output, 0, 255)
        
        output = output.permute(2, 0, 1)
        swap = t512(output)
        
        if parameters.get('StrengthSwitch', False) and itex > 0 and prev_face is not None:
            alpha = np.mod(parameters.get('StrengthSlider', 100), 100) * 0.01
            if alpha == 0:
                alpha = 1
            
            prev_face = torch.mul(prev_face, 255)
            prev_face = torch.clamp(prev_face, 0, 255)
            prev_face = prev_face.permute(2, 0, 1)
            prev_face = t512(prev_face)
            swap = torch.mul(swap, alpha)
            prev_face = torch.mul(prev_face, 1-alpha)
            swap = torch.add(swap, prev_face)
        
        return swap, tform
    
    def blend_face_back(self, original_img, swapped_face, tform):
        """Blend the swapped face back into the original image"""
        # Convert to tensors if needed
        if isinstance(original_img, np.ndarray):
            original_img = torch.from_numpy(original_img).float().to(self.device)
            if original_img.ndim == 3 and original_img.shape[2] == 3:
                original_img = original_img.permute(2, 0, 1)
        
        inv_tform = trans.SimilarityTransform()
        inv_tform.params = np.linalg.inv(tform.params)
        
        inv_rotation = -float(tform.rotation * 57.2958)
        inv_translation = [-float(tform.translation[0]/tform.scale), -float(tform.translation[1]/tform.scale)]
        inv_scale = float(1/tform.scale)
        swapped_face_positioned = v2.functional.affine(
            swapped_face, 
            inv_rotation, 
            inv_translation, 
            inv_scale, 
            [0], 
            center=[0, 0], 
            interpolation=v2.InterpolationMode.BILINEAR
        )
        
        result = original_img.clone()
        h, w = swapped_face_positioned.shape[1], swapped_face_positioned.shape[2]
        if h <= result.shape[1] and w <= result.shape[2]:
            result[:, :h, :w] = swapped_face_positioned
        
        return result
    
    def process_face_swap(self, input_image_b64, source_faces_b64, parameters=None):
        """Main function to process face swap from base64 inputs"""
        try:
            input_img = self.base64_to_image(input_image_b64)
            
            source_embeddings = []
            for face_b64 in source_faces_b64:
                face_img = self.base64_to_image(face_b64)
                face_kps = self.detect_faces(face_img, max_num=1)
                if len(face_kps) > 0:
                    embeddings, _ = self.get_face_embeddings(face_img, face_kps)
                    if len(embeddings) > 0:
                        source_embeddings.append(embeddings[0])
            
            if not source_embeddings:
                return {
                    "status": "error",
                    "message": "No faces detected in source images",
                    "result": None
                }
            
            target_faces = self.detect_faces(input_img)
            
            if not target_faces:
                return {
                    "status": "error", 
                    "message": "No faces detected in input image",
                    "result": None
                }
            
            input_tensor = torch.from_numpy(input_img).float().to(self.device)
            input_tensor = input_tensor.permute(2, 0, 1)
            
            result_img = input_tensor.clone()
            source_embedding = source_embeddings[0]  # Use first source face
            
            for target_kps in target_faces:
                swapped_face, tform = self.swap_face_core(input_tensor, target_kps, source_embedding, parameters)
                result_img = swapped_face
                break
            
            result_b64 = self.image_to_base64(result_img)
            
            return {
                "status": "success",
                "message": f"Successfully swapped {len(target_faces)} face(s)",
                "result": result_b64,
                "faces_detected": len(target_faces),
                "source_faces_used": len(source_embeddings)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Face swap processing failed: {str(e)}",
                "result": None
            }