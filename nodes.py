# nodes.py
import torch
import torchaudio
import numpy as np
import os
import folder_paths
from pathlib import Path

class SoproTTSNode:
    """
    Sopro TTS Node - Lightweight CPU-based Text-to-Speech with zero-shot voice cloning
    """
    
    def __init__(self):
        self.model = None
        self.device = "cpu"  # Sopro runs efficiently on CPU
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is a test of the Sopro text to speech system."
                }),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.5,
                    "step": 0.1,
                    "display": "slider"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/generation"
    OUTPUT_NODE = False
    
    def load_model(self):
        """Lazy load the Sopro model"""
        if self.model is None:
            try:
                from sopro import Sopro
                print("Loading Sopro TTS model...")
                self.model = Sopro()
                print("Sopro TTS model loaded successfully!")
            except ImportError:
                raise ImportError(
                    "Sopro is not installed. Please install it with: "
                    "pip install git+https://github.com/samuel-vitorino/sopro.git"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load Sopro model: {str(e)}")
        return self.model
    
    def preprocess_text(self, text):
        """Clean and preprocess text for better TTS results"""
        # Convert numbers and symbols to words as recommended
        text = text.strip()
        # Add basic preprocessing
        replacements = {
            " + ": " plus ",
            " - ": " minus ",
            " = ": " equals ",
            " & ": " and ",
            " @ ": " at ",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    def generate_speech(self, text, reference_audio=None, speed=1.0, temperature=0.7, seed=0):
        """Generate speech from text using Sopro TTS"""
        
        # Set seed for reproducibility
        if seed > 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Load model
        model = self.load_model()
        
        # Preprocess text
        text = self.preprocess_text(text)
        
        if not text:
            raise ValueError("Text input cannot be empty")
        
        try:
            # Generate audio
            if reference_audio is not None:
                # Voice cloning mode with reference audio
                ref_waveform = reference_audio['waveform']
                ref_sample_rate = reference_audio['sample_rate']
                
                # Convert to numpy and ensure correct format
                if isinstance(ref_waveform, torch.Tensor):
                    # ComfyUI format: (batch, channels, samples)
                    ref_waveform = ref_waveform[0].cpu().numpy()
                
                # Convert to mono if stereo
                if ref_waveform.shape[0] > 1:
                    ref_waveform = ref_waveform.mean(axis=0, keepdims=True)
                
                # Resample if needed (Sopro typically uses 24kHz)
                if ref_sample_rate != 24000:
                    ref_waveform_tensor = torch.from_numpy(ref_waveform).float()
                    ref_waveform_tensor = torchaudio.functional.resample(
                        ref_waveform_tensor, 
                        ref_sample_rate, 
                        24000
                    )
                    ref_waveform = ref_waveform_tensor.numpy()
                
                # Generate with voice cloning
                audio = model.generate(
                    text=text,
                    reference_audio=ref_waveform.squeeze(),
                    speed=speed,
                    temperature=temperature
                )
            else:
                # Standard TTS mode
                audio = model.generate(
                    text=text,
                    speed=speed,
                    temperature=temperature
                )
            
            # Convert to torch tensor
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = audio.float()
            
            # Ensure correct shape: (batch, channels, samples)
            if audio_tensor.dim() == 1:
                # (samples,) -> (1, 1, samples)
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif audio_tensor.dim() == 2:
                # (channels, samples) -> (1, channels, samples)
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Normalize audio to [-1, 1]
            if audio_tensor.abs().max() > 1.0:
                audio_tensor = audio_tensor / audio_tensor.abs().max()
            
            # Return in ComfyUI audio format
            return ({
                "waveform": audio_tensor,
                "sample_rate": 24000
            },)
            
        except Exception as e:
            raise RuntimeError(f"Error generating speech: {str(e)}")


class SoproLoadReferenceAudio:
    """Load reference audio file for voice cloning"""
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.exists(input_dir):
            files = [f for f in os.listdir(input_dir) 
                    if f.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
        
        return {
            "required": {
                "audio_file": (sorted(files),),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("reference_audio",)
    FUNCTION = "load_audio"
    CATEGORY = "audio/loading"
    
    def load_audio(self, audio_file):
        """Load audio file and return in ComfyUI format"""
        input_dir = folder_paths.get_input_directory()
        audio_path = os.path.join(input_dir, audio_file)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio with torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Ensure batch dimension: (channels, samples) -> (1, channels, samples)
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
            
            return ({
                "waveform": waveform,
                "sample_rate": sample_rate
            },)
            
        except Exception as e:
            raise RuntimeError(f"Error loading audio file: {str(e)}")


class SoproSaveAudio:
    """Save generated audio to file"""
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "sopro_audio"}),
            },
            "optional": {
                "format": (["wav", "mp3", "flac"],),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "audio/output"
    
    def save_audio(self, audio, filename_prefix="sopro_audio", format="wav"):
        """Save audio to file"""
        
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']
        
        # Remove batch dimension for saving
        if waveform.dim() == 3:
            waveform = waveform[0]
        
        # Generate unique filename
        counter = 0
        while True:
            filename = f"{filename_prefix}_{counter:05d}.{format}"
            filepath = os.path.join(self.output_dir, filename)
            if not os.path.exists(filepath):
                break
            counter += 1
        
        try:
            # Save audio
            torchaudio.save(filepath, waveform.cpu(), sample_rate)
            print(f"Audio saved to: {filepath}")
            
            return {"ui": {"audio": [filename]}}
            
        except Exception as e:
            raise RuntimeError(f"Error saving audio: {str(e)}")


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "SoproTTSNode": SoproTTSNode,
    "SoproLoadReferenceAudio": SoproLoadReferenceAudio,
    "SoproSaveAudio": SoproSaveAudio,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "SoproTTSNode": "Sopro TTS Generator",
    "SoproLoadReferenceAudio": "Sopro Load Reference Audio",
    "SoproSaveAudio": "Sopro Save Audio",
}
