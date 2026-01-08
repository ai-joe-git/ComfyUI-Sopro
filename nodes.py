# nodes.py
import torch
import torchaudio
import soundfile as sf
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
        self.device = "cpu"
        
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
                    "max": 2147483647
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
                print("Loading Sopro TTS model...")
                
                # Import and instantiate SoproTTS with minimal args
                from sopro import SoproTTS
                
                # Initialize model with default settings - it will auto-download
                self.model = SoproTTS(device=self.device)
                
                print("Sopro TTS model loaded successfully!")
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(
                    f"Failed to load Sopro model: {str(e)}\n"
                    "Make sure Sopro is installed: pip install git+https://github.com/samuel-vitorino/sopro.git"
                )
        return self.model
    
    def preprocess_text(self, text):
        """Clean and preprocess text for better TTS results"""
        text = text.strip()
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
            seed = min(seed, 2147483647)
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Load model
        model = self.load_model()
        
        # Preprocess text
        text = self.preprocess_text(text)
        
        if not text:
            raise ValueError("Text input cannot be empty")
        
        try:
            # Prepare reference audio if provided
            ref_audio_np = None
            if reference_audio is not None:
                ref_waveform = reference_audio['waveform']
                ref_sample_rate = reference_audio['sample_rate']
                
                if isinstance(ref_waveform, torch.Tensor):
                    ref_waveform = ref_waveform[0].cpu().numpy()
                
                if ref_waveform.shape[0] > 1:
                    ref_waveform = ref_waveform.mean(axis=0, keepdims=True)
                
                # Resample to 24kHz if needed
                if ref_sample_rate != 24000:
                    ref_waveform_tensor = torch.from_numpy(ref_waveform).float()
                    ref_waveform_tensor = torchaudio.functional.resample(
                        ref_waveform_tensor, ref_sample_rate, 24000
                    )
                    ref_waveform = ref_waveform_tensor.numpy()
                
                ref_audio_np = ref_waveform.squeeze()
            
            # Generate speech
            audio_output = model.infer(
                text=text,
                reference_audio=ref_audio_np,
                speed=speed,
                temperature=temperature
            )
            
            # Convert to torch tensor if needed
            if isinstance(audio_output, np.ndarray):
                audio_tensor = torch.from_numpy(audio_output).float()
            else:
                audio_tensor = audio_output.float()
            
            # Ensure correct shape: (batch, channels, samples)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Normalize audio to [-1, 1]
            if audio_tensor.abs().max() > 1.0:
                audio_tensor = audio_tensor / audio_tensor.abs().max()
            
            return ({
                "waveform": audio_tensor,
                "sample_rate": 24000
            },)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Error generating speech: {str(e)}")


class SoproLoadReferenceAudio:
    """Load reference audio file for voice cloning using soundfile"""
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.exists(input_dir):
            files = [f for f in os.listdir(input_dir)
                    if f.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
        
        return {
            "required": {
                "audio_file": (sorted(files) if files else ["No audio files found"],),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("reference_audio",)
    FUNCTION = "load_audio"
    CATEGORY = "audio/loading"
    
    def load_audio(self, audio_file):
        """Load audio file using soundfile (no FFmpeg needed!)"""
        input_dir = folder_paths.get_input_directory()
        audio_path = os.path.join(input_dir, audio_file)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio with soundfile
            data, sample_rate = sf.read(audio_path, dtype='float32')
            
            # Convert to torch tensor
            waveform = torch.from_numpy(data).float()
            
            # Ensure correct shape: (channels, samples)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
            
            # Add batch dimension
            waveform = waveform.unsqueeze(0)
            
            return ({
                "waveform": waveform,
                "sample_rate": sample_rate
            },)
            
        except Exception as e:
            raise RuntimeError(f"Error loading audio file: {str(e)}")


class SoproSaveAudio:
    """Save generated audio to file using soundfile"""
    
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
                "format": (["wav", "flac", "ogg"],),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "audio/output"
    
    def save_audio(self, audio, filename_prefix="sopro_audio", format="wav"):
        """Save audio to file using soundfile"""
        
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']
        
        # Remove batch dimension
        if waveform.dim() == 3:
            waveform = waveform[0]
        
        # Convert to numpy and transpose
        audio_numpy = waveform.cpu().numpy().T
        
        # Generate unique filename
        counter = 0
        while True:
            filename = f"{filename_prefix}_{counter:05d}.{format}"
            filepath = os.path.join(self.output_dir, filename)
            if not os.path.exists(filepath):
                break
            counter += 1
        
        try:
            sf.write(filepath, audio_numpy, sample_rate)
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
