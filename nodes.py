# nodes.py
import torch
import torchaudio
import soundfile as sf
import numpy as np
import os
import folder_paths
import inspect
import tempfile

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
        """Lazy load the Sopro model from HuggingFace"""
        if self.model is None:
            try:
                print("Loading Sopro TTS model from HuggingFace...")
                
                from sopro import SoproTTS
                
                # Load from HuggingFace with the official repo ID
                self.model = SoproTTS.from_pretrained(
                    "samuel-vitorino/sopro",
                    device=self.device
                )
                
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
        # Convert common symbols to words
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
        
        # Temporary file for reference audio
        temp_audio_path = None
        
        try:
            # Get the synthesize method signature to understand parameters
            sig = inspect.signature(model.synthesize)
            params = list(sig.parameters.keys())
            print(f"Sopro synthesize parameters: {params}")
            
            # Build kwargs based on available parameters
            kwargs = {"text": text}
            
            # Handle reference audio - Sopro needs a FILE PATH, not raw audio
            if reference_audio is not None:
                ref_waveform = reference_audio['waveform']
                ref_sample_rate = reference_audio['sample_rate']
                
                if isinstance(ref_waveform, torch.Tensor):
                    ref_waveform = ref_waveform.cpu().numpy()
                
                # Remove batch dimension if present
                if ref_waveform.ndim == 3:
                    ref_waveform = ref_waveform[0]
                
                # Convert stereo to mono
                if ref_waveform.shape[0] > 1:
                    ref_waveform = ref_waveform.mean(axis=0, keepdims=True)
                
                # Resample to 24kHz if needed (Sopro's expected rate)
                if ref_sample_rate != 24000:
                    ref_waveform_tensor = torch.from_numpy(ref_waveform).float()
                    ref_waveform_tensor = torchaudio.functional.resample(
                        ref_waveform_tensor, ref_sample_rate, 24000
                    )
                    ref_waveform = ref_waveform_tensor.numpy()
                    ref_sample_rate = 24000
                
                # Transpose to (samples, channels) for soundfile
                audio_to_save = ref_waveform.T
                
                # Create temporary WAV file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    temp_audio_path = tmp_file.name
                    sf.write(temp_audio_path, audio_to_save, ref_sample_rate)
                    print(f"Saved reference audio to: {temp_audio_path}")
                
                # Use ref_audio_path parameter
                kwargs["ref_audio_path"] = temp_audio_path
            
            # Add temperature if supported
            if "temperature" in params:
                kwargs["temperature"] = temperature
            
            # Speed isn't directly supported in Sopro, but we can adjust post-generation
            # by resampling if needed
            
            print(f"Calling model.synthesize with: {list(kwargs.keys())}")
            
            # Generate speech
            audio_output = model.synthesize(**kwargs)
            
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
            
            # Apply speed adjustment if needed (by resampling)
            if speed != 1.0:
                # Speed up/down by resampling
                original_sample_rate = 24000
                target_sample_rate = int(original_sample_rate * speed)
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor,
                    orig_freq=target_sample_rate,
                    new_freq=original_sample_rate
                )
            
            # Normalize audio to [-1, 1] if needed
            max_val = audio_tensor.abs().max()
            if max_val > 1.0:
                audio_tensor = audio_tensor / max_val
            
            return ({
                "waveform": audio_tensor,
                "sample_rate": 24000
            },)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Error generating speech: {str(e)}")
        
        finally:
            # Clean up temporary file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                    print(f"Cleaned up temporary file: {temp_audio_path}")
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {temp_audio_path}: {e}")


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
                "audio_file": (sorted(files) if files else ["No audio files found"],),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("reference_audio",)
    FUNCTION = "load_audio"
    CATEGORY = "audio/loading"
    
    def load_audio(self, audio_file):
        """Load audio file using soundfile"""
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
