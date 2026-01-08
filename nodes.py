# nodes.py - ENHANCED VERSION
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
    Enhanced with full parameter control
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
                "reference_audio": ("AUDIO",),
            },
            "optional": {
                # Quality & Style Controls
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.5,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Higher = more creative/varied, Lower = more consistent"
                }),
                "style_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "How closely to match reference voice style"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.5,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Nucleus sampling - lower = more focused"
                }),
                
                # Generation Length Controls
                "max_frames": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Maximum audio length in frames"
                }),
                "min_gen_frames": ("INT", {
                    "default": 16,
                    "min": 8,
                    "max": 128,
                    "step": 8,
                    "tooltip": "Minimum frames to generate"
                }),
                
                # Reference Audio Controls
                "ref_seconds": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5,
                    "display": "slider",
                    "tooltip": "Seconds of reference audio to use for cloning"
                }),
                "use_prefix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use reference audio as prefix for more natural flow"
                }),
                "prefix_sec_fixed": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.5,
                    "display": "slider",
                    "tooltip": "Fixed prefix duration in seconds"
                }),
                
                # Anti-repetition & Stopping
                "anti_loop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Prevent repetitive loops in generation"
                }),
                "use_stop_head": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use stopping mechanism for natural endings"
                }),
                "stop_patience": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Patience for stopping criteria"
                }),
                "stop_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Threshold for stop detection"
                }),
                
                # Post-processing
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Playback speed adjustment"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "tooltip": "Random seed for reproducibility (0 = random)"
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
                print("🎙️ Loading Sopro TTS model from HuggingFace...")
                
                from sopro import SoproTTS
                
                # Load from HuggingFace with the official repo ID
                self.model = SoproTTS.from_pretrained(
                    "samuel-vitorino/sopro",
                    device=self.device
                )
                
                print("✅ Sopro TTS model loaded successfully!")
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
            " % ": " percent ",
            " $ ": " dollars ",
            " # ": " number ",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove multiple spaces
        text = " ".join(text.split())
        
        return text
    
    def generate_speech(self, text, reference_audio, 
                       temperature=0.7, style_strength=1.0, top_p=0.95,
                       max_frames=512, min_gen_frames=16,
                       ref_seconds=3.0, use_prefix=True, prefix_sec_fixed=2.0,
                       anti_loop=True, use_stop_head=True, 
                       stop_patience=10, stop_threshold=0.5,
                       speed=1.0, seed=0):
        """Generate speech from text using Sopro TTS with full control"""
        
        # Set seed for reproducibility
        if seed > 0:
            seed = min(seed, 2147483647)
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"🎲 Seed set to: {seed}")
        
        # Load model
        model = self.load_model()
        
        # Preprocess text
        text = self.preprocess_text(text)
        
        if not text:
            raise ValueError("Text input cannot be empty")
        
        print(f"📝 Text to synthesize: '{text[:100]}{'...' if len(text) > 100 else ''}'")
        
        # Temporary file for reference audio
        temp_audio_path = None
        
        try:
            # Build kwargs based on available parameters
            kwargs = {"text": text}
            
            # Handle reference audio - Sopro needs a FILE PATH
            if reference_audio is not None:
                print("🎵 Processing reference audio...")
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
                    print("  ↳ Converted stereo to mono")
                
                # Resample to 24kHz if needed (Sopro's expected rate)
                if ref_sample_rate != 24000:
                    print(f"  ↳ Resampling from {ref_sample_rate}Hz to 24000Hz")
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
                    print(f"  ↳ Saved reference to: {temp_audio_path}")
                
                kwargs["ref_audio_path"] = temp_audio_path
            
            # Add all control parameters
            kwargs.update({
                "temperature": temperature,
                "style_strength": style_strength,
                "top_p": top_p,
                "max_frames": max_frames,
                "min_gen_frames": min_gen_frames,
                "ref_seconds": ref_seconds,
                "use_prefix": use_prefix,
                "prefix_sec_fixed": prefix_sec_fixed,
                "anti_loop": anti_loop,
                "use_stop_head": use_stop_head,
                "stop_patience": stop_patience,
                "stop_threshold": stop_threshold,
            })
            
            print(f"⚙️  Generation settings:")
            print(f"  ↳ Temperature: {temperature}, Style: {style_strength}, Top-p: {top_p}")
            print(f"  ↳ Max frames: {max_frames}, Ref seconds: {ref_seconds}")
            print(f"  ↳ Anti-loop: {anti_loop}, Use stop: {use_stop_head}")
            
            # Generate speech
            print("🎙️  Generating speech...")
            audio_output = model.synthesize(**kwargs)
            print("✅ Speech generated successfully!")
            
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
            
            duration = audio_tensor.shape[-1] / 24000
            print(f"⏱️  Generated audio duration: {duration:.2f}s")
            
            # Apply speed adjustment if needed (by resampling)
            if speed != 1.0:
                print(f"⚡ Applying speed adjustment: {speed}x")
                original_sample_rate = 24000
                target_sample_rate = int(original_sample_rate * speed)
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor,
                    orig_freq=target_sample_rate,
                    new_freq=original_sample_rate
                )
                new_duration = audio_tensor.shape[-1] / 24000
                print(f"  ↳ New duration: {new_duration:.2f}s")
            
            # Normalize audio to [-1, 1] if needed
            max_val = audio_tensor.abs().max()
            if max_val > 1.0:
                audio_tensor = audio_tensor / max_val
                print(f"🔊 Normalized audio (peak was {max_val:.3f})")
            
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
                    print(f"🧹 Cleaned up temporary file")
                except Exception as e:
                    print(f"⚠️  Warning: Could not delete temporary file: {e}")


class SoproTTSPresets:
    """Preset configurations for common use cases"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": ([
                    "balanced",
                    "high_quality",
                    "fast",
                    "creative",
                    "consistent",
                    "expressive",
                    "natural",
                ], {
                    "default": "balanced"
                }),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "INT", "FLOAT", "BOOLEAN", "FLOAT")
    RETURN_NAMES = ("temperature", "style_strength", "top_p", "max_frames", "ref_seconds", "anti_loop", "speed")
    FUNCTION = "get_preset"
    CATEGORY = "audio/generation/presets"
    
    def get_preset(self, preset):
        """Return preset values"""
        presets = {
            "balanced": (0.7, 1.0, 0.95, 512, 3.0, True, 1.0),
            "high_quality": (0.5, 1.2, 0.9, 1024, 5.0, True, 0.95),
            "fast": (0.8, 0.8, 0.98, 256, 2.0, True, 1.1),
            "creative": (1.2, 0.8, 0.98, 768, 3.0, False, 1.0),
            "consistent": (0.3, 1.5, 0.85, 512, 4.0, True, 1.0),
            "expressive": (1.0, 1.3, 0.95, 640, 4.0, True, 0.98),
            "natural": (0.6, 1.1, 0.92, 512, 3.5, True, 0.97),
        }
        
        values = presets.get(preset, presets["balanced"])
        print(f"🎚️  Loaded preset: {preset}")
        return values


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
            print(f"📂 Loading reference audio: {audio_file}")
            
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
            
            duration = waveform.shape[-1] / sample_rate
            print(f"  ↳ Duration: {duration:.2f}s, Sample rate: {sample_rate}Hz")
            
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
            duration = len(audio_numpy) / sample_rate
            print(f"💾 Audio saved to: {filepath}")
            print(f"  ↳ Duration: {duration:.2f}s, Format: {format}")
            
            return {"ui": {"audio": [filename]}}
            
        except Exception as e:
            raise RuntimeError(f"Error saving audio: {str(e)}")


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "SoproTTSNode": SoproTTSNode,
    "SoproTTSPresets": SoproTTSPresets,
    "SoproLoadReferenceAudio": SoproLoadReferenceAudio,
    "SoproSaveAudio": SoproSaveAudio,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "SoproTTSNode": "🎙️ Sopro TTS Generator",
    "SoproTTSPresets": "🎚️ Sopro TTS Presets",
    "SoproLoadReferenceAudio": "📂 Sopro Load Reference Audio",
    "SoproSaveAudio": "💾 Sopro Save Audio",
}
