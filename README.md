# ComfyUI-Sopro

Sopro TTS custom nodes for ComfyUI - Lightweight CPU-based text-to-speech with zero-shot voice cloning.

## Features
- ⚡ Runs efficiently on CPU (0.25 RTF - 30s audio in 7.5s)
- 🎤 Zero-shot voice cloning with 3-12s reference audio
- 🔧 Compatible with all ComfyUI audio workflows
- 💾 169M parameters - lightweight and fast

## Installation

1. Navigate to your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes/
```

2. Clone this repository:
```bash
git clone https://github.com/ai-joe-git/ComfyUI-Sopro.git
cd ComfyUI-Sopro
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Restart ComfyUI

## Nodes

### Sopro TTS Generator
Main text-to-speech generation node with optional voice cloning.

**Inputs:**
- `text` (required): Text to synthesize
- `reference_audio` (optional): Reference audio for voice cloning
- `speed` (optional): Speech speed (0.5-2.0, default 1.0)
- `temperature` (optional): Generation temperature (0.1-1.5, default 0.7)
- `seed` (optional): Random seed for reproducibility

**Outputs:**
- `audio`: Generated audio in ComfyUI format

### Sopro Load Reference Audio
Load audio files for voice cloning.

**Inputs:**
- `audio_file`: Audio file from input directory

**Outputs:**
- `reference_audio`: Audio in ComfyUI format

### Sopro Save Audio
Save generated audio to output directory.

**Inputs:**
- `audio`: Audio to save
- `filename_prefix`: Output filename prefix
- `format`: Output format (wav/mp3/flac)

## Example Workflow

1. Add "Sopro TTS Generator" node
2. Enter your text
3. (Optional) Add "Sopro Load Reference Audio" and connect for voice cloning
4. Connect output to "Sopro Save Audio" or other audio processing nodes
5. Generate!

## Tips
- Use phonemes instead of abbreviations (e.g., "1 plus 2" not "1 + 2")
- Reference audio should be 3-12 seconds for best voice cloning
- Lower temperature for more consistent output
- Works great with other ComfyUI audio nodes!

## Credits
- Sopro TTS by [samuel-vitorino](https://github.com/samuel-vitorino/sopro)
```

