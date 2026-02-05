"""
Video Uniqueizer Pro - Configuration
Global settings and constants
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import random


# ============================================================================
# PATHS
# ============================================================================

APP_DIR = Path(__file__).parent
ASSETS_DIR = APP_DIR / "assets"
WATERMARKS_DIR = ASSETS_DIR / "watermarks"
OVERLAYS_DIR = ASSETS_DIR / "overlays"
SUBLIMINAL_DIR = ASSETS_DIR / "subliminal"
VFX_TEXTURES_DIR = ASSETS_DIR / "vfx_textures"
LUTS_DIR = ASSETS_DIR / "luts"
NEURAL_MODELS_DIR = APP_DIR / "neural" / "models"

DEFAULT_OUTPUT_DIR = Path.home() / "Videos" / "Uniqueized"


# ============================================================================
# RANDOMIZATION RANGES (Aggressive)
# ============================================================================

@dataclass
class VisualRanges:
    """Visual transformation ranges (Cinematic & Subtle)"""
    speed: tuple = (0.85, 1.20) # User requested range
    crop_percent: tuple = (1.5, 4) # Tighter for shorts
    rotation_degrees: tuple = (-2.0, 2.0)
    trim_start_range: tuple = (0.05, 0.45) # Base, modified by SmartFit
    trim_end_range: tuple = (0.05, 0.45)
    hue_shift: tuple = (0.0, 0.0) # DISABLED: Too risky for user aesthetic
    saturation_shift: tuple = (-5, 5) # Narrowed for safety
    value_shift: tuple = (-4, 4)
    contrast: tuple = (0.95, 1.05)
    brightness: tuple = (0.95, 1.05)
    gamma: tuple = (0.92, 1.08)
    noise_percent: tuple = (0.02, 0.4) # Even softer
    vignette_percent: tuple = (1, 4)
    blur_edge_px: tuple = (0.2, 1.2)
    grid_opacity: tuple = (0.003, 0.01)
    bg_offset_percent: tuple = (-2.0, 2.0)
    mirror_chance: float = 0.5
    color_balance_rs: tuple = (-0.04, 0.04) 
    color_balance_gs: tuple = (-0.05, -0.01) # ALWAYS negative green bias
    color_balance_bs: tuple = (-0.04, 0.04) 


@dataclass
class AudioRanges:
    """Audio transformation ranges"""
    pitch_semitones: tuple = (-5, 5)
    speed: tuple = (0.88, 1.12)
    reverb_percent: tuple = (5, 30)
    bass_boost_db: tuple = (0, 6)
    treble_adjust_db: tuple = (-4, 4)
    echo_delay_ms: tuple = (10, 80)
    echo_wet_percent: tuple = (10, 30)
    stereo_width: tuple = (0.7, 1.5)


@dataclass
class VFXRanges:
    """VFX effect ranges (Professional tuning)"""
    glitch_intensity: tuple = (0.02, 0.15)
    rgb_split_px: tuple = (1, 4)
    shake_amplitude_px: tuple = (0.5, 3.0) # Slower, more subtle
    light_leak_opacity: tuple = (0.10, 0.25)
    lens_distortion: tuple = (-0.015, 0.015)
    film_grain: tuple = (0.05, 0.15)
    chromatic_aberration_px: tuple = (0.5, 3)
    speed_ramp_factor: tuple = (0.7, 0.9)
    speed_ramp_duration: tuple = (0.5, 1.5)


@dataclass
class EncodingRanges:
    """Encoding parameter ranges"""
    resolution_variance: tuple = (0.02, 0.08) # Reduced for social media compatibility
    fps_options: tuple = (24, 25, 30, 50, 60)
    gop_size: tuple = (30, 90)
    b_frames: tuple = (0, 3)
    bitrate_variance: tuple = (0.15, 0.25)
    crf: tuple = (18, 28)
    keyframe_interval_sec: tuple = (1, 5)


@dataclass
class NeuralRanges:
    """Neural network effect ranges"""
    style_strength: tuple = (0.05, 0.20)
    face_blur_strength: tuple = (15, 35)


# ============================================================================
# METADATA POOLS
# ============================================================================

DEVICE_POOL = [
    "iPhone 14 Pro",
    "iPhone 14 Pro Max", 
    "iPhone 15 Pro",
    "iPhone 15 Pro Max",
    "iPhone 16 Pro",
    "iPhone 16 Pro Max",
]

DEVICE_TEMPLATES = {
    "iPhone 16 Pro": {
        "make": "Apple",
        "model": "iPhone 16 Pro",
        "lens": "Triple Camera (24mm, 13mm, 120mm)",
        "software": "iOS 18.2",
        "aperture": "f/1.78",
    },
    "Samsung Galaxy S24 Ultra": {
        "make": "Samsung",
        "model": "SM-S928B",
        "lens": "Quad Camera (23mm, 13mm, 67mm, 111mm)",
        "software": "Android 14",
        "aperture": "f/1.7",
    },
    "Sony A7 IV": {
        "make": "Sony",
        "model": "ILCE-7M4",
        "lens": "FE 24-70mm F2.8 GM II",
        "software": "v2.01",
        "aperture": "f/2.8",
    },
    "GoPro HERO12 Black": {
        "make": "GoPro",
        "model": "HERO12 Black",
        "lens": "HyperView Lens",
        "software": "v1.20",
        "aperture": "f/2.5",
    }
}

SOFTWARE_POOL = [
    "CapCut 12.0.0",
    "CapCut 12.1.0",
    "CapCut 12.2.0",
    "CapCut 12.3.0",
    "CapCut 13.0.0",
    "CapCut 13.1.0",
]

FILENAME_POOL = [
    "summer_vibes_edit_final",
    "anime_compilation_v",
    "gaming_moments_2024",
    "meme_mashup_best",
    "edit_capcut_export",
    "fyp_edit_new",
    "viral_clip_remix",
    "best_moments_compilation",
    "funny_edit_v",
    "night_aesthetic_edit",
    "anime_amv_edit",
    "gaming_highlights_",
    "tiktok_trend_edit",
    "reels_export_",
    "shorts_upload_",
    "content_final_",
    "daily_vlog_edit",
    "compilation_part",
    "edit_draft_final",
    "export_hd_",
]

COMMENT_TAGS_POOL = [
    "#anime", "#gaming", "#meme", "#fyp", "#viral",
    "#edit", "#aesthetic", "#trend", "#funny", "#compilation",
    "#amv", "#highlights", "#bestmoments", "#capcut", "#tiktok",
    "#reels", "#shorts", "#youtube", "#instagram", "#content",
]

GPS_COORDINATES_POOL = [
    (35.6762, 139.6503),    # Tokyo
    (37.5665, 126.9780),    # Seoul
    (34.0522, -118.2437),   # Los Angeles
    (40.7128, -74.0060),    # New York
    (51.5074, -0.1278),     # London
    None,                    # No GPS
    None,                    # No GPS
    None,                    # No GPS
]

CODEC_POOL = [
    ("h264_nvenc", "NVIDIA H.264"),
    ("hevc_nvenc", "NVIDIA H.265"),
    ("libx264", "x264"),
    ("libx265", "x265"),
]


# ============================================================================
# PLATFORM PRESETS
# ============================================================================

@dataclass
class PlatformPreset:
    """Platform-specific settings"""
    name: str
    base_resolution: tuple
    aspect_ratio: str
    fps_options: tuple
    bitrate_range: tuple  # Mbps
    max_duration: Optional[int] = None  # seconds
    
    def get_random_resolution(self) -> tuple:
        """Get randomized resolution within variance"""
        variance = random.uniform(
            EncodingRanges().resolution_variance[0],
            EncodingRanges().resolution_variance[1]
        )
        w, h = self.base_resolution
        w_offset = int(w * variance * random.choice([-1, 1]))
        h_offset = int(h * variance * random.choice([-1, 1]))
        # Ensure even numbers
        return ((w + w_offset) // 2 * 2, (h + h_offset) // 2 * 2)
    
    def get_random_fps(self) -> int:
        """Get random FPS from options"""
        return random.choice(self.fps_options)
    
    def get_random_bitrate(self) -> float:
        """Get random bitrate in Mbps"""
        return random.uniform(*self.bitrate_range)


PRESETS = {
    "tiktok": PlatformPreset(
        name="TikTok",
        base_resolution=(1080, 1920),
        aspect_ratio="9:16",
        fps_options=(30, 60),
        bitrate_range=(8, 12),
        max_duration=180,
    ),
    "youtube_shorts": PlatformPreset(
        name="YouTube Shorts",
        base_resolution=(1080, 1920),
        aspect_ratio="9:16",
        fps_options=(30, 60),
        bitrate_range=(10, 15),
        max_duration=60,
    ),
    "instagram_reels": PlatformPreset(
        name="Instagram Reels",
        base_resolution=(1080, 1920),
        aspect_ratio="9:16",
        fps_options=(30,),
        bitrate_range=(6, 10),
        max_duration=90,
    ),
    "long_video": PlatformPreset(
        name="Long Video",
        base_resolution=(1920, 1080),
        aspect_ratio="16:9",
        fps_options=(24, 25, 30, 60),
        bitrate_range=(15, 25),
        max_duration=None,
    ),
}


# ============================================================================
# APP SETTINGS
# ============================================================================

@dataclass
class AppSettings:
    """Application settings"""
    # Processing
    use_gpu: bool = True
    gpu_device: int = 0
    max_threads: int = 4
    
    # Default uniqueization level (0.0 - 1.0)
    uniqueization_level: float = 0.9
    
    # Feature toggles
    enable_25th_frame: bool = True
    enable_watermark: bool = False
    enable_vfx: bool = True
    enable_neural: bool = True
    enable_random_metadata: bool = True
    enable_random_codecs: bool = True
    enable_vfr: bool = True
    enable_ultrasonic_noise: bool = True
    enable_pixel_jitter: bool = True
    enable_edge_blur: bool = True
    enable_subliminal: bool = True
    enable_color_shift: bool = True
    enable_subtitles: bool = True
    enable_junk_data: bool = True
    enable_dynamic_crop: bool = True
    enable_sensor_noise: bool = True
    enable_lum_jitter: bool = True
    enable_audio_jitter: bool = True
    enable_speed_warp: bool = True
    enable_chroma_jitter: bool = True
    enable_tint_flash: bool = True
    enable_spectral_masking: bool = True
    enable_atomic_reorder: bool = True
    
    # Ranges
    visual: VisualRanges = field(default_factory=VisualRanges)
    audio: AudioRanges = field(default_factory=AudioRanges)
    vfx: VFXRanges = field(default_factory=VFXRanges)
    encoding: EncodingRanges = field(default_factory=EncodingRanges)
    neural: NeuralRanges = field(default_factory=NeuralRanges)


# Global settings instance
settings = AppSettings()
