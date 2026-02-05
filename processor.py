
import os
import random
import tempfile
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, List
import subprocess
import json

from config import (
    settings, PRESETS, DEVICE_POOL, SOFTWARE_POOL, 
    FILENAME_POOL, CODEC_POOL, AppSettings
)
from transformations.metadata import randomize_metadata
from transformations.visual import generate_visual_filters, VisualParams
from transformations.audio import generate_audio_filters, AudioParams
from transformations.vfx import generate_vfx_filters, VFXParams
from transformations.watermark import WatermarkOverlay, WatermarkParams, WatermarkPosition


@dataclass
class VideoInfo:
    """Information about a video file"""
    path: str
    width: int
    height: int
    fps: float
    duration: float
    bitrate: int
    codec: str
    has_audio: bool
    audio_codec: Optional[str] = None
    audio_bitrate: Optional[int] = None


class VideoProcessor:
    """
    Main video processing pipeline.
    Coordinates all transformation modules to generate unique video variants.
    """

    def __init__(self, app_settings: AppSettings = None):
        self.settings = app_settings or settings
        self.progress_callback: Optional[Callable[[float, str], None]] = None

        # Lazy-loaded transformation modules
        self._visual = None
        self._audio = None
        self._metadata = None
        self._vfx = None
        self._subliminal = None
        self._watermark = None
        self._neural = None
        self._encoder = None

        # Initialize LUT pool
        from config import LUTS_DIR
        self.lut_pool = list(LUTS_DIR.glob("*.cube")) if LUTS_DIR.exists() else []

        # Initialize Neural Stylizer
        from neural.style_transfer import StyleTransfer
        self._style_transfer = StyleTransfer()

    # ... analyze_video, process и др. уже есть выше ...

    def _process_single(
        self,
        video_info: VideoInfo,
        preset,
        output_dir: str,
        base_progress: float,
        progress_range: float
    ) -> str:
        """Process a single variant for one platform"""
        # Generate random parameters
        params = self._generate_random_params(preset, video_info)

        # Generate output filename
        output_filename = self._generate_filename(preset.name)
        output_path = os.path.join(output_dir, output_filename)

        # Build FFmpeg filter chain
        filters = self._build_filter_chain(video_info, params)

        # Build FFmpeg command
        cmd = self._build_ffmpeg_command(
            video_info.path,
            output_path,
            params,
            filters
        )

        self._update_progress(
            base_progress + progress_range * 0.3,
            f"Encoding {preset.name}..."
        )

        # Add watermark input if needed
        if 'watermark_path' in params and params['watermark_path']:
            cmd.insert(4, '-i')
            cmd.insert(5, params['watermark_path'])

        # Execute FFmpeg
        try:
            extra_args = []

            if self.settings.enable_junk_data:
                import string
                import uuid
                junk_msg = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
                u_id = uuid.uuid4().hex
                extra_args.extend(['-bsf:v', f"h264_metadata=sei_user_data='{u_id}+{junk_msg}'"])
                extra_args.extend(['-metadata', f'junk_hash={junk_msg}'])

            if getattr(self.settings, 'enable_atomic_reorder', False):
                extra_args.extend(['-movflags', '+faststart+frag_keyframe+empty_moov'])

            if not self.settings.enable_neural:
                # Standard (fast) path
                final_cmd = cmd[:-1] + extra_args + [cmd[-1]]
                subprocess.run(final_cmd, check=True, capture_output=True)
            else:
                # Neural path
                self._process_neural_wrapper(
                    cmd, output_path, video_info, params,
                    base_progress, progress_range, extra_args
                )

        except subprocess.CalledProcessError as e:
            stderr_text = e.stderr.decode(errors='replace') if e.stderr else "No stderr output"
            raise RuntimeError(f"FFmpeg encoding failed: {stderr_text}")
        except Exception as e:
            raise RuntimeError(f"Processing error: {str(e)}")

        # Apply metadata
        if self.settings.enable_random_metadata:
            self._update_progress(
                base_progress + progress_range * 0.9,
                "Applying metadata..."
            )
            self._apply_metadata(output_path, params)

        # Save processing report
        self._save_report(output_path, params)
        return output_path

    def _process_neural_wrapper(
        self,
        cmd: list[str],
        output_path: str,
        video_info: VideoInfo,
        params: dict,
        base_progress: float,
        progress_range: float,
        extra_args: List[str] = None
    ):
        """Piped processing with frame-by-frame neural stylization"""
        import numpy as np
        import cv2
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_base = os.path.join(tmp_dir, "base_variant.mp4")
            extra_args = extra_args or []

            # 1. Create base variant without neural processing
            cmd_base = cmd[:-1] + extra_args + [temp_base]
            subprocess.run(cmd_base, check=True, capture_output=True)

            # 2. Get ACTUAL dimensions of processed video
            analyze_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=s=x:p=0',
                temp_base
            ]
            result = subprocess.run(analyze_cmd, capture_output=True, text=True)
            w, h = map(int, result.stdout.strip().split('x'))

            self._update_progress(
                base_progress + progress_range * 0.4,
                "Applying Neural Styling..."
            )

            # 3. Get FPS from params
            fps = params['fps']
            frame_size = w * h * 3
            total_frames = int(video_info.duration * fps)

            # 4. Start neural processing pipeline
            proc_in = subprocess.Popen(
                ['ffmpeg', '-i', temp_base, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )

            codec_name, _ = params['codec']
            proc_out = subprocess.Popen(
                [
                    'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                    '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', str(fps),
                    '-i', '-', '-i', temp_base,
                    '-map', '0:v', '-map', '1:a',
                    '-c:v', codec_name, '-c:a', 'copy',
                    '-pix_fmt', 'yuv420p',
                    '-crf', str(params['crf']), '-preset', 'medium',
                    *extra_args,
                    output_path
                ],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
            )

            try:
                frame_idx = 0
                batch_frames = []
                batch_size = 8

                while True:
                    raw_frame = proc_in.stdout.read(frame_size)
                    if not raw_frame or len(raw_frame) < frame_size:
                        if batch_frames:
                            styled_batch = self._style_transfer.apply_random_style(
                                np.array(batch_frames)
                            )
                            for frame in styled_batch:
                                proc_out.stdin.write(frame.tobytes())
                        break

                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((h, w, 3))
                    batch_frames.append(frame)

                    if len(batch_frames) >= batch_size:
                        styled_batch = self._style_transfer.apply_random_style(
                            np.array(batch_frames)
                        )
                        for f in styled_batch:
                            proc_out.stdin.write(f.tobytes())
                        batch_frames = []

                    frame_idx += 1
                    if frame_idx % 30 == 0:
                        prog = (frame_idx / total_frames) * 0.5
                        self._update_progress(
                            base_progress + progress_range * (0.4 + prog),
                            f"Neural Processing: {frame_idx}/{total_frames} frames"
                        )

            finally:
                try:
                    proc_in.stdout.close()
                    proc_out.stdin.close()
                    proc_in.wait(timeout=10)
                    proc_out.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc_in.terminate()
                    proc_out.terminate()
                    proc_in.wait()
                    proc_out.wait()

    def _generate_random_params(self, preset, video_info: VideoInfo) -> dict:
        """Generate randomized transformation parameters"""
        level = self.settings.uniqueization_level

        def scale_range(range_tuple):
            """Scale a range based on uniqueization level"""
            min_val, max_val = range_tuple
            mid = (min_val + max_val) / 2
            half_range = (max_val - min_val) / 2 * level
            return (mid - half_range, mid + half_range)

        v = self.settings.visual
        a = self.settings.audio
        e = self.settings.encoding

        params = {
            # Resolution and format
            'resolution': preset.get_random_resolution(),
            'fps': preset.get_random_fps(),
            'bitrate': preset.get_random_bitrate(),

            # Base speed
            'speed': random.uniform(*scale_range(v.speed)),
            'crop_percent': random.uniform(*scale_range(v.crop_percent)),
            'mirror': random.random() < v.mirror_chance,
            'rotation': random.uniform(*scale_range(v.rotation_degrees)),
            'hue_shift': random.uniform(*scale_range(v.hue_shift)),
            'saturation': random.uniform(*scale_range(v.saturation_shift)),
            'brightness': random.uniform(*scale_range(v.brightness)),
            'contrast': random.uniform(*scale_range(v.contrast)),
            'gamma': random.uniform(*scale_range(v.gamma)),
            'noise': random.uniform(*scale_range(v.noise_percent)),
            'vignette': random.uniform(*scale_range(v.vignette_percent)),
            'grid_opacity': random.uniform(*scale_range(v.grid_opacity)) if random.random() < 0.2 else 0.0,

            # SmartFit flag
            'smart_fit_active': False,

            # Color balance
            'cb_rs': random.uniform(*scale_range(v.color_balance_rs)),
            'cb_gs': random.uniform(*scale_range(v.color_balance_gs)),
            'cb_bs': random.uniform(*scale_range(v.color_balance_bs)),

            'neutralizer': -0.06 if random.random() < 0.8 else -0.03,
            'aesthetic_bias': random.choice(['cold', 'warm', 'neutral', 'cinematic', 'neutral', 'neutral']),

            # Trimming
            'trim_start': random.uniform(*scale_range(v.trim_start_range)),
            'trim_end': random.uniform(*scale_range(v.trim_end_range)),

            'lut_path': str(random.choice(self.lut_pool)) if self.lut_pool and random.random() < 0.6 else None,
            'lut_strength': random.uniform(0.2, 0.4),

            'speed_ramp_active': random.random() < 0.4,
            'speed_ramp_points': [],

            # Audio
            'pitch_shift': random.uniform(*scale_range(a.pitch_semitones)),
            'audio_speed': random.uniform(*scale_range(a.speed)),
            'reverb': random.uniform(*scale_range(a.reverb_percent)),
            'bass_boost': random.uniform(*scale_range(a.bass_boost_db)),

            # Encoding
            'codec': random.choice(CODEC_POOL),
            'gop_size': random.randint(*e.gop_size),
            'b_frames': random.randint(*e.b_frames),
            'crf': random.randint(*e.crf),

            # Metadata
            'device': random.choice(DEVICE_POOL),
            'software': random.choice(SOFTWARE_POOL),
        }

        # SmartFit 59s
        short_form_platforms = ['TikTok', 'YouTube Shorts', 'Instagram Reels']
        if preset.name in short_form_platforms and video_info.duration > 59.0:
            params['smart_fit_active'] = True
            target_duration = 59.0
            current_duration = video_info.duration
            required_speed = current_duration / target_duration

            if required_speed <= 1.20:
                params['speed'] = required_speed
            else:
                params['speed'] = 1.20
                excess = current_duration - (target_duration * 1.20)
                params['trim_start'] += excess * 0.5
                params['trim_end'] += excess * 0.5

        # Speed ramp points
        if params['speed_ramp_active']:
            duration = video_info.duration
            num_points = random.randint(2, 3)
            for _ in range(num_points):
                params['speed_ramp_points'].append({
                    'time': random.uniform(0.2, 0.8) * duration,
                    'factor': random.uniform(0.7, 1.3),
                    'duration': random.uniform(0.5, 1.5),
                })

        # VFX & audio filters
        audio_sync = None
        vfx_filters_str = None
        vfx_params = None

        new_duration = video_info.duration - params['trim_start'] - params['trim_end']
        new_duration = max(0.5, new_duration)

        if self.settings.enable_vfx:
            vfx_filters_str, vfx_params = generate_vfx_filters(level=level)
            from transformations.vfx import VFXProcessor
            vfx_proc = VFXProcessor()
            audio_sync = vfx_proc.build_audio_sync_filters(
                vfx_params, video_fps=video_info.fps
            )

        from transformations.audio import generate_audio_filters
        v_speed = params.get('speed', 1.0)
        audio_filters_str, audio_params = generate_audio_filters(
            level=self.settings.uniqueization_level,
            video_speed=v_speed,
            sync_filter=audio_sync,
            trim_start=params['trim_start'],
            trim_duration=new_duration
        )

        params.update({
            'audio_filters': audio_filters_str,
            'audio_params': audio_params,
            'vfx_filters': vfx_filters_str,
            'vfx_params': vfx_params,
            'final_duration': new_duration / params['speed'],
        })

        if self.settings.enable_watermark and hasattr(self.settings, 'watermark_path'):
            params['watermark_path'] = self.settings.watermark_path
            params['watermark_pos'] = getattr(self.settings, 'watermark_pos', 'Bottom Right')

        return params

    def _generate_filename(self, platform_name: str) -> str:
        """Generate random filename for output"""
        base = random.choice(FILENAME_POOL)
        suffix = random.randint(1, 999)
        platform_short = platform_name.lower().replace(' ', '_')[:3]
        return f"{base}{suffix}_{platform_short}.mp4"

    
    # ========================================================================
    # FFMPEG BUILDING
    # ========================================================================
    
    def _build_filter_chain(self, video_info: VideoInfo, params: dict) -> str:
        """Build FFmpeg video filter chain string (for use in filter_complex)"""
        filters = []
        
        # 1. Trimming (Applied FIRST to avoid processing excess frames)
        if params['trim_start'] > 0:
             filters.append(f"trim=start={params['trim_start']:.3f}:duration={max(0.1, video_info.duration - params['trim_start'] - params['trim_end']):.3f},setpts=PTS-STARTPTS")
        
        # 2. Visual Transformations
        # 1. Base Resolution & Speed (Smart Fit & Ramping)
        base_speed = params.get('speed', 1.0)
        
        # Build unified speed expression (SmartFit + Ramping)
        # We use a linear multiplicative format to avoid exponential nesting (which crashes FFmpeg)
        # Format: base_speed * if(between(T, t1, t1+d1), f1, 1) * if(...)
        speed_expr = f"{base_speed:.4f}"
        
        if params.get('speed_ramp_active') and params.get('speed_ramp_points'):
             for point in params['speed_ramp_points']:
                  t = point['time']
                  f = point['factor']
                  d = point['duration']
                  # Multiply by the factor if in window, else by 1.0 (neutral)
                  speed_expr += f" * if(between(T,{t:.2f},{t+d:.2f}),{f:.4f},1.0)"
        
        # Apply unified timing (Escaped with single quotes to handle commas)
        if self.settings.enable_vfr:
            # Add subtle jitter (±0.001s) to break constant frame rate fingerprint
            speed_expr = f"({speed_expr}) * (1 + (random(0)-0.5)*0.002)"
        
        filters.append(f"setpts='PTS/({speed_expr})'")
             
        # Mirror
        if params['mirror']:
            filters.append("hflip")
        
        # Rotation with Auto-Zoom
        if abs(params['rotation']) > 0.05:
            rad = params['rotation'] * 3.14159 / 180
            import math
            zoom = 1.0 / math.cos(abs(rad)) + 0.02
            filters.append(f"rotate={rad:.4f}:c=black@0,scale=iw*{zoom:.4f}:-1,crop=iw/{zoom:.4f}:ih/{zoom:.4f}")
            
        # VFX Filters (Glitch, Shake, etc.)
        if self.settings.enable_vfx and params.get('vfx_filters'):
             filters.extend(params['vfx_filters'])
        
        # Color & Noise
        eq_parts = []
        if abs(params['brightness'] - 1.0) > 0.01: eq_parts.append(f"brightness={params['brightness'] - 1:.4f}")
        if abs(params['contrast'] - 1.0) > 0.01: eq_parts.append(f"contrast={params['contrast']:.4f}")
        if abs(params['gamma'] - 1.0) > 0.01: eq_parts.append(f"gamma={params['gamma']:.4f}")
        if abs(params['saturation']) > 0.01: eq_parts.append(f"saturation={1 + params['saturation']/100:.4f}")
        if eq_parts: filters.append(f"eq={':'.join(eq_parts)}")
        
        # Color Balance (ULTIMATE NUCLEAR DEFENSIVE STABILIZATION)
        # We target all spectral ranges: shadows(s), midtones(m), highlights(h)
        rs, gs, bs = params.get('cb_rs', 0), params.get('cb_gs', 0), params.get('cb_bs', 0)
        
        # 1. SHADOWS: Extreme negative tilt (-12% to -15%)
        # This kills green in the darkest parts of the image (Family Mart shadows)
        gs_shadows = -0.14 + random.uniform(-0.02, 0.02)
        
        # 2. MIDTONES: Strong suppression (-6% to -9%)
        # High impact on skin tones and walls
        gm_midtones = -0.07 + random.uniform(-0.01, 0.01)
        
        # 3. HIGHLIGHTS: Definitive safety (-4%)
        # Kills green in lights and white surfaces
        gh_highs = -0.04
        
        # Aesthetic override (Structural Consistency)
        bias = params.get('aesthetic_bias', 'neutral')
        if bias == 'cold':
             bs += 0.09 # Professional blue shadows
             rs -= 0.01
             gm_midtones -= 0.01
        elif bias == 'warm':
             rs += 0.07 # Golden hour warmth
             bs -= 0.03
             gm_midtones -= 0.01
        elif bias == 'cinematic':
             bs += 0.05
             rs += 0.03
             gs_shadows -= 0.03 # Extra deep shadows

        # Construct filter with strictly negative green multipliers
        cb_str = (
            f"colorbalance="
            f"rs={rs:.3f}:gs={gs_shadows:.3f}:bs={bs:.3f}:"
            f"rm=0.01:gm={gm_midtones:.3f}:bm=0.01:"
            f"rh=0.01:gh={gh_highs:.3f}:bh=0.01"
        )
        filters.append(cb_str)
        
        if abs(params['hue_shift']) > 0.1: filters.append(f"hue=h={params['hue_shift']:.2f}")
        if params['noise'] > 0.1: filters.append(f"noise=alls={int(params['noise'] * 10)}:allf=t")
        
        # Vignette
        if params['vignette'] > 1: filters.append(f"vignette=angle={params['vignette'] / 100 * 0.5:.4f}")
        
        if params.get('lut_path'):
             # Create a blended LUT effect for softer look (Simulating LUT Strength)
             lpath = params['lut_path'].replace('\\', '/').replace(':', '\\:')
             strength = params.get('lut_strength', 0.4)
             # Use split=2 to generate two copies for blending
             filters.append(f"split=2[v_orig_copy][v_to_grade];[v_to_grade]lut3d='{lpath}'[v_graded];[v_orig_copy][v_graded]blend=all_mode=normal:all_opacity={strength:.2f}")
        
        # Transparent Grid
        if params.get('grid_opacity'):
             filters.append(f"drawgrid=w=iw/20:h=ih/20:t=1:c=white@{params['grid_opacity']:.3f}")
        
        # Scale to target
        w, h = params['resolution']
        filters.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black")
        filters.append(f"fps={params['fps']}")
        
        if self.settings.enable_pixel_jitter:
             # noise filter strength (alls) must be a constant, it doesn't support expressions
             noise_val = random.randint(10, 20)
             filters.append(f"noise=alls={noise_val}:allf=t+u")
        if self.settings.enable_edge_blur:
             sigma_val = random.uniform(0.5, 1.5)
             filters.append(f"gblur=sigma={sigma_val:.2f}")
        if self.settings.enable_subliminal:
             filters.append("drawbox=x='random(1)*w':y='random(2)*h':w=1:h=1:c=black@0.01:t=fill:enable='lt(mod(n,120),1)'")
        if self.settings.enable_color_shift:
             # Use setparams instead of colorspace to avoid errors with unknown input metadata
             filters.append("setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709")
        if self.settings.enable_subtitles:
             import string
             rand_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
             # Added fontfile for Windows robustness
             f_path = "C\\:/Windows/Fonts/arial.ttf"
             filters.append(f"drawtext=fontfile='{f_path}':text='{rand_str}':x=(w-tw)/2:y=h-20:fontcolor=white@0.002:fontsize=12")
        if self.settings.enable_dynamic_crop:
             filters.append("crop=w=iw*0.99:h=ih*0.99:x='(iw-ow)/2 + (iw*0.005)*sin(t*0.5)':y='(ih-oh)/2 + (ih*0.005)*cos(t*0.7)'")
        if self.settings.enable_sensor_noise:
             filters.append("hqdn3d=1.5:1.5:6:6,noise=alls=4:allf=t")
        if self.settings.enable_lum_jitter:
             # Use hue for jitter as eq doesn't support temporal expressions consistently
             filters.append("hue=b='0.001*sin(t*10)'")
        if self.settings.enable_speed_warp:
             filters.append("setpts='PTS*(1 + 0.005*sin(T*0.2))'")
        if self.settings.enable_chroma_jitter:
             # colorbalance parameters are constants, they don't support expressions
             rj = random.uniform(-0.03, 0.03)
             bj = random.uniform(-0.03, 0.03)
             filters.append(f"colorbalance=rs={rj:.3f}:bs={bj:.3f}")
        if self.settings.enable_tint_flash:
             filters.append("hue=h='0.5*sin(t*5)':s='1+0.02*sin(t*4)'")

        # Force YUV420P for compatibility
        filters.append("format=yuv420p")
        
        # Assemble the chain with segment-based label management
        valid_filters = [f.strip(' ,;') for f in filters if f and f.strip(' ,;')]
        if not valid_filters:
            return "[0:v]null[vout]"

        # Robust assembly logic
        full_chain_parts = []
        last_label = "[0:v]"
        for i, f in enumerate(valid_filters):
            # Unique next label for this step
            nxt = f"[v_tr_{i+1}]"
            # If it's the very last one, use [vout]
            if i == len(valid_filters) - 1:
                nxt = "[vout]"
            
            # If the filter already defines its own inputs/outputs (contains semicolons or starts with [)
            # we treat it as a more complex segment that needs careful joining
            if ';' in f or f.startswith('['):
                # Ensure the complex filter connects to the last label
                # This part is tricky as complex filters might handle multiple streams
                # But for our linear-ish chain, we assume they take the 'current' stream
                # unless they are something like overlay/blend already handled
                if not f.startswith('['):
                    full_chain_parts.append(f"{last_label}{f}{nxt}")
                else:
                    full_chain_parts.append(f"{f}{nxt}")
            else:
                full_chain_parts.append(f"{last_label}{f}{nxt}")
            
            # The next filter needs the output label of this filter as its input
            # BUT if 'f' itself produced a different final label than 'nxt' (rare with our flow), 
            # we'd need to extract it. For our current logic, 'nxt' is forced at the end.
            last_label = nxt
        
        vf_chain = ";".join(full_chain_parts)
        
        if params.get('watermark_path'):
             wm_w = int(params['resolution'][0] * 0.15)
             # Re-map the current [vout] to pre-watermark
             vf_chain = vf_chain.replace("[vout]", "[v_pre_wm]")
             return f"{vf_chain};[1:v]scale={wm_w}:-1[wm];[v_pre_wm][wm]overlay=W-w-20:H-h-20[vout]"
        else:
             return vf_chain

    
    def _build_ffmpeg_command(
        self,
        input_path: str,
        output_path: str,
        params: dict,
        video_complex_str: str
    ) -> list[str]:
        """Build complete FFmpeg command using filter_complex"""
        
        codec_name, _ = params['codec']
        
        cmd = [
            'ffmpeg',
            '-y', 
            '-i', input_path,
        ]
        
        # Secondary inputs
        if params.get('watermark_path'):
             cmd.extend(['-i', params['watermark_path']])
             
        cmd.extend(['-async', '1'])
        
        # Combine video and audio into filter_complex
        # Cleaning both strings to be absolutely sure there are no trailing separators
        v_str = video_complex_str.strip('; ,')
        a_str = params.get('audio_filters', "[0:a]anull[aout]").strip('; ,')
        
        cmd.extend(['-filter_complex', f"{v_str};{a_str}"])
        cmd.extend(['-map', '[vout]', '-map', '[aout]'])
        
        # Encoding parameters
        cmd.extend(['-c:v', codec_name])
        if 'nvenc' in codec_name:
            cmd.extend(['-preset', 'p4', '-rc', 'vbr', '-cq', str(params['crf'])])
        else:
            cmd.extend(['-crf', str(params['crf']), '-preset', 'medium'])
        
        cmd.extend([
            '-g', str(params['gop_size']),
            '-bf', str(params['b_frames']),
            '-b:v', f"{int(params['bitrate'] * 1000)}k",
        ])
        
        # Audio codec
        cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
        cmd.append(output_path)
        
        return cmd
    
    def _apply_metadata(self, video_path: str, params: dict):
        """Apply metadata to output video"""
        # Choose theme based on filename/context if possible, or random
        try:
            randomize_metadata(video_path)
        except Exception as e:
            print(f"Failed to apply metadata: {e}")

    def _save_report(self, output_path: str, params: dict):
        """Save a text report detailing all applied transformations"""
        report_path = os.path.splitext(output_path)[0] + "_report.txt"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=== VIDEO UNIQUEIZER PRO - PROCESSING REPORT ===\n")
                f.write(f"Target: {os.path.basename(output_path)}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 50 + "\n\n")
                
                f.write("[1] GEOMETRY & BASICS\n")
                f.write(f"  - Speed: {params.get('speed', 1.0):.4f}x\n")
                f.write(f"  - Resolution: {params.get('resolution')}\n")
                f.write(f"  - FPS: {params.get('fps')}\n")
                if params.get('rotation'): 
                    f.write(f"  - Rotation: {params['rotation']:.2f}° (with Auto-Zoom)\n")
                if params.get('crop_percent'):
                    f.write(f"  - Zoom/Crop: {params['crop_percent']:.2f}%\n")
                if params.get('smart_fit_active'):
                    f.write("  - SmartFit 59s: ENABLED (Auto-optimized for Social Media)\n")
                if params.get('speed_ramp_active'):
                    f.write("  - Dynamic Speed Ramp: ENABLED (Variable Tempo)\n")
                if params.get('mirror'):
                    f.write(f"  - Horizontal Mirror: ENABLED\n")
                
                f.write(f"  - Random Trim (Start): {params['trim_start']:.3f}s\n")
                f.write(f"  - Random Trim (End): {params['trim_end']:.3f}s\n")
                f.write(f"  - Final Duration: {params.get('final_duration', 0):.2f}s\n")
                
                f.write("\n[2] COLOR & LOOK\n")
                f.write(f"  - Brightness/Contrast/Saturation: Adjusted\n")
                if params.get('lut_path') and params.get('lut_strength', 0) > 0.01:
                    f.write(f"  - Cinematic LUT: Applied ({os.path.basename(params['lut_path'])})\n")
                elif params.get('lut_path'):
                    f.write(f"  - Cinematic LUT: Skipped in this variant (Random choice)\n")
                
                if params.get('noise'):
                    f.write(f"  - Film Grain/Noise: {params['noise']:.2f}%\n")
                if params.get('grid_opacity'):
                    f.write(f"  - Transparent Grid Overlay: ENABLED ({params['grid_opacity']:.3f} opacity)\n")
                
                f.write("\n[3] VFX & UNIQUEIZATION\n")
                vfx = params.get('vfx_params')
                uniqueness_score = 80 # Base score + Trim is mandatory unique
                if vfx:
                    if getattr(vfx, 'glitch_enabled', False): 
                        f.write("  - Smart Glitch: ENABLED\n")
                        uniqueness_score += 5
                    if getattr(vfx, 'rgb_split_enabled', False): 
                        f.write("  - RGB Split: ENABLED\n")
                        uniqueness_score += 5
                    if getattr(vfx, 'shake_enabled', False): 
                        f.write("  - Pro Handheld Shake: ENABLED (Smooth steadycam feel)\n")
                        uniqueness_score += 2
                    if getattr(vfx, 'light_leak_enabled', False): 
                        f.write("  - Cinematic Light Leaks: ENABLED (Warm gold/red overlays)\n")
                        uniqueness_score += 3
                    if getattr(vfx, 'film_grain_enabled', False): 
                        f.write("  - Professional Film Grain: ENABLED (Authentic texture)\n")
                        uniqueness_score += 3
                    if getattr(vfx, 'chromatic_aberration_enabled', False): 
                        f.write("  - Chromatic Aberration: ENABLED\n")
                        uniqueness_score += 3
                    if getattr(vfx, 'frame_shuffle_enabled', False): 
                        f.write("  - Frame Shuffling: ENABLED\n")
                        uniqueness_score += 8
                    if getattr(vfx, 'frame_drop_enabled', False): 
                        f.write(f"  - Intelligent Frame Dropping: Every {vfx.frame_drop_interval} frames\n")
                        uniqueness_score += 5
                
                if self.settings.enable_neural:
                    f.write("  - Neural Style Transfer: ENABLED (Subtle Artistic Refinement)\n")
                    uniqueness_score = 100 # Pre-max for neural
                
                if params.get('speed') != 1.0: uniqueness_score += 5
                if params.get('rotation'): uniqueness_score += 5
                
                f.write("\n[4] AUDIO\n")
                a_params = params.get('audio_params')
                if a_params:
                    f.write(f"  - Pitch Shift: {a_params.pitch_semitones:.2f} semitones\n")
                    f.write(f"  - Final Audio Tempo: Adjusted for perfect sync\n")
                    if a_params.reverb_percent > 1: f.write("  - Reverb: Applied\n")
                    if a_params.bass_boost_db > 1: f.write("  - Bass Boost: Applied\n")
                    if getattr(a_params, 'white_noise_level', 0) > 0.0001:
                        f.write(f"  - Unique Background Noise: ENABLED (Subtle White Noise)\n")
                    uniqueness_score += 5
                
                f.write("\n[5] METADATA (Spoofing)\n")
                f.write(f"  - Device: Apple {params.get('device')}\n") # Corrected syntax
                f.write(f"  - Software: {params.get('software')}\n") # Corrected syntax
                f.write(f"  - Encoding: {params.get('codec')[1]}\n") # Corrected syntax
                
                f.write("\n[6] SOCIAL MEDIA ASSETS (Auto-Generated)\n")
                # Generate simple relevant hashtags and a description
                try:
                    from config import COMMENT_TAGS_POOL
                    tags = random.sample(COMMENT_TAGS_POOL, min(5, len(COMMENT_TAGS_POOL)))
                    filename_clean = os.path.basename(output_path).split('_')[0].replace('_', ' ').title()
                    f.write(f"  - Recommended Title: {filename_clean} 🔥 #viral\n")
                    f.write(f"  - Recommended Tags: {' '.join(tags)}\n")
                    f.write(f"  - Description: Check out this epic {filename_clean} edit! Don't forget to like and subscribe for more contents. \n")
                except Exception as ex:
                    f.write(f"  - (Presets loading...) {ex}\n")
                
                f.write("\n" + "=" * 50 + "\n")
                f.write(f"OVERALL UNIQUENESS INDEX: {min(100, uniqueness_score)}%\n")
                
        except Exception as e:
            print(f"Failed to create report: {e}")
