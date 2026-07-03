import os
import subprocess
import shutil
import cv2

def get_video_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int
) -> cv2.VideoWriter:
    """
    Initializes a cv2.VideoWriter for the temporary silent video.
    Uses 'mp4v' codec as it is universally supported on Windows out-of-the-box.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def finalize_video_with_audio(
    temp_video_path: str,
    source_video_path: str,
    final_output_path: str
):
    """
    Stitches the original audio back onto the processed silent video.
    """
    if not os.path.exists(temp_video_path):
        raise FileNotFoundError(f"Temporary silent video not found: {temp_video_path}")
        
    print(f"[Stitch] Finalizing video: merging audio from {source_video_path}")
    
    out_dir = os.path.dirname(final_output_path) or "."
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_video_path,
        "-i", source_video_path,
        "-map", "0:v",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        final_output_path
    ]
    
    ffmpeg_success = False
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print("[Stitch] FFmpeg completed successfully.")
        ffmpeg_success = True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[Stitch] Warning: FFmpeg integration failed or FFmpeg is not installed.")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"FFmpeg stderr: {e.stderr}")
        print("[Stitch] Copying silent video as final output (no audio)...")
        try:
            shutil.copy2(temp_video_path, final_output_path)
        except Exception as copy_err:
            print(f"[Stitch] Error copying fallback file: {copy_err}")
            
    if os.path.exists(temp_video_path):
        try:
            os.remove(temp_video_path)
            print("[Stitch] Cleaned up temporary silent video.")
        except Exception as ex:
            print(f"[Stitch] Note: could not remove temp file {temp_video_path}: {ex}")
            
    print(f"[Stitch] Process finished. Output written to: {final_output_path}")
    return ffmpeg_success
