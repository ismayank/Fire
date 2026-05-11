import cv2
import os
import subprocess

def convert_to_web_compatible(input_path, output_path):
    """Convert video to web-compatible format using ffmpeg if available"""
    try:
        # Try ffmpeg conversion first (best quality)
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',  # H.264 codec
            '-c:a', 'aac',       # AAC audio
            '-movflags', '+faststart',  # For web streaming
            '-y', output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except:
        # Fallback to OpenCV conversion
        try:
            cap = cv2.VideoCapture(input_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            cap.release()
            out.release()
            return True
        except:
            return False

def is_video_web_compatible(video_path):
    """Check if video is web-compatible"""
    try:
        cap = cv2.VideoCapture(video_path)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()
        
        # Common web-compatible codecs
        web_codecs = ['H264', 'XVID', 'AVC1', 'MP4V']
        codec_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        return any(codec in codec_str for codec in web_codecs)
    except:
        return False
