import os
import shutil
import subprocess
import yt_dlp
from pydub import AudioSegment
from groq import Groq

#  Clean old session
def clean_previous_session():
    BASE_DIR = "user_session"
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)
    os.makedirs(BASE_DIR)

#  Base directory
BASE_DIR = "user_session"

#  Init Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Download full video
def download_full_video(youtube_url, output_name="downloaded"):
    base_output = os.path.join(BASE_DIR, output_name)  # No extension

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': base_output + '.%(ext)s',  # let yt_dlp handle extension
        'merge_output_format': 'mp4',  # ensure merged to mp4
        'noplaylist': True,
        'quiet': False
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            # Try to find merged file
            merged_filename = ydl.prepare_filename(info).rsplit('.', 1)[0] + '.mp4'

        if not os.path.exists(merged_filename):
            raise FileNotFoundError(f"[ERROR] Merged .mp4 file not found: {merged_filename}")

        return merged_filename

    except Exception as e:
        print(f"[ERROR] Failed to download or locate video: {e}")
        raise e

# Convert to mp3
def convert_to_mp3(video_path):
    mp3_path = video_path.rsplit('.', 1)[0] + ".mp3"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[ERROR] MP4 file not found: {video_path}")
    subprocess.run(['ffmpeg', '-y', '-i', video_path, mp3_path], check=True)
    return mp3_path

#  Transcribe video
def transcribe_youtube(link, output_txt=None):
    clean_previous_session()

    if output_txt is None:
        output_txt = os.path.join(BASE_DIR, "transcribed_output.txt")

    video_path = download_full_video(link)
    mp3_path = convert_to_mp3(video_path)

    audio = AudioSegment.from_mp3(mp3_path)
    chunks = [audio[i:i + 5 * 60 * 1000] for i in range(0, len(audio), 5 * 60 * 1000)]
    filenames = []

    for i, chunk in enumerate(chunks):
        fname = os.path.join(BASE_DIR, f"chunk_{i}.mp3")
        chunk.export(fname, format="mp3")
        filenames.append(fname)

    with open(output_txt, "w", encoding="utf-8") as f_out:
        for fname in filenames:
            with open(fname, "rb") as f:
                text = client.audio.transcriptions.create(
                    file=f,
                    model="whisper-large-v3",
                    language="en",
                    response_format="text"
                )
                f_out.write(text + "\n\n")
            os.remove(fname)

    if os.path.exists(mp3_path):
        os.remove(mp3_path)
    if os.path.exists(video_path):
        os.remove(video_path)

    with open(output_txt, "r", encoding="utf-8") as f:
        full_transcript = f.read()

    return full_transcript
