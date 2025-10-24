import os
import json
import re
import subprocess
from typing import List, Optional
import requests
from gtts import gTTS
import streamlit as st
from openai import OpenAI
from PIL import Image

# ✅ Initialize OpenAI client (make sure OPENAI_API_KEY is set in environment)

# ======================
# OPENAI CLIENT
# ======================
from dotenv import load_dotenv

# Initialize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pillow compatibility patch for ANTIALIAS (Pillow >= 10)
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS



# -------------------------------
# IMAGE FETCHER (Pollinations)
# -------------------------------
def fetch_image_pollinations(keyword: str, story: str, index: int) -> Optional[str]:
    safe_keyword = keyword.replace(" ", "_")
    save_as = f"image_{index}_{safe_keyword}.jpg"
    prompt = f"{keyword} - {story} - children's storybook illustration, traditional style"
    url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
    try:
        r = requests.get(url, stream=True, timeout=30)
        if r.status_code == 200:
            with open(save_as, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            return save_as
        else:
            st.warning(f"⚠️ Pollinations failed for '{keyword}' (status {r.status_code})")
            return None
    except Exception as e:
        st.error(f"Error fetching image for {keyword}: {e}")
        return None

# -------------------------------
# AUDIO GENERATOR
# -------------------------------
def generate_audio(text: str, lang: str = "en") -> str:
    audio_path = "story_audio.mp3"
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        st.error(f"⚠️ Audio generation failed: {e}")
        return None

# -------------------------------
# VIDEO CREATOR
# -------------------------------
def get_audio_duration(audio_path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0

def create_slideshow(image_files: List[str], audio_file: str, output_file: str = "slideshow.mp4") -> bool:
    if not image_files or not audio_file:
        return False

    audio_duration = get_audio_duration(audio_file)
    if audio_duration <= 0:
        return False

    per_image_duration = audio_duration / len(image_files)
    txt_path = "slideshow.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for img in image_files[:-1]:
            f.write(f"file '{img}'\n")
            f.write(f"duration {per_image_duration}\n")
        f.write(f"file '{image_files[-1]}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", txt_path,
        "-i", audio_file, "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", output_file
    ]
    return subprocess.run(cmd).returncode == 0

# -------------------------------
# LLM HELPERS
# -------------------------------
def generate_story_from_prompt(prompt: str, word_limit: int = 120) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a creative storyteller."},
            {"role": "user",
             "content": f"Write a children's story of around {word_limit} words based on this prompt: {prompt}"}
        ],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def extract_keywords_from_story(story: str, max_keywords: int = 5) -> List[str]:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "Extract 5 main visual keywords (important nouns or characters) for illustration from the given children's story. Respond with a simple JSON list of strings only."},
                {"role": "user", "content": story}
            ],
            temperature=0.3,
            max_tokens=100
        )
        raw_output = response.choices[0].message.content.strip()
       

        cleaned_output = re.sub(r"^```.*?json|```$", "", raw_output, flags=re.MULTILINE).strip()
        try:
            keywords = json.loads(cleaned_output)
        except Exception:
            cleaned = [kw.strip("*- \n") for kw in cleaned_output.splitlines() if kw.strip()]
            keywords = [k for k in cleaned if len(k.split()) <= 3]
        return keywords[:max_keywords]
    except Exception as e:
        st.error(f"Keyword extraction failed: {e}")
        return []

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="AI Story → Video", layout="wide")
st.title("📖 AI Story → 🎬 Video Slideshow")

# Language selection FIRST
languages = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "Telugu (తెలుగు)": "te",
    "Tamil (தமிழ்)": "ta",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Malayalam (മലയാളം)": "ml",
    "Bengali (বাংলা)": "bn",
    "Gujarati (ગુજરાતી)": "gu",
    "Punjabi (ਪੰਜਾਬੀ)": "pa"
}
lang_display = st.selectbox("🌐 Select narration language:", list(languages.keys()))
lang_code = languages[lang_display]

prompt = st.text_input("✍️ Enter your story prompt (e.g., *Akbar and Birbal story of 100 words*):")

if st.button("🚀 Generate Story Audio/Video"):
    if not prompt.strip():
        st.error("⚠️ Please enter a story prompt first.")
    else:
        progress = st.progress(0)
        progress.progress(10)

        st.info("📖 Generating story...")
        story = generate_story_from_prompt(prompt)
        progress.progress(30)

        st.success("✅ Story Generated!")
        st.write(story)

        st.info("🔑 Extracting illustration keywords...")
        keywords = extract_keywords_from_story(story)
        
        progress.progress(50)

        image_files = []
        if keywords:
            st.info("🎨 Generating illustrations...")
            for i, kw in enumerate(keywords):
                img = fetch_image_pollinations(kw, story, i)
                if img:
                    image_files.append(img)
                   
            progress.progress(70)

        st.info("🎙️ Generating audio narration...")
        audio_file = generate_audio(story, lang=lang_code)
        if audio_file:
            st.audio(audio_file)
        progress.progress(85)

        if image_files and audio_file:
            st.info("🎬 Creating slideshow video...")
            success = create_slideshow(image_files, audio_file, "slideshow.mp4")
            if success:
                st.success("✅ Video created successfully!")
                progress.progress(100)

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.video("slideshow.mp4")
                with col2:
                    st.download_button("⬇️ Download Video", open("slideshow.mp4", "rb"),
                                       file_name="slideshow.mp4", mime="video/mp4")
                    st.download_button("⬇️ Download Audio", open(audio_file, "rb"),
                                       file_name="story_audio.mp3", mime="audio/mpeg")
            else:
                st.error("❌ Failed to generate slideshow video.")
        else:
            st.warning("⚠️ Missing images or audio. Cannot create slideshow.")
