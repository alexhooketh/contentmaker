import asyncio
import re
import json
import traceback
from typing import Optional, Tuple, List
from pathlib import Path

import requests
from openai import AsyncOpenAI
import aiofiles
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi

from config import load_env_var

# Configuration constants
CACHE_DIR = Path("transcript_cache")
OUTPUTS_DIR = Path("outputs")
TELEGRAM_BOT_TOKEN = load_env_var("TELEGRAM_BOT_TOKEN", prompt_if_missing=False)
TELEGRAM_CHAT_ID = load_env_var("TELEGRAM_CHAT_ID", prompt_if_missing=False)

# OpenAI client initialization
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=load_env_var("OPENROUTER_API_KEY"),
)

def send_telegram_message(message: str) -> None:
    """
    Send a message to Telegram.
    
    Args:
        message: The message to send
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram configuration not found. Skipping notification.")
        return
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=data)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def ensure_directories() -> None:
    """Ensure required directories exist."""
    CACHE_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

def get_cache_path(video_id: str) -> Path:
    """
    Get the cache file path for a video ID.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Path object for the cache file
    """
    return CACHE_DIR / f"{video_id}.json"

async def read_from_cache(video_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Read transcript data from cache if it exists.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Tuple of (title, transcript) if found, else (None, None)
    """
    try:
        cache_path = get_cache_path(video_id)
        if cache_path.exists():
            async with aiofiles.open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.loads(await f.read())
                return cache_data["title"], cache_data["transcript"]
    except Exception as e:
        send_telegram_message(f"❌ Error reading from cache: {str(e)}\n\n{traceback.format_exc()}")
    return None, None

async def write_to_cache(video_id: str, title: str, transcript: str) -> None:
    """
    Write transcript data to cache.
    
    Args:
        video_id: YouTube video ID
        title: Video title
        transcript: Video transcript
    """
    try:
        cache_path = get_cache_path(video_id)
        cache_data = {"title": title, "transcript": transcript}
        async with aiofiles.open(cache_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(cache_data, ensure_ascii=False, indent=2))
    except Exception as e:
        send_telegram_message(f"❌ Error writing to cache: {str(e)}\n\n{traceback.format_exc()}")

async def fetch_transcript(video_url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch the YouTube transcript and title given a URL.
    
    Args:
        video_url: YouTube video URL
        
    Returns:
        Tuple of (title, transcript) if successful, else (None, None)
    """
    try:
        ensure_directories()

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": "%(id)s.%(ext)s",
            "noplaylist": True,
            "writesubtitles": True,
            "subtitleslangs": ["en"],
            "skip_download": True,
        }

        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            video_id = info_dict["id"]
            title = info_dict["title"]

            # Check cache first
            cached_title, cached_transcript = await read_from_cache(video_id)
            if cached_title and cached_transcript:
                print(f"Using cached transcript for video: {video_id}")
                return cached_title, cached_transcript

            # Fetch transcript from YouTube
            transcript = await get_youtube_transcript(video_id)
            if transcript:
                await write_to_cache(video_id, title, transcript)
                return title, transcript

    except Exception as e:
        error_msg = f"❌ Error fetching video info for {video_url}: {str(e)}\n\n{traceback.format_exc()}"
        send_telegram_message(error_msg)
        print(error_msg)
    
    return None, None

async def get_youtube_transcript(video_id: str) -> Optional[str]:
    """
    Get transcript for a YouTube video.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Transcript text if successful, else None
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        
        # Try to get English transcript first
        for t in transcript_list:
            if t.language_code.startswith("en"):
                transcript = t.fetch()
                break
                
        # Fall back to any available transcript
        if not transcript:
            transcript = transcript_list.find_transcript(
                transcript_list._manually_created_transcripts
                + transcript_list._generated_transcripts
            ).fetch()
            
        return " ".join([seg["text"] for seg in transcript]) if transcript else None
        
    except Exception as e:
        error_msg = f"❌ Error fetching transcript: {str(e)}\n\n{traceback.format_exc()}"
        send_telegram_message(error_msg)
        print(error_msg)
        return None

async def call_openai_api(prompt: str) -> Optional[str]:
    """
    Call OpenAI API for transcript normalization.
    
    Args:
        prompt: Input prompt for the API
        
    Returns:
        API response text if successful, else None
    """
    try:
        response = await client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet:beta",
            messages=[
                {
                    "role": "system",
                    "content": "You need to continue generation that was left by the previous AI assistant. DO NOT output anything which is not explicitly defined in its guidelines.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"❌ Error calling OpenAI API: {str(e)}\n\n{traceback.format_exc()}"
        send_telegram_message(error_msg)
        print(error_msg)
        raise

async def normalize_transcript(title: str, transcript: str, glossary: str) -> Optional[str]:
    """
    Normalize transcript using OpenAI API.
    
    Args:
        title: Video title
        transcript: Raw transcript
        glossary: Reference glossary
        
    Returns:
        Normalized transcript if successful, else None
    """
    try:
        prompt = (
            "<START GAINING KNOWLEDGE>"
            + glossary
            + "<END GAINING KNOWLEDGE> <START AUTOMATICALLY GENERATED TRANSCRIPT>"
            + title
            + "\n\n"
            + transcript
            + "<END AUTOMATICALLY GENERATED TRANSCRIPT> <START GUIDELINES FOR THE ASSISTANT> "
            "Normalize this transcript verbatim from the start to the end. Do not write anything else. "
            "The original transcript was automatically generated, so it contains mistakes and incorrectly transcribed terms that you should already know. "
            "Stop only when you hit the length limit. Do not output that you've reached the limit. If you reach the end of the original transcript, write THE END. "
            "<END GUIDELINES FOR THE ASSISTANT> <START NORMALIZED TRANSCRIPT>"
        )
        
        prev_result = ""
        while True:
            result = await call_openai_api(prompt)
            if not result:
                return None
                
            print("Processing:", result)
            # sometimes it gives the equivalent stuff
            if result[:20] != prev_result[:20]:
                prompt += result
            
            if result.strip().endswith("THE END") and all([x not in result for x in ["[", "]"]]):
                break

        return prompt.split("<START NORMALIZED TRANSCRIPT>")[1]
        
    except Exception as e:
        error_msg = f"❌ Error normalizing transcript: {str(e)}\n\n{traceback.format_exc()}"
        send_telegram_message(error_msg)
        print(error_msg)
        raise

def sanitize_filename(filename: str) -> str:
    """
    Create a safe filename from a title.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    filename = filename.strip().replace(" ", "_")
    filename = re.sub(r"[^A-Za-z0-9_\-\.]", "", filename)
    return filename[:200]

async def process_video(url: str, glossary: str) -> None:
    """
    Process a single video URL.
    
    Args:
        url: YouTube video URL
        glossary: Reference glossary
    """
    try:
        title, transcript = await fetch_transcript(url)
        if not transcript:
            error_msg = f"❌ No transcript found for: {url}"
            send_telegram_message(error_msg)
            print(error_msg)
            return

        print(f"Normalizing transcript for: {title}")
        normalized = await normalize_transcript(title, transcript, glossary)
        if not normalized:
            return

        safe_title = sanitize_filename(title) or "video"
        output_file = OUTPUTS_DIR / f"{safe_title}.txt"
        
        async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
            await f.write(normalized)

        success_msg = f"✅ Successfully processed video:\nTitle: {title}\nOutput: {output_file}"
        send_telegram_message(success_msg)
        print(f"Saved normalized transcript to {output_file}")
        
    except Exception as e:
        error_msg = f"❌ Error processing video {url}: {str(e)}\n\n{traceback.format_exc()}"
        send_telegram_message(error_msg)
        print(error_msg)

def main() -> None:
    """Main execution function."""
    try:
        glossary_file = Path(input("Enter the path to the glossary/docs .txt file: ").strip().replace("'", ""))
        if not glossary_file.exists():
            error_msg = "❌ Glossary file not found. Please provide a valid file path."
            send_telegram_message(error_msg)
            print(error_msg)
            return

        with open(glossary_file, "r", encoding="utf-8") as gf:
            glossary = gf.read()

        video_urls: List[str] = []
        while True:
            video_url = input("Enter YouTube video URL (press Enter to stop): ").strip()
            if not video_url:
                break
            video_urls.append(video_url)

        if not video_urls:
            print("No video URLs provided.")
            return

        video_urls = list(set(video_urls))  # Remove duplicates

        async def process_all_videos() -> None:
            tasks = [process_video(url, glossary) for url in video_urls]
            await asyncio.gather(*tasks)

        asyncio.run(process_all_videos())
        
    except Exception as e:
        error_msg = f"❌ Error in main function: {str(e)}\n\n{traceback.format_exc()}"
        send_telegram_message(error_msg)
        print(error_msg)

if __name__ == "__main__":
    main()
