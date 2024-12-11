import asyncio
import os
import re
import json
from openai import AsyncOpenAI
import aiofiles
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi
from config import load_api_key

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=load_api_key(),
)

MAIN_PROMPT = "Normalize the transcript of the podcast. It was automatically generated, so it contains mistakes and incorrectly transcribed terms. Start normalizing the transcript from the start to the end, and don't output anything else. Use the provided glossary and documents as references for correct spellings and terms. Stop when you hit the rate limit."

# Cache directory for storing transcripts
CACHE_DIR = "transcript_cache"


def ensure_cache_dir():
    """Ensure the cache directory exists."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)


def get_cache_path(video_id: str) -> str:
    """Get the cache file path for a video ID."""
    return os.path.join(CACHE_DIR, f"{video_id}.json")


async def read_from_cache(video_id: str):
    """Read transcript data from cache if it exists."""
    cache_path = get_cache_path(video_id)
    if os.path.exists(cache_path):
        async with aiofiles.open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.loads(await f.read())
            return cache_data["title"], cache_data["transcript"]
    return None, None


async def write_to_cache(video_id: str, title: str, transcript: str):
    """Write transcript data to cache."""
    cache_path = get_cache_path(video_id)
    cache_data = {"title": title, "transcript": transcript}
    async with aiofiles.open(cache_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(cache_data, ensure_ascii=False, indent=2))


async def fetch_transcript(video_url: str):
    """Fetch the YouTube transcript and title given a URL."""
    ensure_cache_dir()

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": "%(id)s.%(ext)s",
        "noplaylist": True,
        "writesubtitles": True,
        "subtitleslangs": ["en"],
        "skip_download": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        try:
            # First get the video ID
            info_dict = ydl.extract_info(video_url, download=False)
            video_id = info_dict["id"]

            # Check cache first
            title, transcript = await read_from_cache(video_id)
            if title and transcript:
                print(f"Using cached transcript for video: {video_id}")
                return title, transcript

            # If not in cache, fetch from YouTube
            title = info_dict["title"]

            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript = None
                for t in transcript_list:
                    if t.language_code.startswith("en"):
                        transcript = t.fetch()
                        break
                if not transcript:
                    transcript = transcript_list.find_transcript(
                        transcript_list._manually_created_transcripts
                        + transcript_list._generated_transcripts
                    ).fetch()
            except Exception as e:
                print(f"Error fetching transcript for {video_url}: {e}")
                transcript = []

            full_text = " ".join([seg["text"] for seg in transcript])

            # Cache the results
            await write_to_cache(video_id, title, full_text)

            return title, full_text
        except Exception as e:
            print(f"Error fetching video info for {video_url}: {e}")
            return None, None


async def call_openai_api(prompt: str):
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


async def normalize_transcript(title: str, transcript: str, glossary: str):
    prompt = (
        "<START GAINING KNOWLEDGE>"
        + glossary
        + "<END GAINING KNOWLEDGE> <START AUTOMATICALLY GENERATED TRANSCRIPT>"
        + title
        + "\n\n"
        + transcript
        + "<END AUTOMATICALLY GENERATED TRANSCRIPT> <START GUIDELINES FOR THE ASSISTANT> Normalize this transcript verbatim from the start to the end and do not write anything else. It was automatically generated, so it contains mistakes and incorrectly transcribed terms that you should already know. Stop only when you hit the length limit. Do not output that you've reached the limit, instead, if you reach the end of the transcript, write THE END. <END GUIDELINES FOR THE ASSISTANT> <START NORMALIZED TRANSCRIPT>"
    )
    while True:
        result = await call_openai_api(prompt)
        print(result[:20] + "...")
        prompt += result
        if "THE END" in result:
            break

    return prompt.split("<START NORMALIZED TRANSCRIPT>")[1]


def sanitize_filename(filename: str):
    """Create a safe filename from a title."""
    filename = filename.strip().replace(" ", "_")
    filename = re.sub(r"[^A-Za-z0-9_\-\.]", "", filename)
    return filename[:200]


async def process_video(url, glossary):
    title, transcript = await fetch_transcript(url)
    if not transcript:
        print(f"No transcript found for: {url}")
        return

    print(f"Normalizing transcript for: {title}")
    normalized = await normalize_transcript(title, transcript, glossary)
    safe_title = sanitize_filename(title) or "video"

    if not os.path.isdir("outputs"):
        os.mkdir("outputs")
    output_file = "outputs/" + safe_title + ".txt"
    async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
        await f.write(normalized)

    print(f"Saved normalized transcript to {output_file}")


def main():
    glossary_file = (
        input("Enter the path to the glossary/docs .txt file: ")
        .strip()
        .replace("'", "")
    )
    if not os.path.exists(glossary_file):
        print("Glossary file not found. Please provide a valid file path.")
        return

    with open(glossary_file, "r", encoding="utf-8") as gf:
        glossary = gf.read()

    video_urls = []
    while True:
        video_url = input("Enter YouTube video URL (press Enter to stop): ").strip()
        if video_url == "":
            break
        video_urls.append(video_url)

    if not video_urls:
        print("No video URLs provided.")
        return

    video_urls = list(set(video_urls))

    async def process_all_videos():
        tasks = [process_video(url, glossary) for url in video_urls]
        await asyncio.gather(*tasks)

    asyncio.run(process_all_videos())


if __name__ == "__main__":
    main()
