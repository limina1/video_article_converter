import os
import json
import subprocess
from typing import Tuple, Dict, Any, List
import anthropic
import pandas as pd
from pysrt import open as srt_open
import util
from importlib import reload
from datetime import datetime
import argparse
reload(util)

def check_transcript_exists(url: str) -> Tuple[str, str, str, str]:
    print(f"Checking transcript for URL: {url}")

    # Get video title
    title_command = ["yt-dlp", "--get-title", url]
    title_result = subprocess.run(title_command, capture_output=True, text=True)
    if title_result.returncode != 0:
        print(f"Error getting video title: {title_result.stderr}")
        return None, None, None, None

    video_title = title_result.stdout.strip()
    sanitized_title = util.sanitize_filename(video_title)
    video_folder = f"transcripts/{sanitized_title}"

    # Create video folder if it doesn't exist
    os.makedirs(video_folder, exist_ok=True)

    transcript_file = os.path.join(video_folder, f"{sanitized_title}.en.srt")
    chapters_file = os.path.join(video_folder, f"{sanitized_title}_chapters.json")

    if os.path.exists(transcript_file) and os.path.exists(chapters_file):
        print(f"Transcript and chapters files exist: {transcript_file}, {chapters_file}")
    else:
        print("Transcript or chapters file not found. Downloading...")
        transcript_file, chapters_file = download_transcript(url, video_folder, sanitized_title)

    return transcript_file, chapters_file, video_folder, sanitized_title

def download_transcript(url: str, video_folder: str, sanitized_title: str) -> Tuple[str, str]:
    print("Downloading transcript and chapters...")
    command = [
        "yt-dlp",
        "--write-auto-sub",
        "--sub-lang", "en",
        "--skip-download",
        "--write-info-json",
        "--convert-subs", "srt",
        "--output", os.path.join(video_folder, f"{sanitized_title}")
    ]
    result = subprocess.run(command + [url], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error downloading transcript: {result.stderr}")
        return None, None

    transcript_file = os.path.join(video_folder, f"{sanitized_title}.en.srt")
    info_json = os.path.join(video_folder, f"{sanitized_title}.info.json")

    # Extract chapters from info.json
    chapters_file = os.path.join(video_folder, f"{sanitized_title}_chapters.json")
    with open(info_json, 'r') as f:
        info = json.load(f)
        chapters = info.get('chapters', [])
        with open(chapters_file, 'w') as cf:
            json.dump(chapters, cf, indent=2)

    os.remove(info_json)  # Clean up the info.json file

    return transcript_file, chapters_file





def generate_content(chapter_title: str, video_title: str, subtitle_section: str, max_tokens: int = 2334, debug: bool = False) -> Dict[str, Any]:
    system_prompt_template = util.read_prompt_file('system_prompt.md')
    user_prompt_template = util.read_prompt_file('user_prompt.md')
    system_prompt = system_prompt_template.format(
        videotype='podcast',
        videofeatures= 'this is a dialogue between two or more people, but the transcript does not indicate who is speaking',
    )

    user_prompt = user_prompt_template.format(
        chapter_title=chapter_title,
        video_title=video_title,
        subtitle_section=subtitle_section
    )

    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=max_tokens,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
        )
        if debug:
            print(f"Generating content for chapter: {chapter_title}")
            print(f"Subtitle section:\n {subtitle_section}")
            print(f"Content generated for chapter: {chapter_title}")
            print(response.content[0].text)
            print("====================================")
        return {"content": response.content[0].text}

    except anthropic.BadRequestError as e:
        print(f"Error generating content for chapter '{chapter_title}': {e}")
        return {"content": f"Error generating content: {str(e)}"}

def save_as_asciidoc(generated_content: List[Dict[str, str]], video_title: str, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"= {video_title}\n")
        f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(":toc:\n")
        f.write(":toclevels: 3\n")
        f.write(":sectnums:\n\n")

        for item in generated_content:
            f.write(f"== {item['chapter_title']}\n\n")
            f.write(item['content'])
            f.write("\n\n")

    print(f"AsciiDoc content saved to {output_file}")

def transcript_analysis_workflow(url: str, debug: bool = False):
    transcript_file, chapters_file, video_folder, sanitized_title = check_transcript_exists(url)

    if not transcript_file or not chapters_file:
        print("Failed to obtain transcript or chapters. Exiting.")
        return

    processed_chapters = util.process_subtitles_by_chapter(transcript_file, chapters_file)

    generated_content = []
    for chapter_title, chapter_data in processed_chapters.items():
        subtitle_section = util.read_subtitle_section(processed_chapters, chapter_title)
        content = generate_content(chapter_title, sanitized_title, subtitle_section, debug=debug)
        generated_content.append({"chapter_title": chapter_title, "content": content["content"]})

    output_file = os.path.join(video_folder, f"{sanitized_title}_content.adoc")
    save_as_asciidoc(generated_content, sanitized_title, output_file)

if __name__ == "__main__":
    # video_url = input("Enter the YouTube video URL: ")
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    video_url = input("Enter the YouTube video URL: ") or "https://www.youtube.com/watch?v=Cn_Da8gtvUo" # sample video
    transcript_analysis_workflow(video_url, debug=args.debug)
