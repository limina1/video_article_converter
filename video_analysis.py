import numpy as np
import matplotlib.pyplot as plt
import json
import os
import cv2
from tqdm import trange
import util
from importlib import reload
reload(util)
import anthropic
import gui

# Initialize the Anthropic client
with open('/home/user/.apis/video-articles.api', 'r') as file:
    api_key = file.read().strip()
    os.environ['ANTHROPIC_API_KEY'] = api_key
client = anthropic.Anthropic()

def process_video(url, video_title, max_tokens=2334):
    video_file, chapters_file, subtitle_file, video_folder = util.video_download_workflow(url)

    if not video_file:
        print("Video processing failed. Exiting.")
        return

    print(f"Video file: {video_file}")
    print(f"Chapters file: {chapters_file}")
    print(f"Subtitle file: {subtitle_file}")

    if os.path.exists(chapters_file):
        with open(chapters_file, 'r') as f:
            print(f"Loading chapters from {chapters_file}")
            try:
                chapters = json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading chapters from {chapters_file}. Using whole video as one chapter.")
                chapters = [{'title': 'Whole Video', 'start_time': 0}]
    else:
        print(f"Warning: Chapters file {chapters_file} not found. Using whole video as one chapter.")
        chapters = [{'title': 'Whole Video', 'start_time': 0}]

    frame_skip = int(input("How many frames to skip? (default 10) ") or 10)

    video_data, fps = util.load_video_data(video_file, chapters, frame_skip=frame_skip)

    # Run the GUI for frame selection
    gui.main(video_data, fps, chapters, video_folder)

    # After GUI closes, process the selected frames
    selected_frames_dir = find_latest_selected_frames_dir(video_folder)
    if selected_frames_dir:
        generated_content = process_selected_frames(selected_frames_dir, subtitle_file, video_title, chapters, max_tokens)
        output_file = os.path.join(video_folder, f"{util.sanitize_filename(video_title)}_content.adoc")
        util.save_as_asciidoc(generated_content, video_title, output_file)
    else:
        print("No selected frames found. Skipping content generation.")

def find_latest_selected_frames_dir(video_folder):
    selected_frames_dirs = [d for d in os.listdir(video_folder) if d.startswith("selected_frames_")]
    if not selected_frames_dirs:
        return None
    return os.path.join(video_folder, max(selected_frames_dirs))
def process_selected_frames(selected_frames_dir, subtitle_file, video_title, chapters, max_tokens):
    metadata_file = os.path.join(selected_frames_dir, "metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Group images by chapter
    chapter_images = {chapter['title']: [] for chapter in chapters}
    for item in metadata:
        chapter_images[item['chapter_title']].append({
            'image': cv2.imread(os.path.join(selected_frames_dir, item['filename'])),
            'timestamp': item['timestamp']
        })

    generated_content = []
    for chapter in chapters:
        chapter_title = chapter['title']
        start_time = chapter['start_time']
        end_time = chapter.get('end_time')  # This might be None for the last chapter

        # Read subtitle section for the entire chapter
        subtitle_section = util.read_subtitle_section(subtitle_file, start_time, end_time)

        # Generate content for the chapter
        content = util.generate_content(chapter_title, video_title, subtitle_section, max_tokens)

        # Add images for this chapter
        content['images'] = chapter_images.get(chapter_title, [])

        generated_content.append({
            'chapter_title': chapter_title,
            'content': content['content'],
            'images': content['images']
        })

    return generated_content

if __name__ == "__main__":
    video_url = input("Enter the YouTube video URL: ")
    video_title = input("Enter a title for the video: ")
    max_tokens = int(input("Enter the maximum number of tokens for content generation (default 2334): ") or 2334)
    process_video(video_url, video_title, max_tokens)
