import numpy as np
import matplotlib.pyplot as plt
import subprocess
import json
import os
import yt_dlp
import cv2
from tqdm import trange, tqdm
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
import re
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import glob
import math
from matplotlib.widgets import Slider, Button, RectangleSelector, CheckButtons
from matplotlib.patches import Rectangle
from matplotlib import cm
import anthropic
import base64
from pysrt import open as srt_open
import os
import json
import logging
import cv2
from typing import List, Dict, Any, Tuple
from datetime import datetime
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import pickle

def calculate_frame_entropy(frame):
    histogram = cv2.calcHist([frame], [0], None, [256], [0, 256])
    histogram = histogram.ravel() / histogram.sum()
    logs = np.log2(histogram + 1e-10)  # Add small value to avoid log(0)
    entropy = -np.sum(histogram * logs)
    return entropy

def get_user_resolution_choice(available_resolutions):
    while True:
        print("\nAvailable resolutions:")
        for i, res in enumerate(available_resolutions, 1):
            print(f"{i}. {res}")
        choice = input("Enter the number of your desired resolution (or 'q' to quit): ")

        if choice.lower() == 'q':
            return None

        try:
            index = int(choice) - 1
            if 0 <= index < len(available_resolutions):
                return available_resolutions[index].rstrip('p')
        except ValueError:
            pass

        print("Invalid choice. Please try again.")
def calculate_frame_entropy(frame):
    histogram = cv2.calcHist([frame], [0], None, [256], [0, 256])
    histogram = histogram.ravel() / histogram.sum()
    logs = np.log2(histogram + 1e-10)  # Add small value to avoid log(0)
    entropy = -np.sum(histogram * logs)
    return entropy
def filter_similar_frames(chapter_data, similarity_threshold=0.95):
    filtered_chapter_data = []

    for chapter in chapter_data:
        filtered_diagrams = []
        if chapter['diagrams']:
            filtered_diagrams = [chapter['diagrams'][0]]  # Keep the first frame

            for i in trange(1, len(chapter['diagrams']), desc=f"Filtering {chapter['title']}"):
                current_timestamp, current_frame = chapter['diagrams'][i]
                prev_timestamp, prev_frame = filtered_diagrams[-1]

                # Convert frames to grayscale for SSIM comparison
                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

                # Compute SSIM between the two frames
                similarity, _ = ssim(prev_gray, current_gray, full=True)

                if similarity < similarity_threshold:
                    filtered_diagrams.append((current_timestamp, current_frame))

        filtered_chapter_data.append({
            'title': chapter['title'],
            'diagrams': filtered_diagrams,
            'entropies': chapter['entropies']
        })

    return filtered_chapter_data
def load_video_data(video_path, chapters, frame_skip=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
        print("Warning: Couldn't determine FPS. Assuming 30 FPS.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_data = []
    for chapter in chapters:
        start_frame = int(chapter['start_time'] * fps)
        end_frame = int(chapter.get('end_time', total_frames / fps) * fps)

        chapter_frames = []
        chapter_entropies = []
        chapter_timestamps = []

        for frame_number in trange(start_frame, end_frame, frame_skip, desc=f"Loading {chapter['title']}"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                entropy = calculate_frame_entropy(gray_frame)
                chapter_frames.append(frame)
                chapter_entropies.append(entropy)
                chapter_timestamps.append(frame_number / fps)

        video_data.append({
            'title': chapter['title'],
            'frames': chapter_frames,
            'entropies': chapter_entropies,
            'timestamps': chapter_timestamps
        })

    cap.release()
    return video_data, fps
def plot_entropy_over_time(video_data, entropy_threshold=4.5, change_threshold=0.5, persistence_threshold=3, stability_window=5, plot=True):
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_entropy = fig.add_subplot(gs[0])
    ax_changes = fig.add_subplot(gs[1], sharex=ax_entropy)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(video_data)))

    all_change_points = []

    for chapter, color in zip(video_data, colors):
        timestamps = chapter['timestamps']
        entropies = chapter['entropies']

        ax_entropy.plot(timestamps, entropies, label=chapter['title'], color=color, alpha=0.7)

        # Detect improved change points
        change_points = improved_detect_entropy_changes(entropies, timestamps,
                                                        entropy_threshold=entropy_threshold,
                                                        change_threshold=change_threshold,
                                                        persistence_threshold=persistence_threshold,
                                                        stability_window=stability_window)
        all_change_points.extend(change_points)

        # Mark extracted diagrams if available
        if 'diagrams' in chapter:
            diagram_timestamps = [d[0] for d in chapter['diagrams']]
            diagram_entropies = [entropies[timestamps.index(t)] for t in diagram_timestamps]
            ax_entropy.scatter(diagram_timestamps, diagram_entropies, marker='o', s=50, zorder=5, color=color)

    # Sort all change points by timestamp
    all_change_points.sort(key=lambda x: x[0])

    # Plot change points with duration and staggered horizontal lines
    for i, (start_time, end_time) in enumerate(all_change_points):
        duration = end_time - start_time

        # Vertical lines for duration in entropy plot
        ax_entropy.axvspan(start_time, end_time, facecolor='g', alpha=0.1)

        # Horizontal line for change in the changes plot, staggered
        y_position = i % 5
        ax_changes.hlines(y=y_position, xmin=start_time, xmax=end_time, color='g', linewidth=2, alpha=0.7)

        # Add text to show duration in the changes plot
        ax_changes.text(start_time, y_position, f'{duration:.2f}s', verticalalignment='bottom',
                        horizontalalignment='left', rotation=45, fontsize=8)

    # Add horizontal line for entropy threshold
    ax_entropy.axhline(y=entropy_threshold, color='r', linestyle='--', label='Entropy Threshold')

    ax_entropy.set_ylabel('Entropy')
    ax_entropy.set_title('Entropy Over Time for All Chapters')
    ax_entropy.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_entropy.grid(True, alpha=0.3)

    ax_changes.set_xlabel('Time (seconds)')
    ax_changes.set_ylabel('Change Timeline')
    ax_changes.set_yticks([])
    ax_changes.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig("entropy_over_time_with_refined_changes.png", dpi=300, bbox_inches='tight')
    if plot:
        plt.show()

    return fig, (ax_entropy, ax_changes)
def improved_detect_entropy_changes(entropies, timestamps, entropy_threshold=4.5, change_threshold=0.5, persistence_threshold=3, stability_window=5):
    change_points = []
    n = len(entropies)

    i = stability_window
    while i < n - stability_window:
        if entropies[i] < entropy_threshold:
            if abs(entropies[i] - entropies[i-1]) > change_threshold:
                if all(entropies[j] < entropy_threshold for j in range(i, min(i+persistence_threshold, n))):
                    if (np.std(entropies[i-stability_window:i]) < change_threshold/2 and
                        np.std(entropies[i:i+stability_window]) < change_threshold/2):
                        end = i + stability_window
                        while end < n and entropies[end] < entropy_threshold:
                            end += 1

                        duration = timestamps[end-1] - timestamps[i]
                        change_points.append((timestamps[i], timestamps[end-1], duration))
                        i = end
                        continue
        i += 1

    return change_points
def extract_longest_segment_frame(frames, timestamps, change_points):
    if not change_points:
        return None, None

    longest_segment = max(change_points, key=lambda x: x[2])
    start_time, end_time, _ = longest_segment

    middle_time = (start_time + end_time) / 2
    frame_index = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - middle_time))

    return frames[frame_index], timestamps[frame_index]
def interactive_entropy_plot(video_data):
    # Create color map for chapters
    num_chapters = len(video_data)
    colors = cm.rainbow(np.linspace(0, 1, num_chapters))

    # Create the main figure
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)

    # Create two subplots
    ax_main = fig.add_subplot(gs[0])
    ax_params = fig.add_subplot(gs[1])

    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.95)

    # Plot each chapter with a different color in the main plot
    for chapter, color in zip(video_data, colors):
        ax_main.plot(chapter['timestamps'], chapter['entropies'], color=color, label=chapter['title'])

    ax_main.set_xlabel('Time (seconds)')
    ax_main.set_ylabel('Entropy')
    ax_main.set_title('Video Entropy Over Time')
    ax_main.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Get overall min and max for axis limits
    all_timestamps = [t for chapter in video_data for t in chapter['timestamps']]
    all_entropies = [entropy for chapter in video_data for entropy in chapter['entropies']]
    x_min, x_max = min(all_timestamps), max(all_timestamps)
    y_min, y_max = min(all_entropies), max(all_entropies)
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)

    # Set up the parameter visualization subplot
    ax_params.set_xlim(x_min, 8)
    ax_params.set_ylim(0, 1)
    ax_params.set_xlabel('Time (seconds)')
    ax_params.set_ylabel('Parameters')
    ax_params.set_title('Parameter Visualization')
    ax_params.set_yticks([])

    # Create sliders
    ax_entropy = plt.axes([0.1, 0.06, 0.8, 0.02])
    ax_change = plt.axes([0.1, 0.04, 0.8, 0.02])
    ax_persistence = plt.axes([0.1, 0.02, 0.8, 0.02])
    ax_stability = plt.axes([0.1, 0.0, 0.8, 0.02])

    s_entropy = Slider(ax_entropy, 'Entropy Threshold', y_min, y_max, valinit=4.5)
    s_change = Slider(ax_change, 'Change Threshold', 0, 2, valinit=0.5)
    s_persistence = Slider(ax_persistence, 'Persistence Threshold', 1, 10, valinit=3, valstep=1)
    s_stability = Slider(ax_stability, 'Stability Window', 1, 20, valinit=5, valstep=1)

    # Create visual representations for parameters
    change_lines, = ax_params.plot([], [], color='red', linewidth=2)
    persistence_rect = ax_params.add_patch(Rectangle((0, 0.4), 3, 0.2, fill=False, edgecolor='purple'))
    stability_shade = ax_params.axvspan(0, 5, ymin=0, ymax=1, alpha=0.2, color='gray')
    threshold_line, = ax_main.plot([x_min, x_max], [4.5, 4.5], color='r', linestyle='--')

    # Store extracted images and their states
    extracted_images = []
    image_states = []

    def update(val):
        nonlocal stability_shade

        entropy_threshold = s_entropy.val
        change_threshold = s_change.val
        persistence_threshold = s_persistence.val
        stability_window = s_stability.val

        # Clear previous rectangles in main plot
        for patch in ax_main.patches:
            if not isinstance(patch, AnnotationBbox):
                patch.remove()

        # Detect changes for each chapter with current parameters
        for chapter, color in zip(video_data, colors):
            changes = improved_detect_entropy_changes(chapter['entropies'], chapter['timestamps'],
                                                      entropy_threshold, change_threshold,
                                                      persistence_threshold, stability_window)

            # Draw rectangles for detected changes in main plot
            for start, end, duration in changes:
                rect = Rectangle((start, y_min), end - start, y_max - y_min,
                                 facecolor=color, alpha=0.3)
                ax_main.add_patch(rect)

        # Update visual representations in parameter plot
        change_lines.set_data([x_min, x_min, x_min+stability_window, x_min+stability_window],
                              [0.4, 0.6, 0.6, 0.4])

        persistence_rect.set_width(persistence_threshold)
        persistence_rect.set_xy((x_min, 0.4))

        stability_shade.remove()
        stability_shade = ax_params.axvspan(x_min, x_min + stability_window, ymin=0, ymax=1, alpha=0.2, color='gray')

        threshold_line.set_ydata([entropy_threshold, entropy_threshold])

        fig.canvas.draw_idle()

    # Connect the update function to the sliders
    s_entropy.on_changed(update)
    s_change.on_changed(update)
    s_persistence.on_changed(update)
    s_stability.on_changed(update)

    # Create buttons
    reset_ax = plt.axes([0.8, 0.09, 0.1, 0.04])
    reset_button = Button(reset_ax, 'Reset')

    ok_ax = plt.axes([0.6, 0.09, 0.1, 0.04])
    ok_button = Button(ok_ax, 'OK')

    finalize_ax = plt.axes([0.4, 0.09, 0.1, 0.04])
    finalize_button = Button(finalize_ax, 'Finalize')

    def reset(event):
        s_entropy.reset()
        s_change.reset()
        s_persistence.reset()
        s_stability.reset()

        # Remove all images
        for ab in ax_main.artists:
            if isinstance(ab, AnnotationBbox):
                ab.remove()
        extracted_images.clear()
        image_states.clear()

        fig.canvas.draw_idle()

    def ok(event):
        nonlocal extracted_images, image_states

        # Remove existing images
        for ab in ax_main.artists:
            if isinstance(ab, AnnotationBbox):
                ab.remove()

        extracted_images.clear()
        image_states.clear()

        for chapter in video_data:
            changes = improved_detect_entropy_changes(chapter['entropies'], chapter['timestamps'],
                                                      s_entropy.val, s_change.val,
                                                      s_persistence.val, s_stability.val)
            for start, end, duration in changes:
                frame, timestamp = extract_longest_segment_frame(chapter['frames'], chapter['timestamps'], [(start, end, duration)])
                if frame is not None:
                    extracted_images.append((frame, timestamp, chapter['title']))
                    image_states.append(True)  # True means the image is active (not blacked out)

        # Display extracted images
        for i, (img, timestamp, _) in enumerate(extracted_images):
            im = OffsetImage(img, zoom=0.1)
            ab = AnnotationBbox(im, (timestamp, y_max), xycoords='data', frameon=False, box_alignment=(0.5, 1), bboxprops=dict(edgecolor='none'))
            ab.set_picker(True)
            ax_main.add_artist(ab)

        fig.canvas.draw_idle()

    def finalize(event):
        plt.close(fig)

    def on_pick(event):
        if isinstance(event.artist, AnnotationBbox):
            ab = event.artist
            idx = [i for i, a in enumerate(ax_main.artists) if a == ab][0]
            image_states[idx] = not image_states[idx]  # Toggle image state

            if image_states[idx]:
                # Restore original image
                ab.get_children()[0].set_data(extracted_images[idx][0])
            else:
                # Black out image
                ab.get_children()[0].set_data(np.zeros_like(extracted_images[idx][0]))

            fig.canvas.draw_idle()

    reset_button.on_clicked(reset)
    ok_button.on_clicked(ok)
    finalize_button.on_clicked(finalize)
    fig.canvas.mpl_connect('pick_event', on_pick)

    # Add explanations for visual representations
    ax_params.text(x_max*1.02, 0.7, "Change Threshold", rotation=0, va='center', ha='left')
    ax_params.text(x_max*1.02, 0.5, "Persistence Threshold", rotation=0, va='center', ha='left')
    ax_params.text(x_max*1.02, 0.3, "Stability Window", rotation=0, va='center', ha='left')

    # Initial plot update
    update(None)

    plt.show()

    # Prepare the output structure
    output_chapters = []
    for chapter in video_data:
        chapter_images = []
        for i, (img, ts, title) in enumerate(extracted_images):
            if title == chapter['title'] and image_states[i]:
                chapter_images.append({"image": img, "timestamp": ts})

        output_chapters.append({
            "title": chapter['title'],
            "images": chapter_images
        })
    return {
        'parameters': {
            'entropy_threshold': s_entropy.val,
            'change_threshold': s_change.val,
            'persistence_threshold': s_persistence.val,
            'stability_window': s_stability.val
        },
        'chapters': output_chapters
    }

with open('/home/user/.apis/video-articles.api', 'r') as file:
    api_key = file.read().strip()
    os.environ['ANTHROPIC_API_KEY'] = api_key
# Initialize the Anthropic client
client = anthropic.Anthropic()

def read_subtitle_section(subtitle_file, start_time, end_time):
    subs = srt_open(subtitle_file)
    section = []
    for sub in subs:
        sub_time = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds
        if start_time <= sub_time <= end_time:
            # if the subtitle text is already in the section, skip it
            if sub.text not in section:
                section.append(sub.text)
    print(f"Read {len(section)} subtitle lines for section from {start_time} to {end_time}")
    print("------------------------")
    print('\n'.join(section))
    return '\n'.join(section)

def encode_image(image):
    # convert to bw
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        return base64.b64encode(image).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        return None


def process_results(results, chapters, subtitle_file, video_title):
    generated_content = []

    for chapter in chapters:
        chapter_title = chapter['title']
        start_time = chapter['start_time']
        end_time = chapter.get('end_time', float('inf'))  # Use infinity if end_time is not specified

        # Find the corresponding chapter in results
        result_chapter = next((c for c in results['chapters'] if c['title'] == chapter_title), None)

        if result_chapter is None:
            logger.warning(f"No matching chapter found in results for '{chapter_title}'")
            continue

        chapter_images = result_chapter.get('images', [])

        subtitle_section = read_subtitle_section(subtitle_file, start_time, end_time)

        content = generate_content(chapter_title, video_title, subtitle_section)
        content['images'] = chapter_images

        generated_content.append({
            'chapter_title': chapter_title,
            'content': content['content'],
            'images': content['images']
        })

    return generated_content

def save_as_asciidoc(generated_content, video_title, output_file):
    # Create a directory for images
    image_dir = os.path.splitext(output_file)[0] + "_images"
    os.makedirs(image_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        # Write document header
        f.write(f"= {video_title}\n")
        f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(":toc:\n")
        f.write(":toclevels: 3\n")
        f.write(":sectnums:\n")
        f.write(f":imagesdir: {image_dir}\n\n")

        # Write each chapter
        for item in generated_content:
            f.write(f"== {item['chapter_title']}\n\n")

            # Add images at the beginning of the chapter
            for i, img in enumerate(item['images']):
                image_filename = f"{item['chapter_title'].replace(' ', '_')}_{i}.png"
                image_path = os.path.join(image_dir, image_filename)
                cv2.imwrite(image_path, img['image'])
                f.write(f"image::{image_filename}[{item['chapter_title']} Image {i+1}]\n\n")

            if item['content'].startswith("Error generating content"):
                # Handle error cases
                f.write("[ERROR]\n")
                f.write("====\n")
                f.write(item['content'])
                f.write("\n====\n\n")
            else:
                f.write(item['content'])

            f.write("\n\n")

    print(f"AsciiDoc content saved to {output_file}")
    print(f"Images saved to {image_dir}")


def save_video_data(video_data, fps, chapters, filename='processed_video_data.pkl'):
    data_to_save = {
        'video_data': video_data,
        'fps': fps,
        'chapters': chapters
    }
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"Video data saved to {filename}")

def sanitize_filename(filename):
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Limit length to 255 characters
    sanitized = sanitized[:255]
    return sanitized

def check_video_exists(url):
    print(f"Checking video for URL: {url}")

    # Get video title
    title_command = ["yt-dlp", "--get-title", url]
    title_result = subprocess.run(title_command, capture_output=True, text=True)
    if title_result.returncode != 0:
        print(f"Error getting video title: {title_result.stderr}")
        return None, None, None, []

    video_title = title_result.stdout.strip()
    sanitized_title = sanitize_filename(video_title)
    video_folder = f"videos/{sanitized_title}"

    # Create video folder if it doesn't exist
    os.makedirs(video_folder, exist_ok=True)

    # Check if video file exists
    video_file = None
    for ext in ['.mp4', '.mkv', '.webm']:
        possible_file = os.path.join(video_folder, f"{sanitized_title}{ext}")
        if os.path.exists(possible_file):
            video_file = possible_file
            break

    chapters_file = None
    subtitle_file = None
    current_resolution = "unknown"
    available_resolutions = []

    if video_file:
        print(f"Video file exists: {video_file}")
        chapters_types = [f"{sanitized_title}.en_chapters.json", f"{sanitized_title}_chapters.json"]
        chapters_file = next((os.path.join(video_folder, c) for c in chapters_types if os.path.exists(os.path.join(video_folder, c))), None)
        subtitle_file = os.path.join(video_folder, f"{sanitized_title}.en.srt") if os.path.exists(os.path.join(video_folder, f"{sanitized_title}.en.srt")) else None

        print(f"Chapters file: {chapters_file}")
        print(f"Subtitle file: {subtitle_file}")

        # Get current resolution using FFmpeg
        ffprobe_command = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=width,height",
            "-of", "json",
            video_file
        ]
        print(f"Running FFprobe command: {' '.join(ffprobe_command)}")
        ffprobe_result = subprocess.run(ffprobe_command, capture_output=True, text=True)
        if ffprobe_result.returncode == 0:
            print(f"FFprobe output: {ffprobe_result.stdout}")
            video_info = json.loads(ffprobe_result.stdout)
            if 'streams' in video_info and video_info['streams']:
                height = video_info['streams'][0].get('height', 'unknown')
                current_resolution = f"{height}"
            print(f"Detected current resolution: {current_resolution}")
        else:
            print(f"Error getting video resolution: {ffprobe_result.stderr}")

    # Get available resolutions
    format_command = ["yt-dlp", "-F", url]
    print(f"Running yt-dlp format command: {' '.join(format_command)}")
    format_result = subprocess.run(format_command, capture_output=True, text=True)
    if format_result.returncode == 0:
        print("yt-dlp format command output:")
        print(format_result.stdout)
        available_resolutions = re.findall(r'(\d+)p', format_result.stdout)
        available_resolutions = sorted(set(available_resolutions), key=int)
        print(f"Extracted available resolutions: {available_resolutions}")
    else:
        print(f"Error getting available resolutions: {format_result.stderr}")

    return video_file, chapters_file, subtitle_file, available_resolutions, video_folder, sanitized_title

def download_video(url, quality='480', video_folder=None, sanitized_title=None):
    if video_folder is None or sanitized_title is None:
        _, _, _, _, video_folder, sanitized_title = check_video_exists(url)

    print(f"Downloading video at {quality}p quality...")
    command = [
        "yt-dlp",
        "-f", f"bestvideo[height<={quality}]+bestaudio/best[height<={quality}]",
        "--write-auto-sub",
        "--sub-lang", "en",
        "--convert-subs", "srt",
        "--embed-chapters",
        "--progress",
        "--print-to-file", "%(chapters)j", os.path.join(video_folder, f"{sanitized_title}_chapters.json"),
        "--output", os.path.join(video_folder, f"{sanitized_title}.%(ext)s"),
        url
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error downloading video: {result.stderr}")
        return None, None, None

    # Find the video file
    video_files = glob.glob(os.path.join(video_folder, f"{sanitized_title}.*"))
    video_file = next((f for f in video_files if f.endswith(('.mp4', '.mkv', '.webm'))), None)
    if not video_file:
        print("Couldn't find downloaded video file")
        return None, None, None

    print("Video downloaded successfully.")

    # Find the chapters file
    chapters_file = os.path.join(video_folder, f"{sanitized_title}_chapters.json")
    if not os.path.exists(chapters_file):
        chapters_file = None

    # Find the subtitle file
    subtitle_file = os.path.join(video_folder, f"{sanitized_title}.en.srt")
    if not os.path.exists(subtitle_file):
        subtitle_file = None

    # Verify that files exist and print results
    print(f"Video file: {video_file if video_file and os.path.exists(video_file) else 'Not found'}")
    print(f"Chapters file: {chapters_file if chapters_file and os.path.exists(chapters_file) else 'Not found'}")
    print(f"Subtitle file: {subtitle_file if subtitle_file and os.path.exists(subtitle_file) else 'Not found'}")

    return video_file, chapters_file, subtitle_file
def video_download_workflow(url):
    video_file, chapters_file, subtitle_file, available_resolutions, video_folder, sanitized_title = check_video_exists(url)

    if video_file and os.path.exists(video_file):
        print(f"\nVideo file already exists: {video_file}")
        while True:
            redownload = input("Do you want to redownload at a different resolution? (yes/no): ").lower()
            if redownload in ['yes', 'no']:
                break
            print("Please enter 'yes' or 'no'.")

        if redownload == 'yes':
            resolution = get_user_resolution_choice(available_resolutions)
            if resolution:
                print(f"Redownloading video at {resolution}p resolution...")
                video_file, chapters_file, subtitle_file = download_video(url, resolution, video_folder, sanitized_title)
            else:
                print("Download cancelled. Using existing video file.")
        else:
            print("Using existing video file.")
    else:
        print("\nVideo file not found. Preparing to download...")
        resolution = get_user_resolution_choice(available_resolutions)
        if resolution:
            video_file, chapters_file, subtitle_file = download_video(url, resolution, video_folder, sanitized_title)
        else:
            print("Download cancelled.")
            return None, None, None

    if video_file:
        print("\nVideo processing complete.")
        return video_file, chapters_file, subtitle_file, video_folder
    else:
        print("\nFailed to download or locate the video.")
        return None, None, None, None
def read_prompt_file(filename: str) -> str:
    with open(filename, 'r') as file:
        return file.read().strip()

def generate_content(chapter_title: str, video_title: str, subtitle_section: str, max_tokens:int =2334) -> Dict[str, Any]:
    system_prompt = read_prompt_file('system_prompt.md')
    user_prompt_template = read_prompt_file('user_prompt.md')

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
        print("Response received from API:")
        print(response.content[0].text)
        print("------------------------")
        return {"content": response.content[0].text, "images": []}

    except anthropic.BadRequestError as e:
        logger.error(f"Error generating content for chapter '{chapter_title}': {e}")
        return {"content": f"Error generating content: {str(e)}", "images": []}
