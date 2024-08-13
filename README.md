# Video Analysis and Content Generation Tool

This project is a Python-based tool for analyzing video content, extracting key frames, and generating summary content. It uses video entropy analysis to identify significant moments in the video and provides a graphical user interface for frame selection and content generation.

## Features

- Download YouTube videos with optional quality selection

- Analyze video entropy to identify significant moments

- Interactive GUI for visualizing video entropy and selecting key frames

- Automatic extraction of video chapters and subtitles

- Content generation based on selected frames and video context

## Prerequisites

- Python 3.7+

- FFmpeg (for video processing)

## Installation

1. Clone this repository:

   ```

   git clone https://github.com/yourusername/video-analysis-tool.git

   cd video-analysis-tool

   ```

2. Set up a Conda environment (recommended):

   ```

   conda create -n video-analysis python=3.8

   conda activate video-analysis

   ```

3. Install the required packages:

   ```

   pip install -r requirements.txt

   ```

   Note: If you're not using Conda, you can still use pip to install the requirements in your preferred Python environment.

4. Install FFmpeg:

   - On macOS (using Homebrew): brew install ffmpeg

   - On Ubuntu/Debian: sudo apt-get install ffmpeg

   - On Windows: Download from [FFmpeg official site](https://ffmpeg.org/download.html) and add to PATH

## Usage

1. Run the main script:

   ```

   python video_analysis.py

   ```

2. When prompted, enter the YouTube URL of the video you want to analyze.

3. Choose the desired video quality for download.

4. The tool will download the video, extract chapters and subtitles, and analyze the video entropy. Moments where the entropy signal is stable is a strong indicator that a significant event (like a diagram) is being shown. You can pull up the video in a video player to verify this.

5. Use the interactive GUI to select key frames based on the entropy plot.

6. The tool will generate content based on the selected frames and video context.

7. Output files (including generated content and selected frames) will be saved in the videos directory.

## Configuration

- API keys: Place your Anthropic API key in a file named video-articles.api in the /home/user/.apis/ directory.

## Troubleshooting

- If you encounter issues with video download, make sure you have the latest version of yt-dlp installed:

  ```

  pip install --upgrade yt-dlp

  ```

- For GUI-related issues, ensure you have the required Qt libraries installed:

  ```

  conda install pyqt

  ```

  or

  ```

  pip install PyQt5

  ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments