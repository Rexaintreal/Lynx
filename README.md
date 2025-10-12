<div align="center">
  <img src="static/assets/logo.png" alt="Lynx Logo" width="200"/>
  
  # Lynx
  
  **A web app packed with computer vision tools - built with OpenCV and Flask**
  
  [![GitHub](https://img.shields.io/badge/GitHub-Lynx-blue?logo=github)](https://github.com/Rexaintreal/lynx)
  [![Hackberry](https://img.shields.io/badge/Built%20for-Hackberry%20YSWS-orange)](https://hackberry.hackclub.com/)
  [![Hackatime](https://hackatime-badge.hackclub.com/U09B8FXUS78/Lynx)](https://hackatime-badge.hackclub.com/U09B8FXUS78/Lynx)

  <!-- Library badges -->
  [![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
  [![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
  [![EasyOCR](https://img.shields.io/badge/EasyOCR-FFD43B?logo=pytorch&logoColor=black)](https://github.com/JaidedAI/EasyOCR)
  
</div>

---

## About

Lynx is this collection of image processing tools I built for [Hackberry YSWS](https://hackberry.hackclub.com/). It's a website where you can play around with different computer vision stuff - from detecting faces to scanning documents and a bunch of other cool things. Building this was honestly a whole journey. Learned a lot about OpenCV and Flask, especially how tricky real-time processing can get.

---

## Demo Video

The showcase video where i try all the features while running Lynx locally

🎥 **[Watch the full demo here](https://drive.google.com/file/d/1pQp9p8zt818QkFMgoA2k8czjXPXUUMQw/view?usp=drive_link)**

---

## Features

Here's everything I've added into Lynx. It's organized the same way as the navbar on the website:

### Face Intelligence
- **Face Detection**: Just counts how many faces are in your picture using Haar Cascades. Pretty basic but it works
- **Face Recognition**: This one's cooler - it tries to guess people's age and gender using some pre-trained models
- **Webcam Live**: Puts filters on your webcam in real-time (had to move this to client-side because the OpenCV version was super laggy)
- **Emoji Reactor**: Looks at your expression and shows a matching emoji. It's pretty fun to mess around with

### Image Filters & Effects
- **Image Filters**: A bunch of preset filters you can slap onto your images. All the heavy lifting happens on the server
- **Special Effects**: The fancy stuff - green screen, color pop, duotone, and isolating specific colors
- **Pixelation**: Make your images look pixelated or add that retro game vibe with stripes
- **Background Remover**: I've added like 5 different ways to remove backgrounds using OpenCV - from GrabCut to simple color stuff
- **ASCII Art Generator**: Turns your images into text art. There are different styles you can try

### Vision Tools
- **Object Detection**: Tries to figure out what objects are in your image using MobileNet SSD
- **Scene Understanding**: Guesses where the photo was taken - like is it a beach, forest, or city
- **QR Code Detection**: Scans and reads QR codes from images
- **Number Plate Detection**: Finds car number plates and reads them using EasyOCR
- **Location Analyzer (Geo Guesser)**: My attempt at making a GeoGuessr type thing. It looks for clues in images to guess the location

### Color Tools
- **Color Analysis**: Shows you the top 10 colors in an image using K-Means clustering
- **Color Manipulation**: Change specific colors, play with saturation, shift hues and stuff

### Document Tools
- **Document Scanner**: Upload a pic of a document and this will straighten it out to look like a proper scan
- **Text Detection**: Just draws boxes around any text it finds
- **Text Extractor (OCR)**: Actually reads the text from images using EasyOCR. Works with a few different languages
- **Captcha Solver**: Solves those simple text captchas. Nothing too fancy

---

## Project Structure

```
Lynx/
├── devlogs/              # Development logs, images, and videos
├── models/               # Python scripts for all computer vision tasks
├── static/               # Static assets (CSS, JavaScript, images)
├── templates/            # HTML templates for the web interface
├── uploads/              # Default directory for user-uploaded images
├── .gitignore            # Git ignore file
├── app.py                # Main Flask application script
├── LICENSE               # MIT License
├── README.md             # You are here!
└── requirements.txt      # Python dependencies
```

---

## Setup and Installation

### Prerequisites
- **Python 3.12+** (I used 3.12.10, so anything 3.12+ should be fine)
- **pip** for installing stuff

### Installation Steps

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Rexaintreal/lynx.git
   cd lynx
   ```

2. **Create a virtual environment (recommended):**
   
   I didn't use one while coding this (my bad), but you probably should so you don't mess up your other Python projects.
   
   **Windows:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
   
   **macOS / Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Main ones are `flask`, `opencv-python`, `numpy`, `pillow`, `scikit-learn`, and `easyocr`.

4. **First-time OCR setup:**
   
   The first time you use any OCR tools (Text Extractor or Captcha Solver), EasyOCR will start downloading language models. Takes about 5-10 minutes and it might look frozen but it's working. Just keep an eye on the terminal - it'll show the progress. (or any errors)
   
   **GPU Acceleration**: If you have a decent GPU, change `gpu=False` to `gpu=True` for better performance. I tested it with my laptop's RTX 4050 (6GB VRAM) and it took around 7 minutes on the first run. After that everything works smoothly.

5. **Run it:**
   ```bash
   python app.py
   ```

6. **Open it up:**
   
   Go to `http://127.0.0.1:5000` in your browser

---

## Usage

Pretty straightforward:
1. Check out the homepage to see all the tools
2. Pick whatever you want to try from the navbar or the grid
3. Upload an image (there's an `examples` folder with test images)
4. Play around with the sliders if there are any
5. See the result and download it if you want

---

## License

MIT [LICENSE](LICENSE).

---

## Acknowledgements

**THE INTERNET** - followed many tutorials, articles and webpages while building this.

---

## You may also like...

Some other projects I've built:

- [Libro Voice](https://github.com/Rexaintreal/Libro-Voice) - A PDF to Audio Converter
- [Snippet Vision](https://github.com/Rexaintreal/Snippet-Vision) - A YouTube Video Summarizer
- [Weather App](https://github.com/Rexaintreal/WeatherApp) - A Python Weather Forecast App
- [Python Screenrecorder](https://github.com/Rexaintreal/PythonScreenrecorder) - A Python Screen Recorder
- [Typing Speed Tester](https://github.com/Rexaintreal/TypingSpeedTester) - A Python Typing Speed Tester
- [Movie Recommender](https://github.com/Rexaintreal/Movie-Recommender) - A Python Movie Recommender
- [Password Generator](https://github.com/Rexaintreal/Password-Generator) - A Python Password Generator
- [Object Tales](https://github.com/Rexaintreal/Object-Tales) - A Python Image to Story Generator
- [Finance Manager](https://github.com/Rexaintreal/Finance-Manager) - A Flask WebApp to Monitor Savings
- [Codegram](https://github.com/Rexaintreal/Codegram) - A Social Media Web App for Coders
- [Simple Flask Notes](https://github.com/Rexaintreal/Simple-Flask-Notes) - A Flask Notes App
- [Key5](https://github.com/Rexaintreal/key5) - Python Keylogger
- [Codegram2024](https://github.com/Rexaintreal/Codegram2024) - A Modern Version of Codegram (Update)
- [Cupid](https://github.com/Rexaintreal/cupid) - A Dating Web App for Teenagers
- [Gym Vogue](https://github.com/Rexaintreal/GymVogue/) - Ecommerce Site for Gym Freaks
- [Confessions](https://github.com/Rexaintreal/Confessions) - Anonymous confession platform
- [Syna](https://github.com/Rexaintreal/syna) - A social music web application where users can log in using their Spotify accounts and find their best matches based on shared music preferences
- [Apollo](https://github.com/Rexaintreal/Apollo) - A Minimal Music Player with a Cat Dancing/Bopping to the beats
- [Eros](https://github.com/Rexaintreal/Eros) - A face symmetry analyzer built using Python and OpenCV
- [Notez](https://github.com/Rexaintreal/Notez) - A clean and minimal Android notes application built with Flutter

---

## Author

Built by **Saurabh Tiwari**

- Email: [saurabhtiwari7986@gmail.com](mailto:saurabhtiwari7986@gmail.com)  
- Twitter: [@Saurabhcodes01](https://x.com/Saurabhcodes01)
- Instagram: [@saurabhcodesawfully](https://instagram.com/saurabhcodesawfully)

