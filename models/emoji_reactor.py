import cv2
import numpy as np
import base64
from collections import deque
import os

class EmojiReactor: 
    def __init__(self):
        # Load face cascade for detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Emoji mappings to file paths
        self.emoji_map = {
            'neutral': 'static/assets/emojis/thinking.jpg',
            'happy': 'static/assets/emojis/laugh.jpg',
            'very_happy': 'static/assets/emojis/extremelaugh.jpg',
            'sad': 'static/assets/emojis/fedup.jpg',
            'angry': 'static/assets/emojis/angry.jpg',
            'surprised': 'static/assets/emojis/shock.jpg',
            'sleepy': 'static/assets/emojis/sleep.jpg',
            'cool': 'static/assets/emojis/smirk.jpg',
            'smart': 'static/assets/emojis/nerd.jpg',
            'thumbsup': 'static/assets/emojis/thumbsup.jpg'
        }
        
        # Pre-load and cache emojis
        self.emoji_cache = {}
        self._load_emojis()
        
        # Smoothing buffer for stable predictions (reduce jitter)
        self.emotion_buffer = deque(maxlen=5)
        
        # Frame skip counter for optimization
        self.frame_count = 0
        self.skip_frames = 2  # Process every 3rd frame
        self.last_emotion = 'neutral'
        self.last_face_region = None
        
    def _load_emojis(self):
        for emotion, path in self.emoji_map.items():
            if os.path.exists(path):
                emoji_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if emoji_img is not None:
                    # Resize to standard size for consistent overlay
                    emoji_img = cv2.resize(emoji_img, (120, 120))
                    self.emoji_cache[emotion] = emoji_img
    
    def detect_emotion(self, face_roi):
        try:
            h, w = face_roi.shape[:2]
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            
            # Divide face into regions for analysis
            upper_half = gray[:h//2, :]
            lower_half = gray[h//2:, :]
            
            # Eye region (upper third)
            eye_region = gray[:h//3, :]
            # Mouth region (lower third)
            mouth_region = gray[2*h//3:, :]
            
            # Calculate intensity metrics
            eye_brightness = np.mean(eye_region)
            mouth_brightness = np.mean(mouth_region)
            upper_brightness = np.mean(upper_half)
            lower_brightness = np.mean(lower_half)
            
            # Edge detection for expression intensity
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / (h * w)
            
            # Variance in mouth region (smile detection)
            mouth_variance = np.var(mouth_region)
            
            # Emotion detection logic
            emotion = 'neutral'
            
            # Happy detection (bright mouth area, high variance)
            if mouth_brightness > upper_brightness * 1.15 and mouth_variance > 500:
                emotion = 'very_happy' if mouth_variance > 800 else 'happy'
            
            # Angry detection (low brightness, high edge density)
            elif edge_density > 0.15 and upper_brightness < 100:
                emotion = 'angry'
            
            # Surprised detection (wide eyes, high overall brightness)
            elif eye_brightness > 130 and edge_density > 0.12:
                emotion = 'surprised'
            
            # Sleepy detection (very low eye brightness)
            elif eye_brightness < 60:
                emotion = 'sleepy'
            
            # Sad detection (low mouth brightness)
            elif mouth_brightness < upper_brightness * 0.85:
                emotion = 'sad'
            
            # Smart/thinking (moderate values, balanced)
            elif abs(upper_brightness - lower_brightness) < 10 and edge_density < 0.1:
                emotion = 'smart'
            
            # Cool/smirk (slight asymmetry)
            elif 0.9 < (mouth_brightness / upper_brightness) < 1.1:
                emotion = 'cool'
            
            return emotion
            
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return 'neutral'
    
    def get_smoothed_emotion(self, current_emotion):
        self.emotion_buffer.append(current_emotion)
        
        # Return most common emotion in buffer
        if len(self.emotion_buffer) >= 3:
            emotion_counts = {}
            for e in self.emotion_buffer:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            return max(emotion_counts, key=emotion_counts.get)
        return current_emotion
    
    def overlay_emoji(self, frame, emoji_emotion, face_rect):
        x, y, w, h = face_rect
        
        if emoji_emotion not in self.emoji_cache:
            return frame
        
        emoji = self.emoji_cache[emoji_emotion].copy()
        
        # Calculate emoji position (above the face)
        emoji_size = min(w, 120)
        emoji = cv2.resize(emoji, (emoji_size, emoji_size))
        
        emoji_x = x + (w - emoji_size) // 2
        emoji_y = max(0, y - emoji_size - 10)
        
        # Ensure emoji fits in frame
        if emoji_y < 0:
            emoji_y = y + h + 10
        
        if emoji_x < 0:
            emoji_x = 0
        if emoji_x + emoji_size > frame.shape[1]:
            emoji_x = frame.shape[1] - emoji_size
        
        # Handle transparency if emoji has alpha channel
        if emoji.shape[2] == 4:
            # Extract alpha channel
            alpha = emoji[:, :, 3] / 255.0
            
            # Ensure we don't go out of bounds
            y_end = min(emoji_y + emoji_size, frame.shape[0])
            x_end = min(emoji_x + emoji_size, frame.shape[1])
            
            emoji_h = y_end - emoji_y
            emoji_w = x_end - emoji_x
            
            if emoji_h > 0 and emoji_w > 0:
                emoji_resized = emoji[:emoji_h, :emoji_w]
                alpha_resized = alpha[:emoji_h, :emoji_w]
                
                # Blend emoji with background
                for c in range(3):
                    frame[emoji_y:y_end, emoji_x:x_end, c] = \
                        frame[emoji_y:y_end, emoji_x:x_end, c] * (1 - alpha_resized) + \
                        emoji_resized[:, :, c] * alpha_resized
        else:
            # No alpha channel, direct overlay
            y_end = min(emoji_y + emoji_size, frame.shape[0])
            x_end = min(emoji_x + emoji_size, frame.shape[1])
            
            emoji_h = y_end - emoji_y
            emoji_w = x_end - emoji_x
            
            if emoji_h > 0 and emoji_w > 0:
                frame[emoji_y:y_end, emoji_x:x_end] = emoji[:emoji_h, :emoji_w, :3]
        
        return frame
    
    def process_frame(self, frame_data):
        try:
            # Decode base64 frame
            img_data = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {'success': False, 'error': 'Failed to decode frame'}
            
            # Frame skipping for performance
            self.frame_count += 1
            should_detect = (self.frame_count % (self.skip_frames + 1) == 0)
            
            # Detect faces (only every N frames)
            if should_detect:
                # Resize for faster detection
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # Scale back to original size
                faces = [(x*2, y*2, w*2, h*2) for (x, y, w, h) in faces]
                
                if len(faces) > 0:
                    # Use the largest face
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    self.last_face_region = largest_face
                    
                    x, y, w, h = largest_face
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Detect emotion
                    emotion = self.detect_emotion(face_roi)
                    self.last_emotion = self.get_smoothed_emotion(emotion)
                else:
                    self.last_face_region = None
            
            # Draw rectangle and overlay emoji
            if self.last_face_region is not None:
                x, y, w, h = self.last_face_region
                
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Overlay emoji
                frame = self.overlay_emoji(frame, self.last_emotion, self.last_face_region)
                
                # Add emotion label
                label = self.last_emotion.replace('_', ' ').title()
                cv2.putText(frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Encode result
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'success': True,
                'frame': f'data:image/jpeg;base64,{frame_base64}',
                'emotion': self.last_emotion,
                'faces_detected': 1 if self.last_face_region is not None else 0
            }
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}


# Global reactor instance (reusing across requests for performance)
_reactor_instance = None

def get_reactor():
    """Get or create the global reactor instance"""
    global _reactor_instance
    if _reactor_instance is None:
        _reactor_instance = EmojiReactor()
    return _reactor_instance

def process_emoji_frame(frame_data):
    """Process a frame for emoji reactions"""
    reactor = get_reactor()
    return reactor.process_frame(frame_data)