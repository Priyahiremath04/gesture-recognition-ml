from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pygame
import time
from gtts import gTTS
from mutagen.mp3 import MP3
from PIL import ImageFont, ImageDraw, Image
import threading

# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Initialize MediaPipe hands
# -------------------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

# -------------------------------
# Font mapping per language
# -------------------------------
FONT_FILES = {
    'kn': "kannada.ttf",
    'en': "qenglish.ttf",  
    'hi': "hindi.ttf",
    'te': "telugu.ttf",
    'ta': "tamil.ttf"
}

# Gesture statements in different languages
gesture_dict = {
    'one': {'kn': 'ಒಂದು','en':'One','hi':'एक','te':'ఒకటి','ta':'ஒன்று'},
    'two': {'kn': 'ಎರಡು','en':'Two','hi':'दो','te':'రెండు','ta':'இரண்டு'},
    'three': {'kn': 'ಮೂರು','en':'Three','hi':'तीन','te':'మూడు','ta':'மூன்று'},
    'four': {'kn': 'ನಾಲ್ಕು','en':'Four','hi':'चार','te':'నాలుగు','ta':'நான்கு'},
    'super': {'kn': 'ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ','en':'I am fine','hi':'मैं ठीक हूँ','te':'నేను బాగున్నాను','ta':'நான் நன்றாக இருக்கிறேன்'},
    'up': {'kn': 'ಮೇಲೆ ಹೋಗಿ','en':'Go up','hi':'ऊपर जाओ','te':'పైనికి వెళ్ళు','ta':'மேலே போ'},
    'down': {'kn': 'ಕೆಳಗೆ ಹೋಗಿ','en':'Go down','hi':'नीचे जाओ','te':'కిందకి వెళ్ళు','ta':'கீழே போ'},
    'call': {'kn': 'ನನ್ನನ್ನು ಕರೆ','en':'Call me','hi':'मुझे कॉल करो','te':'నన్ను కాల్ చెయ్యి','ta':'என்னை கூப்பிடு'},
    'smile': {'kn': 'ದಯವಿಟ್ಟು ನಗಿರಿ','en':'Smile please','hi':'कृपया मुस्कुराइए','te':'దయచేసి నవ్వండి','ta':'தயவு செய்து சிரி'},
    'zero': {'kn': 'ನಿಲ್ಲಿಸಿ','en':'Stop','hi':'रुको','te':'ఆపు','ta':'நிறுத்து'}
}

# -------------------------------
# Global selected language
# -------------------------------
selected_lang = "en"

# -------------------------------
# Draw multilingual text
# -------------------------------
def draw_text(img, text, position, lang='en', font_size=40, color=(0,0,255)):
    font_path = FONT_FILES.get(lang, "english.ttf")
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# -------------------------------
# Video Processing Function
# -------------------------------
def process_video():
    global selected_lang
    cap = cv2.VideoCapture(0)
    
    # Create a named window
    cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        x, y, c = frame.shape
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)

        className = ''
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])
                    
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                prediction = model.predict([landmarks])
                classID = np.argmax(prediction)
                className = classNames[classID]

                statement = ''
                if className in gesture_dict:
                    statement = gesture_dict[className][selected_lang]

                if len(statement.strip()) > 0:
                    try:
                        myobj = gTTS(text=statement, lang=selected_lang, slow=False)
                        myobj.save("voice.mp3")
                        song = MP3("voice.mp3")
                        pygame.mixer.init()
                        pygame.mixer.music.load('voice.mp3')
                        pygame.mixer.music.play()
                        time.sleep(min(song.info.length, 1.5))
                        pygame.quit()
                    except Exception as e:
                        print("TTS Error:", e)

                frame = draw_text(frame, statement, (50, 50), lang=selected_lang, font_size=50)

        # Display language info
        lang_name = {
            'kn': 'Kannada',
            'en': 'English',
            'hi': 'Hindi',
            'te': 'Telugu',
            'ta': 'Tamil'
        }.get(selected_lang, 'Unknown')
        
        cv2.putText(frame, f"Language: {lang_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to return to main page", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Gesture Recognition", frame)

        # Check for 'q' key press to return to main page
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        global selected_lang
        selected_lang = request.form.get("language", "en")
        
        # Start video processing in a separate thread
        video_thread = threading.Thread(target=process_video)
        video_thread.daemon = True
        video_thread.start()
        
        return render_template("processing.html", language=selected_lang)
    return render_template("index.html")

@app.route("/stop")
def stop():
    # This route is not needed for OpenCV display, but kept for consistency
    return redirect(url_for("index"))

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
