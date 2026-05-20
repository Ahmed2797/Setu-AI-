import os
import json
import base64
import asyncio
import websockets
from dotenv import load_dotenv
import pyaudio
import numpy as np
from scipy.signal import resample
import tkinter as tk
from threading import Thread

# ----------------------------
# Configuration
# ----------------------------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "OpenAI-Beta": "realtime=v1"
}

# Audio Constants
INPUT_RATE = 44100
TARGET_RATE = 16000     # OpenAI expects 16kHz for input
AI_OUTPUT_RATE = 24000  # OpenAI outputs 24kHz by default
CHANNELS = 1
CHUNK = 1024            # Larger chunk for stability

# ----------------------------
# Audio Processor
# ----------------------------
class AudioProcessor:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        
        # Mic Stream (Input)
        self.input_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=INPUT_RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        # Speaker Stream (Output - Set to 24kHz for AI voice)
        self.output_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=AI_OUTPUT_RATE,
            output=True
        )

    def read(self):
        try:
            data = self.input_stream.read(CHUNK, exception_on_overflow=False)
            return self._process(data)
        except Exception:
            return b''

    def play(self, chunk):
        self.output_stream.write(chunk)

    def _process(self, data):
        audio = np.frombuffer(data, dtype=np.int16).copy()
        
        # Simple Noise Gate: Silence very quiet background noise
        audio[np.abs(audio) < 500] = 0
        
        # Resample from 44.1kHz to 16kHz
        num_samples = int(len(audio) * TARGET_RATE / INPUT_RATE)
        resampled = resample(audio, num_samples)
        return resampled.astype(np.int16).tobytes()

    def close(self):
        self.input_stream.stop_stream()
        self.output_stream.stop_stream()
        self.p.terminate()

# ----------------------------
# Realtime WebSocket Client
# ----------------------------
class RealtimeClient:
    def __init__(self, audio, on_text):
        self.audio = audio
        self.on_text = on_text
        self.running = False
        self.loop = None

    async def connect(self):
        try:
            async with websockets.connect(URL, additional_headers=HEADERS) as ws:
                self.on_text("System: Connected to OpenAI")
                await self._init_session(ws)
                
                # Run sender and receiver concurrently
                await asyncio.gather(
                    self._send_audio(ws),
                    self._receive_events(ws)
                )
        except Exception as e:
            self.on_text(f"Error: {str(e)}")

    async def _init_session(self, ws):
        session_config = {
            "type": "session.update",
            "session": {
                "instructions": "You are a real-time translator. Auto-detect if the user is speaking Bangla or English and translate to the other language instantly.",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {"type": "server_vad"}
            }
        }
        await ws.send(json.dumps(session_config))

    async def _send_audio(self, ws):
        while self.running:
            audio_data = self.audio.read()
            if audio_data:
                payload = {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(audio_data).decode("utf-8")
                }
                await ws.send(json.dumps(payload))
            await asyncio.sleep(0.01) # Small delay to prevent loop saturation

    async def _receive_events(self, ws):
        async for message in ws:
            if not self.running:
                break
            
            event = json.loads(message)
            event_type = event.get("type")

            # 1. Play AI Voice
            if event_type == "response.audio.delta":
                audio_chunk = base64.b64decode(event["delta"])
                self.audio.play(audio_chunk)

            # 2. Show what YOU said (transcription)
            elif event_type == "conversation.item.input_audio_transcription.completed":
                self.on_text(f"You: {event['transcript']}")

            # 3. Show what AI says (translation)
            elif event_type == "response.audio_transcript.done":
                self.on_text(f"AI: {event['transcript']}")

# ----------------------------
# GUI Application
# ----------------------------
class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎤 Real-time Translator (Bangla ↔ English)")
        self.audio = AudioProcessor()
        self.client = RealtimeClient(self.audio, self.display_text)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        self.status_label = tk.Label(self.root, text="Ready", font=("Arial", 12, "bold"), fg="gray")
        self.status_label.pack(pady=10)

        self.text_box = tk.Text(self.root, height=15, width=60, font=("Segoe UI", 10))
        self.text_box.pack(pady=10, padx=10)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.start_btn = tk.Button(btn_frame, text="▶ Start", bg="green", fg="white", width=10, command=self.start)
        self.start_btn.pack(side="left", padx=10)

        self.stop_btn = tk.Button(btn_frame, text="⛔ Stop", bg="red", fg="white", width=10, command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=10)

    def start(self):
        self.client.running = True
        self.status_label.config(text="🎧 Listening & Translating...", fg="green")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        
        # Start the asyncio loop in a background thread
        Thread(target=self._run_async_loop, daemon=True).start()

    def stop(self):
        self.client.running = False
        self.status_label.config(text="Stopped", fg="red")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _run_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.client.connect())

    def display_text(self, text):
        # Thread-safe GUI update
        self.root.after(0, self._update_ui_text, text)

    def _update_ui_text(self, text):
        self.text_box.insert(tk.END, text + "\n")
        self.text_box.see(tk.END)

    def on_close(self):
        self.stop()
        self.audio.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()