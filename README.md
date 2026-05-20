# Setu-AI

Setu AI (সেতু)  is a next-generation, high-performance AI interpreter designed for seamless, live communication between Bangla and English speakers or others language.

I built a real-time speech translation system using WebSocket streaming, Whisper for STT, and TTS synthesis. The system processes audio in chunks to achieve low latency (~2 seconds), similar to Google Translate voice mode.

    conda create --name setu python=3.10 -y
    conda activate setu
    pip install -r requirements.txt
