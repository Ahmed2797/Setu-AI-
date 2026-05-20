import os, io, asyncio, json, time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from openai import OpenAI
from dotenv import load_dotenv
import base64

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

@app.get("/")
async def serve_ui():
    return FileResponse("index.html")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            if len(data) < 2000: continue
            
            start_time = time.time()
            

            try:
                audio_buffer = io.BytesIO(data)
                audio_buffer.name = "input.webm"
                # 1. Transcription
                transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_buffer)
                user_text = transcript.text
                if not user_text.strip(): continue

                # 2. Translation
                det_lang = "bn" if any("\u0980" <= c <= "\u09FF" for c in user_text) else "en"
                target_lang = "Bengali" if det_lang == "en" else "English"
                
                trans_res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": f"Translate to {target_lang}."}, 
                              {"role": "user", "content": user_text}]
                )
                translated_text = trans_res.choices[0].message.content

                # 3. TTS
                tts_res = client.audio.speech.create(model="tts-1", voice="alloy", input=translated_text)
                
                # Calculate processing time
                proc_time = round(time.time() - start_time, 2)

                # 4. Send JSON with text and base64 audio
                audio_base64 = base64.b64encode(tts_res.content).decode('utf-8')
                await ws.send_json({
                    "user_text": user_text,
                    "ai_text": translated_text,
                    "proc_time": proc_time,
                    "audio": audio_base64
                })

            except Exception as e:
                print(f"Error: {e}")
    except WebSocketDisconnect:
        print("Disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

## uvicorn main:app --reload