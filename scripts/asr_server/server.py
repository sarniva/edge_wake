#!/usr/bin/env python3
"""ASR server for edge_wake Phase 7 (runs on the RTX 2050 laptop).

Receives 16 kHz int16 mono PCM over websocket from the ESP32-S3,
transcribes each utterance with faster-whisper (CUDA), prints the text
and appends it to results.log (+ saves the utterance wav).

Protocol (must match main/streamer.c):
  <- text   {"type":"start","sr":16000,"ch":1}
  <- binary int16le PCM chunks
  <- text   {"type":"end"}
  -> text   {"type":"result","text":"...","language":"..","duration_s":..}

Setup (Arch, RTX 2050, driver already installed):
  sudo pacman -Syu cuda cudnn python python-pip
  python -m venv ~/asr && source ~/asr/bin/activate
  pip install -U pip && pip install -r requirements.txt

Run:
  source ~/asr/bin/activate
  python server.py --host 0.0.0.0 --port 8765 --model large-v3-turbo
Find the laptop IP (put it in the ESP's main/wifi_config.h as ASR_WS_URI):
  ip -4 addr show | grep inet
"""
import argparse, asyncio, datetime, json, os, struct, wave

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:
    raise SystemExit("pip install faster-whisper (see header)")

import websockets


async def handle(ws, model, outdir, logf):
    pcm = bytearray()
    sr = 16000
    peer = ws.remote_address
    print(f"[conn] {peer}", flush=True)
    try:
        async for msg in ws:
            if isinstance(msg, str):
                try:
                    req = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                if req.get("type") == "start":
                    sr = int(req.get("sr", 16000))
                    pcm = bytearray()
                    print(f"[start] sr={sr} from {peer}", flush=True)
                elif req.get("type") == "end":
                    dur = len(pcm) / 2 / sr
                    print(f"[end] {dur:.1f}s audio, transcribing...", flush=True)
                    loop = asyncio.get_running_loop()
                    try:
                        segs, info = await loop.run_in_executor(
                            None, transcribe, model, bytes(pcm), sr)
                    except Exception as e:
                        # Never kill the connection on a bad utterance (e.g.
                        # missing CUDA libs): report and keep serving. The
                        # classic cause: libcublas.so not on the loader path.
                        # Fix: export LD_LIBRARY_PATH=/opt/cuda/lib64 (Arch).
                        err = f"{type(e).__name__}: {e}"
                        print(f"[transcribe FAILED] {err}", flush=True)
                        await ws.send(json.dumps({"type": "error",
                                                  "message": err}))
                        pcm = bytearray()
                        continue
                    text = " ".join(s.text.strip() for s in segs).strip()
                    print(f"[text:{info.language} p={info.language_probability:.2f}] {text}",
                          flush=True)
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    with wave.open(os.path.join(outdir, f"utt_{ts}.wav"), "wb") as w:
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                        w.writeframes(bytes(pcm))
                    logf.write(f"{ts} [{info.language}] {text}\n"); logf.flush()
                    await ws.send(json.dumps({"type": "result", "text": text,
                                              "language": info.language,
                                              "duration_s": round(dur, 1)}))
                    pcm = bytearray()
            elif isinstance(msg, (bytes, bytearray)):
                pcm += msg
    except websockets.ConnectionClosed:
        print(f"[disc] {peer}", flush=True)


def transcribe(model, raw, sr):
    audio = np.array(struct.unpack(f"<{len(raw)//2}h", raw),
                     dtype=np.float32) / 32768.0
    # vad_filter trims fan-hum/silence so whisper doesn't invent words.
    return model.transcribe(audio, beam_size=5, vad_filter=True,
                            vad_parameters={"min_silence_duration_ms": 500})


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--outdir", default="utterances")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    print(f"loading {a.model} (first run downloads ~800 MB)...", flush=True)
    model = WhisperModel(a.model, device="cuda", compute_type="int8")
    logf = open(os.path.join(a.outdir, "results.log"), "a")
    print(f"serving on {a.host}:{a.port} (int8, cuda, vad_filter)", flush=True)
    async with websockets.serve(
            lambda ws: handle(ws, model, a.outdir, logf), a.host, a.port,
            max_size=8 * 1024 * 1024):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
