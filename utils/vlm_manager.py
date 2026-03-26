import subprocess
import time
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_HOST = "http://localhost:8002"
VLLM_HOST = os.getenv("VLLM_HOST", "http://localhost:8000")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")

_vlm_process = None

def stop_vllm_14b():
    print("⏸️  Stopping vLLM 14B...")
    subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True)
    time.sleep(8)
    print("✅ vLLM 14B stopped")

def start_vllm_14b():
    print("▶️  Restarting vLLM 14B...")
    cmd = [
        "vllm", "serve", VLLM_MODEL,
        "--quantization", "awq",
        "--tensor-parallel-size", "2",
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.50",
        "--port", "8000"
    ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("⏳ Waiting for vLLM 14B...")
    for i in range(60):
        try:
            r = httpx.get(f"{VLLM_HOST}/health", timeout=3)
            if r.status_code == 200:
                print("✅ vLLM 14B ready!")
                return True
        except:
            pass
        time.sleep(5)
        print(f"  Waiting... ({(i+1)*5}s)")
    print("❌ vLLM 14B failed to restart")
    return False

def start_vlm():
    global _vlm_process
    print("🎨 Starting Qwen2.5-VL-7B...")
    cmd = [
        "vllm", "serve", VLM_MODEL,
        "--tensor-parallel-size", "2",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.60",
        "--port", "8002",
        "--enforce-eager",
        "--limit-mm-per-prompt", '{"image": 3}'
    ]
    _vlm_process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print("⏳ Waiting for VLM...")
    for i in range(60):
        try:
            r = httpx.get(f"{VLM_HOST}/health", timeout=3)
            if r.status_code == 200:
                print("✅ Qwen2.5-VL-7B ready!")
                return True
        except:
            pass
        time.sleep(5)
        print(f"  Loading... ({(i+1)*5}s)")
    print("❌ VLM failed to start")
    return False

def stop_vlm():
    global _vlm_process
    print("⏸️  Stopping VLM...")
    subprocess.run(["pkill", "-9", "-f", "Qwen2.5-VL"], capture_output=True)
    if _vlm_process:
        _vlm_process.terminate()
        _vlm_process = None
    time.sleep(5)
    print("✅ VLM stopped")