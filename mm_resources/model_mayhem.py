#!/usr/bin/env python3
"""
llama_guardian
Flask application for managing HuggingFace model downloads, conversion, and serving
using llama.cpp. It supports GPU memory checks, on-demand model loading,
and proxies requests to the underlying llama-server instance.

Modified to:
  - Remove numeric labels in HTML sections
  - Add SSE-based model-loading logs so the UI can show real-time feedback
  - Enforce linear (one-at-a-time) model loads with a global lock
  - Refresh GPU stats every 3 seconds (instead of only on demand)
  - Highlight entire row in the models table on mouse hover
"""

import os
import time
import subprocess
import json
import threading
import uuid
import re
from typing import Union, Dict, List, Any, Optional, Tuple

import requests
from flask import Flask, request, jsonify, Response, stream_with_context
import torch
from huggingface_hub import snapshot_download
import logging

# Configure logging (adjust level and formatting as needed)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Global dictionary for tracking conversion jobs (HuggingFace download + convert)
jobs: Dict[Any, Any] = {}

# Global dictionary for tracking model-loading jobs (SSE streaming)
model_load_jobs: Dict[str, Any] = {}

app = Flask(__name__)

# Configuration defaults and memory parameters
MODEL_DIR: str = os.environ.get("MODEL_DIR", "/models")
HF_CACHE: str = os.environ.get("HF_CACHE", "/hf_cache")
START_PORT: int = 11435  # Starting port for llama-server instances

# Memory parameters
KERNEL_MEMORY: int = 150 * 1024 * 1024  # 150 MB in bytes
MIN_CONTEXT_TOKENS: int = 8000
MAX_CONTEXT_TOKENS: int = 64000
TOKEN_MEMORY_RATIO: int = 2000  # estimated bytes per token

# Multiplier for model on-disk size to approximate GPU memory usage
SIZE_MULTIPLIER: float = 1.15

# Enable output logging from llama.cpp servers if desired
PRINT_LLM_OUTPUT: bool = os.environ.get("PRINT_LLM_OUTPUT", "false").lower() in (
    "true", "1", 1, "yes"
)
if PRINT_LLM_OUTPUT:
    logging.info("llama.cpp output logging enabled")

# Global dictionary mapping model name to server information.
# { model_name: {"process": <subprocess.Popen>, "port": <int>, "gpu": <gpu_id>} }
loaded_models: Dict[str, Dict[str, Any]] = {}
lock = threading.Lock()
# Global dictionary for tracking update jobs (only one allowed at a time)
update_jobs: Dict[str, Any] = {}

# NEW: Global lock for forcing linear model loading
model_load_lock = threading.Lock()


class HealthFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        # Suppress logs for successful /health requests (status 200)
        if "/health" in message and "200" in message:
            return False
        return True


logging.getLogger('werkzeug').addFilter(HealthFilter())


def get_available_gpu(model_size_bytes: int, suggested_gpus: Optional[List[int]] = None) -> Optional[List[int]]:
    """
    Determine the available GPU(s) to load the model based on its size.
    If suggested_gpus is provided, check if they have enough memory.
    Returns a list of GPU indices if enough memory is available, else None.
    """
    if not torch.cuda.is_available():
        return None

    required_min: float = (model_size_bytes * SIZE_MULTIPLIER) + KERNEL_MEMORY + (
        MIN_CONTEXT_TOKENS * TOKEN_MEMORY_RATIO)
    logging.info(f"Required minimum GPU memory: {required_min / 1024 / 1024:.2f} MB")

    # If suggested GPUs were provided, try to use them first.
    if suggested_gpus:
        # Check each suggested GPU individually.
        for gpu in suggested_gpus:
            free_memory, total_memory = torch.cuda.memory.mem_get_info(gpu)
            logging.info(f"Suggested GPU {gpu} free memory: {free_memory / 1024 / 1024:.2f} MB")
            if free_memory >= required_min:
                logging.info(f"Using suggested GPU {gpu} because it has sufficient memory.")
                return [gpu]
        # If no single GPU qualifies, check if combined free memory on suggested GPUs is enough.
        total_free = sum(torch.cuda.memory.mem_get_info(gpu)[0] for gpu in suggested_gpus)
        if total_free >= required_min:
            logging.info(f"Using suggested GPUs {suggested_gpus} as their combined memory is sufficient.")
            return suggested_gpus
        else:
            logging.info("Suggested GPUs do not have sufficient memory. Falling back to auto-selection.")

    # Default auto-selection: check all available GPUs.
    num_gpus: int = torch.cuda.device_count()
    available_info: List[Tuple[int, int]] = []
    for i in range(num_gpus):
        free_memory, total_memory = torch.cuda.memory.mem_get_info(i)
        logging.info(f"GPU {i} free memory: {free_memory / 1024 / 1024:.2f} MB / "
                     f"{total_memory / 1024 / 1024:.2f} MB total")
        if free_memory >= required_min:
            logging.info(f"GPU {i} is available for model loading as a single device")
            return [i]
        available_info.append((i, free_memory))

    total_free = sum(mem for _, mem in available_info)
    if total_free < required_min:
        logging.warning("Not enough combined GPU memory available.")
        return None

    # Select GPUs in descending order of free memory until the required amount is met.
    available_info.sort(key=lambda x: x[1], reverse=True)
    selected_gpus: List[int] = []
    accumulated_free: int = 0
    for gpu_index, free_mem in available_info:
        selected_gpus.append(gpu_index)
        accumulated_free += free_mem
        if accumulated_free >= required_min:
            logging.info(f"Using GPUs {selected_gpus} to load the model (combined memory sufficient)")
            return selected_gpus

    return None


def find_free_port() -> int:
    """
    Find a free port number starting from START_PORT that is not used by any loaded model.
    """
    global START_PORT
    port: int = START_PORT
    while True:
        if all(info['port'] != port for info in loaded_models.values()):
            return port
        port += 1


def compute_context_tokens(gpu_id: Optional[Union[str, int]], model_size_bytes: int) -> int:
    """
    Compute how many context tokens are feasible based on free GPU memory.
    """
    if gpu_id is None:
        return MIN_CONTEXT_TOKENS

    # Convert GPU id to int if needed
    if isinstance(gpu_id, str):
        if ',' in gpu_id:
            gpu_id = int(gpu_id.split(',')[0])
        else:
            gpu_id = int(gpu_id)

    free_memory, total_memory = torch.cuda.memory.mem_get_info(gpu_id)
    free_context_memory: int = int(free_memory) - int(
        model_size_bytes * SIZE_MULTIPLIER) - KERNEL_MEMORY
    if free_context_memory < (MIN_CONTEXT_TOKENS * TOKEN_MEMORY_RATIO):
        return MIN_CONTEXT_TOKENS

    context_tokens: int = int(free_context_memory // TOKEN_MEMORY_RATIO)
    return max(MIN_CONTEXT_TOKENS, min(context_tokens, MAX_CONTEXT_TOKENS))


def launch_llama_server_process(
        model_name: str,
        model_path: str,
        gpu_id: Union[str, int],
        model_size_bytes: int) -> Tuple[subprocess.Popen, int]:
    """
    Launch a llama-server process to serve the model.
    Returns (Popen, port).  The caller is responsible for reading the logs from the process.
    """
    port: int = find_free_port()
    context_tokens: int = compute_context_tokens(gpu_id, model_size_bytes)
    command: List[str] = [
        "llama-server",
        "--port", str(port),
        "--host", "127.0.0.1",
        "-ngl", "9999",
        "-m", model_path,
        "-c", str(context_tokens),
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "-fa"
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logging.info(
        f"Launching llama-server for model '{model_name}' on "
        f"GPU {gpu_id} with context tokens {context_tokens}")
    logging.info("Command: %s", " ".join([f"CUDA_VISIBLE_DEVICES={str(gpu_id)}"] + command[1:]))

    proc: subprocess.Popen = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    return proc, port


def shutdown_oldest_model() -> bool:
    """
    Shut down the oldest loaded model (FIFO in loaded_models) to free GPU memory.
    """
    with lock:
        if not loaded_models:
            return False
        oldest_model_name, oldest_info = next(iter(loaded_models.items()))
        logging.info(f"Shutting down oldest model: {oldest_model_name} on GPU {oldest_info['gpu']}")
        oldest_info["process"].terminate()
        try:
            oldest_info["process"].wait(timeout=10)
        except subprocess.TimeoutExpired:
            logging.warning("Oldest model process did not terminate in time, killing process")
            oldest_info["process"].kill()
        del loaded_models[oldest_model_name]
        return True


def get_filtered_models() -> List[str]:
    """
    Return model filenames in MODEL_DIR that match .gguf or partial .gguf parts.
    """
    try:
        all_files: List[str] = os.listdir(MODEL_DIR)
        models: List[str] = []
        for f in all_files:
            if f.endswith(".gguf") or (".gguf.part" in f and "of" in f):
                models.append(f)
        models.sort()
        return models
    except Exception:
        return []


def load_model(model_name: str, suggested_gpus: Optional[List[int]] = None) -> int:
    """
    Synchronous load of a model if not already loaded.
    Accepts an optional suggested_gpus list. If provided and sufficient memory is available,
    those GPUs are used; otherwise, auto-selection is performed.
    Returns the port of the llama-server.
    """
    if model_name in loaded_models:
        return loaded_models[model_name]["port"]

    with lock:
        if model_name in loaded_models:
            return loaded_models[model_name]["port"]
        else:
            # If it's a HuggingFace URL ending in .gguf, download it.
            if model_name.startswith("https://huggingface.co/") and model_name.endswith(".gguf"):
                https_model_name: str = model_name.split("/")[-1]
                if https_model_name in loaded_models:
                    return loaded_models[https_model_name]["port"]
                if https_model_name in get_filtered_models():
                    model_name = https_model_name
                else:
                    logging.info(f"Downloading model '{model_name}'")
                    model_path: str = os.path.join(MODEL_DIR, os.path.basename(model_name))
                    response = requests.get(model_name, stream=True)
                    response.raise_for_status()
                    model_size_bytes: int = int(response.headers.get('content-length', 0))
                    bytes_written: int = 0
                    start_t: float = time.time()
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            bytes_written += len(chunk)
                            f.write(chunk)
                    logging.info(f"Downloaded {bytes_written} bytes in {time.time()-start_t:.2f}s.")
                    model_name = https_model_name

        model_path: str = os.path.abspath(os.path.join(MODEL_DIR, model_name))
        if not model_path.startswith(MODEL_DIR):
            logging.error("Invalid model path")
            raise ValueError("Invalid model path")
        if not os.path.exists(model_path):
            logging.error(f"Model '{model_name}' not found in {MODEL_DIR}")
            raise FileNotFoundError(f"Model '{model_name}' not found in {MODEL_DIR}")

        # Enforce one-at-a-time model loading.
        with model_load_lock:
            model_size_bytes: int = os.path.getsize(model_path)
            gpu_list: Optional[List[int]] = get_available_gpu(model_size_bytes, suggested_gpus)
            if gpu_list is None:
                if shutdown_oldest_model():
                    logging.info("Waiting 2 seconds after shutting down an older model...")
                    time.sleep(2)
                    gpu_list = get_available_gpu(model_size_bytes, suggested_gpus)
                if gpu_list is None:
                    logging.error("No available GPU with sufficient memory to load the model")
                    raise Exception("No available GPU memory")

            gpu_str: str = ",".join(str(i) for i in gpu_list)
            proc, port = launch_llama_server_process(model_name, model_path, gpu_str, model_size_bytes)
            loaded_models[model_name] = {"process": proc, "port": port, "gpu": gpu_str}

            output_lines: List[str] = []
            start_time: float = time.time()
            timeout: int = 600
            while True:
                line: bytes = proc.stderr.readline()
                if not line:
                    line = proc.stdout.readline()
                    if proc.poll() is not None:
                        logging.error("llama-server process terminated unexpectedly")
                        raise Exception("llama-server process terminated unexpectedly")
                    continue
                decoded_line: str = line.decode('utf-8', errors="replace").strip()
                output_lines.append(decoded_line)
                if "server is listening on" in decoded_line:
                    break
                if time.time() - start_time > timeout:
                    logging.error("Timeout waiting for llama-server to start")
                    raise Exception("Timeout waiting for llama-server to start")

            if PRINT_LLM_OUTPUT:
                logging.info("llama-server startup output:\n%s", "\n".join(output_lines))

            while True:
                time.sleep(2)
                try:
                    r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
                    if r.status_code == 200 and r.json().get("status") == "ok":
                        logging.info(f"Model '{model_name}' loaded on GPU {gpu_str} at port {port}")
                        return port
                except Exception:
                    pass


def background_model_load(job_id: str, model_name: str, suggested_gpus: Optional[List[int]] = None):
    """
    Background thread to load the model with SSE-friendly log capture.
    Accepts an optional suggested_gpus list to try first.
    """
    model_load_jobs[job_id]['status'] = 'loading'
    logs = []

    def push_log(msg: str):
        logs.append(msg)
        model_load_jobs[job_id]['logs'] = logs[:]

    try:
        push_log(f"Starting load of {model_name}...")

        if model_name in loaded_models:
            push_log(f"Model {model_name} already loaded.")
            model_load_jobs[job_id]['status'] = 'completed'
            return

        model_path: str = os.path.abspath(os.path.join(MODEL_DIR, model_name))
        if not model_path.startswith(MODEL_DIR):
            raise ValueError("Invalid model path")

        if not os.path.exists(model_path):
            push_log(f"Model file {model_name} not found. Will not load.")
            model_load_jobs[job_id]['status'] = 'failed'
            model_load_jobs[job_id]['error'] = "File not found."
            return

        push_log("Waiting to acquire global load lock (only one load at a time)...")
        with model_load_lock:
            push_log("Lock acquired. Checking GPU memory...")
            model_size_bytes: int = os.path.getsize(model_path)
            gpu_list: Optional[List[int]] = get_available_gpu(model_size_bytes, suggested_gpus)
            if gpu_list is None:
                push_log("No GPU available, trying to unload oldest model...")
                if shutdown_oldest_model():
                    push_log("Oldest model unloaded. Retrying GPU check...")
                    time.sleep(2)
                    gpu_list = get_available_gpu(model_size_bytes, suggested_gpus)
                if gpu_list is None:
                    push_log("Still no GPU memory available. Aborting load.")
                    model_load_jobs[job_id]['status'] = 'failed'
                    model_load_jobs[job_id]['error'] = 'No GPU memory'
                    return

            gpu_str: str = ",".join(str(i) for i in gpu_list)
            push_log(f"Launching llama-server on GPU(s) {gpu_str}...")
            proc, port = launch_llama_server_process(model_name, model_path, gpu_str, model_size_bytes)

            with lock:
                loaded_models[model_name] = {"process": proc, "port": port, "gpu": gpu_str}

            start_time: float = time.time()
            timeout: int = 600

            while True:
                line: bytes = proc.stderr.readline()
                if not line:
                    line = proc.stdout.readline()

                if proc.poll() is not None:
                    push_log("llama-server process exited unexpectedly.")
                    model_load_jobs[job_id]['status'] = 'failed'
                    model_load_jobs[job_id]['error'] = "Process died."
                    return

                if line:
                    decoded_line: str = line.decode('utf-8', errors="replace").rstrip('\n')
                    push_log(decoded_line)
                    if "server is listening on" in decoded_line:
                        break

                if time.time() - start_time > timeout:
                    push_log("Timeout waiting for llama-server to start.")
                    model_load_jobs[job_id]['status'] = 'failed'
                    model_load_jobs[job_id]['error'] = "Timeout"
                    return

            push_log("Checking server health...")
            while True:
                time.sleep(2)
                try:
                    r = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
                    if r.status_code == 200 and r.json().get("status") == "ok":
                        push_log("Model loaded and healthy!")
                        model_load_jobs[job_id]['status'] = 'completed'
                        return
                    else:
                        push_log("Not healthy yet, waiting..")
                except Exception as exc:
                    push_log(f"Health check error: {exc}")

    except Exception as e:
        logging.exception("Error loading model in background:")
        model_load_jobs[job_id]['status'] = 'failed'
        model_load_jobs[job_id]['error'] = str(e)


@app.route('/load_sse', methods=['POST'])
def load_sse_api():
    """
    POST JSON { "model_name": "xyz.gguf", "suggested_gpus": [0, 1] }
    Returns { "job_id": ... }
    """
    data = request.get_json(silent=True) or {}
    model_name = data.get('model_name')
    suggested_gpus = data.get("suggested_gpus")  # expected as a list of integers
    if not model_name:
        return jsonify({"error": "Missing model_name"}), 400

    job_id = str(uuid.uuid4())
    model_load_jobs[job_id] = {
        "status": "starting",
        "logs": [],
        "model_name": model_name,
        "suggested_gpus": suggested_gpus
    }
    t = threading.Thread(target=background_model_load, args=(job_id, model_name, suggested_gpus), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route('/load_sse/status/<job_id>')
def load_sse_status(job_id: str):
    """
    SSE endpoint to stream loading logs. Ends once status is 'completed' or 'failed'.
    """
    def generate():
        last_len = 0
        while True:
            job = model_load_jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'job not found'})}\n\n"
                break

            status = job['status']
            logs = job['logs']
            if len(logs) > last_len:
                # send any new logs
                new_segment = logs[last_len:]
                payload = {
                    "status": status,
                    "logs": new_segment,
                    "error": job.get('error', '')
                }
                yield f"data: {json.dumps(payload)}\n\n"
                last_len = len(logs)

            if status in ['completed', 'failed']:
                break
            time.sleep(1)

    return Response(generate(), content_type='text/event-stream')


def background_update(job_id: str, rm_build: bool = False):
    """
    Background thread to run the update script and capture all logs.
    """
    update_jobs[job_id]['logs'].append("Starting update process...")
    command = "/mm_resources/update_llama.cpp.sh"
    if rm_build:
        command += " -rm_build"
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr with stdout so all output is captured.
            shell=True,
            bufsize=1,
            universal_newlines=True  # Enable text mode for line-by-line iteration.
        )
    except Exception as e:
        update_jobs[job_id]['status'] = 'failed'
        update_jobs[job_id]['error'] = str(e)
        return

    for line in iter(process.stdout.readline, ""):
        if line:
            update_jobs[job_id]['logs'].append(line.rstrip())
    process.stdout.close()
    retcode = process.wait()
    if retcode == 0:
        update_jobs[job_id]['logs'].append("Update process finished successfully.")
        update_jobs[job_id]['status'] = 'completed'
    else:
        update_jobs[job_id]['logs'].append(f"Update process failed with return code {retcode}.")
        update_jobs[job_id]['status'] = 'failed'


@app.route('/update_llama', methods=['POST'])
def update_llama_endpoint():
    """
    POST JSON { "password": "XXXX_password_here", "rm_build": False }
    Starts the update process as a background job and returns a job_id.
    Only one update job can be running at a time.
    """
    data = request.get_json(silent=True) or {}
    provided_password = data.get("password", "")
    rm_build = data.get("rm_build", False)
    token_file = "/mm_resources/token.txt"
    
    try:
        with open(token_file, "r") as f:
            expected_password = f.readline().strip()
    except Exception as e:
        return jsonify({"error": f"Error reading token file: {str(e)}"}), 500

    if provided_password != expected_password:
        return jsonify({"error": "Unauthorized"}), 401

    # Check if an update job is already running
    for job in update_jobs.values():
        if job['status'] not in ['completed', 'failed']:
            return jsonify({"error": "An update job is already in progress"}), 409

    job_id = str(uuid.uuid4())
    update_jobs[job_id] = {
        "status": "starting",
        "logs": []
    }
    t = threading.Thread(target=background_update, args=(job_id,rm_build,), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route('/update_llama/status/<job_id>')
def update_llama_status(job_id: str):
    """
    SSE endpoint to stream update logs in real time.
    Ends once the update job status is 'completed' or 'failed'.
    """
    def generate():
        last_len = 0
        while True:
            job = update_jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'job not found'})}\n\n"
                break

            status = job.get('status', '')
            logs = job.get('logs', [])
            if len(logs) > last_len:
                new_segment = logs[last_len:]
                payload = {
                    "status": status,
                    "logs": new_segment,
                    "error": job.get('error', '')
                }
                yield f"data: {json.dumps(payload)}\n\n"
                last_len = len(logs)
            if status in ['completed', 'failed']:
                break
            time.sleep(1)
    return Response(generate(), content_type='text/event-stream')


def download_and_convert(job_id: str, repo_id: str) -> None:
    """
    Background thread: download a HF repo to HF_CACHE, convert to .gguf in MODEL_DIR.
    """
    try:
        jobs[job_id]['status'] = 'downloading'
        jobs[job_id]['progress'] = 0

        model_folder_name: str = repo_id.split('/')[-1]
        cache_dir: str = os.path.abspath(f"{HF_CACHE}/{model_folder_name}")
        if not cache_dir.startswith(HF_CACHE):
            raise ValueError("Invalid cache directory")
        os.makedirs(cache_dir, exist_ok=True)

        output_file: str = os.path.abspath(f"{MODEL_DIR}/{model_folder_name}_q8_0.gguf")
        if not output_file.startswith(MODEL_DIR):
            raise ValueError("Invalid output directory")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        jobs[job_id]['status'] = 'Downloading...'
        jobs[job_id]['progress'] = 1
        # Download snapshot from HF
        snapshot_download(
            repo_id=repo_id,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            revision="main"
        )

        jobs[job_id]['status'] = 'converting'
        jobs[job_id]['progress'] = 50

        convert_cmd: List[str] = [
            "python",
            "/mm_resources/llama.cpp/convert_hf_to_gguf.py",
            cache_dir,
            "--outfile", output_file,
            "--outtype", "q8_0"
        ]
        p = subprocess.Popen(
            convert_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        done_pattern = re.compile(r"Writing:\s+(\d+)%")
        for line in iter(p.stdout.readline, ''):
            logging.info(line.strip())
            done_match = done_pattern.search(line)
            jobs[job_id]['progress'] = (int(done_match.group(1))/2 + 50) if done_match else jobs[job_id]['progress']
            jobs[job_id]['message'] = f"{line.strip()}</br>"
        p.wait()

        if p.returncode == 0:
            jobs[job_id]['message'] = (
                f"</br>Conversion completed successfully</br>"
                f"<a href=\"/go/{model_folder_name}_q8_0.gguf\">Test {model_folder_name}_q8_0.gguf</a>"
            )
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['progress'] = 100
            jobs[job_id]['result_path'] = output_file
        else:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = 'Conversion process failed'
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)


@app.route('/oob_download', methods=['POST'])
def oob_download_api() -> Union[Dict[str, Any], Tuple[Dict[str, Any], int]]:
    """
    POST endpoint to trigger a background job to download & convert a HF repo.
    Returns { job_id } on success.
    """
    data = request.get_json() or {}
    repo_id: Optional[str] = data.get('repo_id')
    if not repo_id:
        return {'error': 'Missing repo_id parameter'}, 400

    job_id: str = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'initializing',
        'progress': 0,
        'repo_id': repo_id,
        'start_time': time.time()
    }
    t = threading.Thread(target=download_and_convert, args=(job_id, repo_id), daemon=True)
    t.start()

    return {'job_id': job_id}


@app.route('/oob_download/status/<job_id>')
def job_status_sse(job_id: str) -> Response:
    """
    SSE to stream job progress updates.
    """
    def generate():
        last_status = None
        while True:
            if job_id not in jobs:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break

            job = jobs[job_id]
            current_status = json.dumps(job)

            if current_status != last_status:
                yield f"data: {current_status}\n\n"
                last_status = current_status

            if job['status'] in ['completed', 'failed']:
                break
            time.sleep(1)
    return Response(generate(), content_type='text/event-stream')


def proxy_model_request() -> Response:
    """
    Proxy the incoming API request to the llama-server instance for the requested model.
    Supports streaming if "stream":true in the JSON payload.
    """
    kwargs: Dict[str, Any] = {}
    method: str = request.method
    path: str = request.path
    streaming: bool = False

    if request.is_json:
        kwargs["json"] = request.get_json(silent=True)
        if kwargs["json"] and kwargs["json"].get("stream", False):
            streaming = True
    else:
        kwargs["data"] = request.data

    if request.args:
        kwargs["params"] = request.args

    model_payload: Dict[str, Any] = kwargs.get("json", {})
    model_name: Optional[str] = model_payload.get("model")
    if not model_name:
        return jsonify({"error": "No model specified"}), 400

    logging.info(f"Proxy request for model: {model_name}")
    try:
        port: int = load_model(model_name)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    target_url: str = f"http://localhost:{port}{path}"

    # Forward relevant headers
    forwarded_headers = {
        h: v
        for h, v in request.headers.items()
        if h.lower() not in ["host", "content-length", "connection"]
    }
    kwargs["headers"] = forwarded_headers
    if streaming:
        kwargs["headers"]["Accept"] = "text/event-stream"

    try:
        resp = requests.request(method, target_url, stream=streaming, **kwargs)
    except Exception as e:
        return jsonify({"error": f"Error forwarding request: {str(e)}"}), 500

    excluded = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = [(k, v) for k, v in resp.headers.items() if k.lower() not in excluded]
    headers.append(('X-Accel-Buffering', 'no'))

    if streaming:
        def generate_and_print():
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    yield line + "\n\n"
        return Response(
            stream_with_context(generate_and_print()),
            mimetype='text/event-stream',
            status=resp.status_code,
            headers=headers
        )
    else:
        return Response(resp.content, status=resp.status_code, headers=headers)


def extract_charset(content_type: str) -> str:
    """
    Extract the charset from a Content-Type header.
    Returns the charset if found, else returns None.
    """
    m = re.search(r'charset=([\w-]+)', content_type, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


@app.route("/go/<model_name>/", defaults={"subpath": ""}, methods=["GET", "POST", "HEAD"])
@app.route("/go/<model_name>/<path:subpath>", methods=["GET", "POST", "HEAD"])
def go_model(model_name: str, subpath: str) -> Response:
    """
    Legacy route: Load the specified model if needed, then proxy to /go/<model>/...
    """
    try:
        port: int = load_model(model_name)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    target_url: str = f"http://localhost:{port}/{subpath}" if subpath else f"http://localhost:{port}/"
    method: str = request.method
    kwargs: dict = {}
    streaming: bool = False

    if request.is_json:
        data = request.get_json(silent=True)
        kwargs["json"] = data
        if data and data.get("stream"):
            streaming = True
    else:
        kwargs["data"] = request.data

    if request.args:
        kwargs["params"] = request.args

    forwarded_headers = {
        h: v
        for h, v in request.headers.items()
        if h.lower() not in ["host", "content-length", "connection"]
    }
    kwargs["headers"] = forwarded_headers

    try:
        resp = requests.request(method, target_url, stream=streaming, **kwargs)
    except Exception as e:
        return jsonify({"error": f"Error forwarding request: {str(e)}"}), 500

    original_content_type = resp.headers.get('Content-Type', '')
    # Only set the encoding if the original Content-Type includes a charset.
    if 'charset=' in original_content_type.lower():
        charset = extract_charset(original_content_type)
        if charset:
            resp.encoding = charset

    # Exclude headers that may interfere with the proxying.
    excluded = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = [(k, v) for k, v in resp.headers.items() if k.lower() not in excluded]
    headers.append(('X-Accel-Buffering', 'no'))

    if streaming:
        def generate_and_print():
            # If resp.encoding is not set, iter_lines will default to its internal fallback.
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    yield line + "\n\n"
        return Response(
            stream_with_context(generate_and_print()),
            mimetype=resp.headers.get('Content-Type', 'text/event-stream'),
            status=resp.status_code,
            headers=headers
        )
    else:
        return Response(resp.content, status=resp.status_code, headers=headers)


@app.route("/models", methods=["GET"])
def list_models() -> Response:
    """
    Returns JSON with "available_models" from MODEL_DIR.
    """
    try:
        models = get_filtered_models()
        return jsonify({"available_models": models})
    except Exception as e:
        return jsonify({"error": f"Error listing models: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health_status() -> Response:
    """
    Returns JSON describing loaded models and GPU memory usage,
    including GPU names if available.
    """
    status: Dict[str, Any] = {}
    status["loaded_models"] = {
        model: {"port": info["port"], "gpu": info["gpu"]}
        for model, info in loaded_models.items()
    }

    gpus: Dict[Any, Any] = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free_mem, total_mem = torch.cuda.memory.mem_get_info(i)
            try:
                gpu_name = torch.cuda.get_device_name(i)
            except:
                gpu_name = f"GPU_{i}"
            gpus[i] = {
                "name": gpu_name,
                "total_memory": total_mem,
                "allocated_memory": total_mem - free_mem,
                "free_memory": free_mem
            }
    else:
        gpus["error"] = "No CUDA available."
    status["gpus"] = gpus
    return jsonify(status)


@app.route('/unload', methods=['POST'])
def unload_models() -> Response:
    """
    POST JSON: { "models": ["model1.gguf", ...] }
    to unload them. Returns which were unloaded.
    """
    data = request.get_json(silent=True) or {}
    models_to_unload = data.get('models', [])
    if not models_to_unload:
        return jsonify({"error": "No models selected"}), 400

    unloaded = []
    with lock:
        for m in models_to_unload:
            if m in loaded_models:
                info = loaded_models[m]
                info['process'].terminate()
                try:
                    info['process'].wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logging.warning(f"Model process {m} didn't terminate in time. Killing.")
                    info['process'].kill()
                del loaded_models[m]
                unloaded.append(m)

    return jsonify({"unloaded": unloaded, "success": True}), 200


@app.route('/version', methods=['GET'])
def get_llama_server_version():
    """
    GET endpoint to retrieve the current version of the llama-server binary.
    
    This endpoint executes the 'llama-server --version' command on demand to fetch
    the latest version information. Caching is explicitly disabled so that any
    updates or recompilations are immediately reflected in the output.
    
    Returns:
        A JSON response with the version information:
            { "version": "<version_string>" }
        In case of an error, returns a JSON error message with a 500 status code.
    """
    try:
        result = subprocess.run(
            ["llama-server", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True,
            check=True
        )
        version_output = result.stdout.strip()
        response = jsonify({"version": version_output})
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response
    except Exception as e:
        logging.exception("Error retrieving llama-server version")
        return jsonify({"error": "Failed to retrieve version", "details": str(e)}), 500


@app.route("/", methods=["GET"])
def unified_interface() -> str:
    """
    Single HTML page (dashboard) that merges:
      - GPU stats
      - Models table (on-disk + loaded)
      - HuggingFace download & conversion
      - SSE-based load logs (now with one-at-a-time enforcement)
    """
    return open("/mm_resources/mm.html").read()


@app.route("/", methods=["POST"])
@app.route("/<path:path>", methods=["POST"])
def catch_all_proxy(path: str = "") -> Response:
    """
    Catch-all for POST requests, typically for OpenAI-style completions that specify "model" in JSON.
    """
    if path in ["models", "health"]:
        if path == "models":
            return list_models()
        else:
            return health_status()
    return proxy_model_request()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Model Mayhem server.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--port", type=int, default=11434, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on.")
    args = parser.parse_args()
    app.run(port=args.port, host=args.host)
