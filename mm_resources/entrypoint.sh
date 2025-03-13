cd /mm_resources
source /venv/bin/activate
if [ ! -d "/mm_resources/llama.cpp" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git
else
    git config --global --add safe.directory /mm_resources/llama.cpp
    cd llama.cpp
    git pull
fi
export PATH=/mm_resources/llama.cpp/build/bin:$PATH
cd /mm_resources/
chmod +x /mm_resources/update_llama.cpp.sh
./update_llama.cpp.sh &
python model_mayhem.py --port 8080 --host 0.0.0.0