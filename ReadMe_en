## F5-TTS Tensorrt-LLM Faster

Accelerating inference for F5-TTS, with test samples as follows:

+ `NVIDIA GeForce RTX 3090`
+ Test text: `这点请您放心，估计是我的号码被标记了，请问您是沈沈吗？`

After testing, the inference speed was reduced from `3.2s` to `0.72s`, a 4x speed improvement!

The basic approach is to first export `F5-TTS` using `ONNX`, and then use `Tensorrt-LLM` to accelerate the `Transformer` component.

Special thanks to the following open-source projects:
+ https://github.com/DakeQQ/F5-TTS-ONNX
+ https://github.com/Bigfishering/f5-tts-trtllm

All references are at the end of the document.

> The project build time is approximately **3 hours**, so it is recommended to build in tmux.

---

## Install

```shell
conda create -n f5_tts_faster python=3.10 -y
source activate f5_tts_faster
```

```shell
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

### F5-TTS Environment

```shell
# huggingface-cli download --resume-download SWivid/F5-TTS --local-dir ./F5-TTS/ckpts
git clone https://github.com/SWivid/F5-TTS.git   # 0.3.4
cd F5-TTS
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```shell
# Modify the source code to load the local vocoder
vim /home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages/f5_tts/infer/infer_cli.py
# If installed from source, the file is located at F5-TTS/src/f5_tts/infer/infer_cli.py
# Around line 124, comment out `vocoder_local_path = "../checkpoints/vocos-mel-24khz"` and change to the local path:
# vocoder_local_path = "/home/wangguisen/projects/tts/f5tts_faster/ckpts/vocos-mel-24khz"

# Run F5-TTS inference
f5-tts_infer-cli \
--model "F5-TTS" \
--ref_audio "./assets/wgs-f5tts_mono.wav" \
--ref_text "那到时候再给你打电话，麻烦你注意接听。" \
--gen_text "这点请您放心，估计是我的号码被标记了，请问您是沈沈吗？" \
--vocoder_name "vocos" \
--load_vocoder_from_local \
--ckpt_file "./ckpts/F5TTS_Base/model_1200000.pt" \
--speed 1.2 \
--output_dir "./output/" \
--output_file "f5tts_wgs_out.wav"
```

### F5-TTS-Faster Environment:
```shell
conda install -c conda-forge ffmpeg cmake openmpi -y

# Set the C compiler for OpenMPI, ensuring gcc is installed
# conda install -c conda-forge compilers
# which gcc
# gcc --version
export OMPI_CC=$(which gcc)
export OMPI_CXX=$(which g++)
pip install -r ./requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```python
# Check if onnxruntime supports CUDA
import onnxruntime as ort
print(ort.get_available_providers())
```

### TensorRT-LLM Environment
```shell
sudo apt-get -y install libopenmpi-dev
pip install tensorrt_llm==0.15.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Verify TensorRT-LLM installation:
```python
python -c "import tensorrt_llm"
python -c "import tensorrt_llm.bindings"
```

---

## Convert to ONNX

Export `F5-TTS` to `ONNX`:

```python
python ./export_onnx/Export_F5.py
```

```python
# Perform inference with ONNX alone
python ./export_onnx/F5-TTS-ONNX-Inference.py
```

The exported `ONNX` structure is as follows:
```shell
./export_onnx/onnx_ckpt/
├── F5_Decode.onnx
├── F5_Preprocess.onnx
└── F5_Transformer.onnx
```

After converting to ONNX, GPU inference speed may not be significantly faster. For those interested, refer to the following:
```python
# Specify CUDAExecutionProvider
import onnxruntime as ort
sess_options = ort.SessionOptions()
session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"], sess_options=sess_options)

# Quantization:
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("model.onnx", "model_quant.onnx", weight_type=QuantType.QInt8)
```

Note: Since the `F5` and `vocos` source code was modified when exporting ONNX, you need to re-download and re-install them to ensure F5-TTS usability.
```shell
pip uninstall -y vocos && pip install vocos -i https://pypi.tuna.tsinghua.edu.cn/simple
```
```python
# ./export_trtllm/origin_f5 corresponds to F5 source code
cp ./export_trtllm/origin_f5/modules.py ../F5-TTS/src/f5_tts/model/
cp ./export_trtllm/origin_f5/dit.py ../F5-TTS/src/f5_tts/model/backbones/
cp ./export_trtllm/origin_f5/utils_infer.py ../F5-TTS/src/f5_tts/infer/
```

---

## Convert to Trtllm

### Source Code Modifications

After installing `TensorRT-LLM`, the directories need to be moved:

In this project, `export_trtllm/model` corresponds to `tensorrt_llm/models` in the TensorRT-LLM source code.

1. In the `tensorrt_llm/models` directory, create a `f5tts` folder and place the code from this repository in the corresponding directory.

```shell
# Locate tensorrt_llm
ll /home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages | grep tensorrt_llm

# Import source code into tensorrt_llm/models
mkdir /home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages/tensorrt_llm/models/f5tts
cp -r /home/wangguisen/projects/tts/f5tts_faster/export_trtllm/model/* /home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages/tensorrt_llm/models/f5tts
```

+ The `tensorrt_llm/models/f5tts` directory structure:
```shell
/home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages/tensorrt_llm/models/f5tts/
├── model.py
└── modules.py
```

2. Import `f5tts` in `tensorrt_llm/models/__init__.py`:

```shell
vim /home/wangguisen/miniconda3/envs/f5_tts_faster/lib/python3.10/site-packages/tensorrt_llm/models/__init__.py
```

```python
from .f5tts.model import F5TTS

__all__ = [..., 'F5TTS']

# Add the model to `MODEL_MAP`:
MODEL_MAP = {..., 'F5TTS': F5TTS}
```

### Convert Checkpoint
```python
python ./export_trtllm/convert_checkpoint.py \
        --timm_ckpt "./ckpts/F5TTS_Base/model_1200000.pt" \
        --output_dir "./ckpts/trtllm_ckpt"

# --dtype float32
```

### Build Engine
> Supports Tensor parallelism, `--tp_size`

```python
trtllm-build --checkpoint_dir ./ckpts/trtllm_ckpt \
             --remove_input_padding disable \
             --bert_attention_plugin disable \
             --output_dir ./ckpts/engine_outputs
# If there is a parameter mismatch error, it's because the default parameter is fp16, but the network parameters require fp32. Update `_DEFAULT_DTYPE` in tensorrt_llm/parameter.py to trt.DataType.HALF
```

---

## Fast Inference
```python
python ./export_trtllm/sample.py \
        --tllm_model_dir "./ckpts/engine_outputs"
```

---

## References

https://github.com/SWivid/F5-TTS  
https://github.com/DakeQQ/F5-TTS-ONNX  
https://github.com/NVIDIA/TensorRT-LLM  
https://github.com/Bigfishering/f5-tts-trtllm  

