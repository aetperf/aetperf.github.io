---
title: Running a Local LLM on a Consumer GPU (8 GB VRAM)
layout: post
comments: true
author: François Pacull
date: 2026-04-18
categories: [LLM]
tags:
- LLM
- Qwen
- llama.cpp
- GPU
- laptop
- OpenCode
- coding agent
- tool calling
- GGUF
- quantization
image: /img/2026-04-18_01/gpu_poor.jpeg
---

<p align="center">
  <img src="/img/2026-04-18_01/gpu_poor.jpeg" alt="GPU poor" width="600" /><br>
  <b>Any spare VRAM?</b>
</p>

In a [previous post](https://aetperf.github.io/llm/aws/2026/04/11/Running-a-Local-LLM-Coding-Agent-on-AWS.html), we ran [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) on an AWS `g5.xlarge` with an NVIDIA A10G (24 GB VRAM). This follow-up covers the same workflow on a consumer laptop GPU with 8 GB VRAM, using [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), [llama.cpp](https://github.com/ggml-org/llama.cpp) for inference, and [ToolCall-15](https://github.com/stevibe/ToolCall-15) for evaluation. This is mostly out of curiosity; we do not expect anything.

**Outline**

- [System Setup](#1-system-setup)
- [Build llama.cpp with CUDA](#2-build-llamacpp-with-cuda)
- [Download the Model](#3-download-the-model)
- [Start llama-server](#4-start-llama-server)
- [Validate with ToolCall-15](#5-validate-with-toolcall-15)
- [Performance](#6-performance)
- [Connect OpenCode to the Local Server](#7-connect-opencode-to-the-local-server)
- [A Coding Task](#8-a-coding-task)
- [Stopping the Server](#9-stopping-the-server)
- [Reflections](#10-reflections)

## 1. System Setup

The machine is an MSI Vector GP66 12UGSO laptop running Ubuntu Linux.

```
$ nvidia-smi
Thu Apr 16 11:39:43 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.211.01             Driver Version: 570.211.01     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3070 ...    Off |   00000000:01:00.0 Off |                  N/A |
| N/A   44C    P0            752W /  125W |      11MiB /   8192MiB |      6%      Default |
+-----------------------------------------+------------------------+----------------------+
```

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 3070 Ti Laptop (8192 MiB) |
| Compute capability | 8.6 (Ampere) |
| Driver | 570.211.01 |
| CUDA | 12.8 |
| llama.cpp | b8816 (3f7c29d31) |
| OpenCode | 1.4.6 |

## 2. Build llama.cpp with CUDA

We pull the latest changes and build with CUDA support targeting compute capability 8.6.

```
$ cd ~/Workspace/llama.cpp
$ git pull
From github.com:ggml-org/llama.cpp
   e2eb39e81..3f7c29d31  master -> origin/master
```

The conda CUDA install on this machine only ships the runtime libraries, not the development headers (no `cuda_runtime.h`), so nvcc cannot compile the CUDA sources against it. We point CMake at the full system-wide CUDA 12.5 installation instead (the 12.8 driver shown earlier is forward-compatible with the older toolkit):

```
$ cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.5/bin/nvcc \
    -DCUDAToolkit_ROOT=/usr/local/cuda-12.5
```

```
-- Found CUDAToolkit: /usr/local/cuda-12.5/include (found version "12.5.40")
-- CUDA Toolkit found
-- The CUDA compiler identification is NVIDIA 12.5.40
-- Using CMAKE_CUDA_ARCHITECTURES=86
-- Including CUDA backend
-- Configuring done
-- Generating done
```

```
$ cmake --build build --config Release -t llama-server llama-cli -j$(nproc)
```

```
$ ./build/bin/llama-server --version
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7850 MiB):
  Device 0: NVIDIA GeForce RTX 3070 Ti Laptop GPU, compute capability 8.6, VMM: yes, VRAM: 7850 MiB
version: 8816 (3f7c29d31)
built with GNU 11.4.0 for Linux x86_64
```

## 3. Download the Model

We use [Unsloth](https://huggingface.co/unsloth)'s UD-Q4_K_XL quantization of Qwen3.5-9B: importance-matrix-guided mixed quantization (Q4_K/Q5_K/Q6_K across layers) at 5.32 bits per weight. Qwen3.5-9B is multimodal; the repo also ships `mmproj-*.gguf` vision-projection files, but we load only the language weights here since the tests are text-only.

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="unsloth/Qwen3.5-9B-GGUF",
    filename="Qwen3.5-9B-UD-Q4_K_XL.gguf",
    local_dir="/home/francois/Workspace/models"
)
```

```
$ ls -lh ~/Workspace/models/Qwen3.5-9B-UD-Q4_K_XL.gguf
-rw-rw-r-- 1 francois francois 5,6G Apr 16 13:33 Qwen3.5-9B-UD-Q4_K_XL.gguf
```

At 5.6 GB, the model file leaves headroom for the KV cache and compute buffers within the 8 GB VRAM budget.

## 4. Start llama-server

All 33 layers are offloaded to GPU (`-ngl 99`), Q4_0 KV cache compression is enabled for keys and values, and sampling parameters follow the thinking-mode preset for precise coding tasks from [Qwen3.5's recommendations](https://huggingface.co/Qwen/Qwen3.5-9B#using-qwen35-via-the-chat-completions-api):

```
$ ~/Workspace/llama.cpp/build/bin/llama-server \
    -m ~/Workspace/models/Qwen3.5-9B-UD-Q4_K_XL.gguf \
    -ngl 99 \
    -np 1 \
    -c 131072 \
    -ctk q4_0 -ctv q4_0 \
    --reasoning auto --reasoning-budget 2048 \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
    --host 0.0.0.0 --port 8080
```

<p align="center">
  <img src="/img/2026-04-18_01/hf_qwen_parameters.png" alt="Recommended sampling parameters from the Qwen3.5-9B model card" width="800" /><br>
  <b>Recommended sampling parameters from the Qwen3.5-9B model card on Hugging Face.</b>
</p>

### Parallel slots (`-np`)

`-np` controls how many requests the server handles simultaneously, with each slot reserving its own GPU memory. The default is `-np 4`, but for a single-user setup `-np 1` dedicates all resources to one request and saves memory. Some per-slot buffers scale with slot count: 201 MiB with 4 slots vs 50 MiB with 1 slot.

### Context window (`-c`)

Qwen3.5-9B uses a hybrid architecture where only the attention layers contribute to KV cache growth. With Q4_0 quantization, the model remains memory-efficient even at large context sizes:

| Context | KV cache (Q4_0) | Total VRAM | Free |
|---------|----------------|------------|------|
| 32K | 288 MiB | 6177 MiB | 2015 MiB |
| 131K | 1152 MiB | 7041 MiB | 1151 MiB |

At 32K, over 2 GB of VRAM sits unused. Pushing to 131K uses that headroom for a 4x larger context window with no impact on decode speed, leaving 1.1 GB free for compute buffers.

### Thinking mode (`--reasoning`)

Qwen3.5 produces `<think>...</think>` tokens before each response. A short question can trigger 200-400 thinking tokens. The `--reasoning` flag controls this behavior:

| Flag | Effect |
|------|--------|
| `--reasoning on` | Always think, ignores `/no_think` hints |
| `--reasoning off` | Disable thinking entirely, all tokens go to `content` |
| `--reasoning auto` | Think by default (detected from the model's chat template), but respect `/no_think` and `/think` hints in the prompt to toggle per-request |

`--reasoning-budget N` caps thinking tokens per response. At 2048 the model has room to plan without too much latency before the reply begins (around 34 seconds of thinking at 60 tok/s); 0 disables thinking, -1 removes the cap.

### Server log

```
load_tensors: offloaded 33/33 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   545.62 MiB
load_tensors:        CUDA0 model buffer size =  5133.63 MiB
llama_kv_cache: size = 1152.00 MiB (131072 cells, 8 layers, 1/1 seqs), K (q4_0): 576.00 MiB, V (q4_0): 576.00 MiB
llama_memory_recurrent: size =   50.25 MiB (1 cells, 32 layers, 1 seqs)
```

VRAM after loading:

```
$ nvidia-smi
|   0  NVIDIA GeForce RTX 3070 ...    Off |   00000000:01:00.0 Off |                  N/A |
| N/A   51C    P8             20W /  125W |    7041MiB /   8192MiB |      0%      Default |
```

```
$ curl http://localhost:8080/health
{"status":"ok"}
```

```
$ curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen3.5-9B-UD-Q4_K_XL",
      "messages": [{"role":"user","content":"Write a Python function that computes the nth Fibonacci number. Be concise."}],
      "temperature": 0.6, "top_p": 0.95, "max_tokens": 512
    }'
```

The response separates reasoning (`reasoning_content`) from the final answer (`content`).

## 5. Validate with ToolCall-15

[ToolCall-15](https://github.com/stevibe/ToolCall-15) is a deterministic benchmark for tool use, organized into 5 categories with 3 scenarios each (0–2 points per scenario).

```
$ cd ~/Workspace
$ git clone https://github.com/stevibe/ToolCall-15.git
$ cd ToolCall-15
$ npm install
```

`.env` configuration:

```
LLAMACPP_HOST=http://localhost:8080
LLM_MODELS=llamacpp:Qwen3.5-9B-UD-Q4_K_XL
MODEL_REQUEST_TIMEOUT_SECONDS=120
```

```
$ npm run cli -- --temperature 0 --timeout 120
```

### Results

| Category | Score | Tests |
|----------|-------|-------|
| A - Tool Selection | 6/6 | TC-01, TC-02, TC-03 |
| B - Parameter Precision | 4/6 | TC-04, TC-06 pass; TC-05 fail |
| C - Multi-Step Chains | 4/6 | TC-08, TC-09 pass; TC-07 fail |
| D - Restraint & Refusal | 6/6 | TC-10, TC-11, TC-12 |
| E - Error Recovery | 6/6 | TC-13, TC-14, TC-15 |
| **Total** | **26/30** | **87/100** |

The two failures:

- **TC-05** (Date and Time Parsing): the model correctly resolved the date and looked up attendee contacts, but emitted the final `create_calendar_event` call using a raw XML-like format (`<tool_call><function=...>`) instead of the expected OpenAI-style JSON tool call. The llama.cpp server did not parse this as a valid tool call.
- **TC-07** (Search, Read, Act): the model found the file and attempted to read it, but again produced the `read_file` call in the wrong format on the second turn, breaking the chain.

Both failures share the same root cause: the model occasionnally falls back to an XML-like tool-call syntax that the server does not parse as a valid tool call. The other 13 scenarios all produced correctly-formatted calls.

For comparison, the 27B model on the A10G scored 29/30 (97/100), failing only TC-10 (Restraint and Refusal).

## 6. Performance

Timings from the `/v1/chat/completions` response across several requests:

| Prompt tokens | Output tokens | Prefill (tok/s) | Decode (tok/s) |
|--------------|---------------|-----------------|----------------|
| 23 | 512 | 441 | 60.3 |
| 20 | 964 | 399 | 60.0 |
| 40 | 976 | 642 | 59.8 |

Decode speed is consistently **60 tok/s** regardless of output length. Prefill variance is dominated by overhead at these short prompt lengths (20-40 tokens).

| Metric | Value |
|--------|-------|
| Decode speed | 60 tok/s |
| Model size | 8.95B parameters (5.6 GB on disk, 5.55 GiB loaded) |
| KV cache | 1152 MiB (Q4_0, 131K context) |
| Total VRAM | 7041 / 8192 MiB |

60 tok/s is fine for interactive use. Thinking tokens add visible latency: a short question typically generates 200-400 reasoning tokens before the reply begins.

## 7. Connect OpenCode to the Local Server

[OpenCode](https://opencode.ai) is a terminal-based coding agent that connects to any OpenAI-compatible API. Unlike the AWS setup, no SSH tunnel is needed since the server runs locally.

### Install

```
$ curl -fsSL https://opencode.ai/install | bash
```

```
Installing opencode version: 1.4.6
```

### Configure

We add the 9B model entry to the existing `~/.config/opencode/config.json` from the [previous post](https://aetperf.github.io/llm/aws/2026/04/11/Running-a-Local-LLM-Coding-Agent-on-AWS.html), sharing the same provider and base URL:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local-llama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local Llama Server",
      "options": {
        "baseURL": "http://localhost:8080/v1",
        "apiKey": "not-needed"
      },
      "models": {
        "qwen3.5-27b": {
          "name": "Qwen3.5-27B"
        },
        "qwen3.5-9b": {
          "name": "Qwen3.5-9B"
        }
      }
    }
  },
  "model": "local-llama/qwen3.5-9b"
}
```

The top-level `"model"` field selects the active model. Use the `/model` command inside OpenCode to switch without editing the file.

### Launch

```
$ cd ~/your-project
$ opencode
```

<p align="center">
  <img src="/img/2026-04-18_01/opencode_01.png" alt="OpenCode startup screen" width="800" /><br>
  <b>OpenCode startup screen, connected to the local llama-server.</b>
</p>

## 8. A Coding Task

We gave the model the following prompt through OpenCode:

> Implement an optimized Jaro-Winkler similarity in Python.
>
> Requirements: 
>- Time complexity better than naive O(n²). 
>- No external libraries, besides Cython, NumPy, Pandas. 
>- Include unit tests. 
>- Handle edge cases (empty strings, identical strings, unicode). 
>- Benchmark against a naive version.
>- Vectorize for batch comparison. 
>- Make it NumPy-friendly. Avoid Python lists for matches (use arrays / bitsets).

<p align="center">
  <img src="/img/2026-04-18_01/opencode_reasoning.gif" alt="OpenCode showing Qwen3.5-9B reasoning on the Jaro-Winkler task" width="600" /><br>
  <b>Qwen3.5-9B reasoning through the Jaro-Winkler task inside OpenCode.</b>
</p>

OpenCode's waiting/thinking progress bar reminds us of Knight Rider's KITT.

<p align="center">
  <img src="/img/2026-04-18_01/knightrider-kitt.gif" alt="Knight Rider KITT scanner animation" width="600" /><br>
  <b>Knight Rider's KITT scanner.</b>
</p>

Getting from the model's first output to a running project took many iterations. The initial attempt was broken: the Cython extension did not compile, pytest could not even collect the test suite because of broken import paths, and the benchmark script crashed on first run. We went back and forth with the model many times, pasting errors and asking for fixes, before the code actually built and ran.

The final layout is 5 files: a pure Python module, a Cython extension, a test suite (51 tests), a benchmark script, and a `setup.py`, with docstrings and type hints throughout. The tests pass:

```
$ pytest test_jaro_winkler.py
============================= 51 passed in 10.76s ==============================
```

The Jaro formula looks correct: `(m1/n1 + m1/n2 + (m1-t)/m1) / 3`. Identical strings return 1.0, empty strings return 0.0. But the canonical reference cases fail:

```python
>>> jaro_winkler("MARTHA", "MARHTA")
1.0          # expected 0.961
>>> jaro_winkler("AB", "BA")
1.0          # expected 0.0
>>> jaro_winkler("ABCD", "ABDC")
1.0          # expected 0.917
>>> jaro_winkler("hello", "hallo")
0.868        # expected 0.880
```

At first, it looks like transposition counting is broken. The code counts how many matched characters in `s1` find *any* matched character of the same value in `s2`, rather than walking matched characters in positional order. For `MARTHA` vs `MARHTA`, all 6 characters match somewhere, so the count is 6 and transpositions register as 0, making the T/H swap invisible. But there are some other issues anyway, and also regarding efficiency.

The 51 tests pass because their expected ranges match the implementation's outputs rather than reference values: `[hello-hallo-0.86-0.87]` accepts the buggy 0.868, and no test exercises a transposition, so the broken transposition counting slips through. A user relying on this code would silently get wrong results on any input with transpositions.

We asked the model to fix these failures across many follow-up prompts. Each time, it assured us the implementation was correct and the tests were passing. The bugs stayed.

The 9B model produces output that *looks* correct (professional layout, docstrings, type hints, a passing test suite) while being quietly wrong underneath.

Verdict: the Jaro-Winkler implementation is kind of rubbish.

## 9. Stopping the Server

`Ctrl+C` stops a foreground server. For a background process:

```
$ pkill -f llama-server
```

## 10. Reflections

The 9B quantized model fits on an 8 GB consumer GPU at 131K context. Q4_0 compression holds the KV cache to 1152 MiB at that length.

Compared to the 27B model on an A10G, we gain decode speed (60 vs 24 tok/s, the smaller model fits the laptop's memory bandwidth better). We lose tool-call quality (87/100 vs 97/100 on ToolCall-15, though both failures are formatting issues rather than reasoning failures) and VRAM headroom (1.1 GB free vs ~5 GB, though the two runs used different context settings). Thinking tokens add 3-5 seconds of latency before short answers.

For local development and private experimentation the setup is practical. For reliable tool use or long-context coding sessions, the 27B model on a larger GPU is a stronger choice. Our setup choices for the 9B run may also not have been ideal; a different configuration might narrow the gap.
