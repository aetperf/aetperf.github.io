---
title: Running a Local LLM Coding Agent on AWS
layout: post
comments: true
author: François Pacull
categories: [LLM, AWS]
tags:
- LLM
- Qwen
- llama.cpp
- AWS
- EC2
- GPU
- OpenCode
- coding agent
- tool calling
- GGUF
- quantization
image: /img/2026-04-11_01/open_code_02.png
---

We deploy [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) on a single GPU instance, serve it with [llama.cpp](https://github.com/ggml-org/llama.cpp), wire it to [OpenCode](https://opencode.ai) as a coding agent frontend, and run [ToolCall-15](https://github.com/stevibe/ToolCall-15) to measure tool-calling quality. The specific model matters less than the setup; new open-weight models land every week, and the procedure is the same for any GGUF-quantized model that fits in VRAM. The quantization is a 4-bit [Unsloth](https://unsloth.ai/) dynamic quantization (UD-Q4_K_XL) that fits in 24 GB of VRAM. "Dynamic" means that sensitive layers (attention, output projections) are kept at higher precision (6-8 bit) while the bulk of the feed-forward layers are compressed to 4-bit. This is a per-layer decision, not a uniform quantization across the whole model. Everything runs on a single g5.xlarge instance in AWS eu-west-1, and gets torn down at the end.

**Outline**

- [Prerequisites](#1-prerequisites)
- [Security Group](#2-security-group)
- [Find the Right AMI](#3-find-the-right-ami)
- [Launch the EC2 Instance](#4-launch-the-ec2-instance)
- [Connect via SSH](#5-connect-via-ssh)
- [Verify the GPU](#6-remote-verify-the-gpu)
- [Build llama.cpp with CUDA](#7-remote-build-llamacpp-with-cuda)
- [Download the Model](#8-remote-download-the-model)
- [Start llama-server](#9-remote-start-llama-server)
- [Connect OpenCode to the Remote Server](#10-connect-opencode-to-the-remote-server)
- [Validate with ToolCall-15](#11-remote-validate-with-toolcall-15)
- [Using OpenCode with Qwen3.5-27B](#12-using-opencode-with-qwen35-27b)
- [Stop, Restart, and Teardown](#13-stop-restart-and-teardown)
- [Cost](#14-cost)
- [Reflections](#15-reflections)

## 1. Prerequisites

On the local machine (Ubuntu Linux), we need:

- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), configured with credentials that can create EC2 instances
- [OpenCode](https://opencode.ai), the coding agent frontend (installation covered in section 10)
- `curl` and `ssh`, standard on any Linux distribution

Everything else (llama.cpp, the model, Node.js) runs on the remote EC2 instance.

**1.** Verify AWS credentials:

```bash
aws sts get-caller-identity --region eu-west-1
```

```json
{
    "UserId": "AIDAEXAMPLEUSERID",
    "Account": "111122223333",
    "Arn": "arn:aws:iam::111122223333:user/myuser"
}
```

**2.** Confirm g5.xlarge is available in eu-west-1:

```bash
aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters Name=instance-type,Values=g5.xlarge \
  --region eu-west-1
```

It is available in all three availability zones (eu-west-1a, 1b, 1c).

### What is a g5.xlarge?

A [g5.xlarge](https://aws.amazon.com/ec2/instance-types/g5/) has one NVIDIA A10G GPU with 24 GB of VRAM, 4 vCPUs, and 16 GB of system RAM. The A10G supports CUDA compute capability 8.6. The 24 GB of VRAM is just enough to hold a 4-bit quantized 27B-parameter model plus a compressed KV cache for a 64K context window.

We also have an existing EC2 key pair (`my-key-pair`) already registered in eu-west-1, with the private key at `~/keys/my-key-pair.pem`.

## 2. Security Group

A security group is a stateful firewall for EC2 instances. We create a dedicated one so it can be cleanly deleted during teardown.

**1.** Get our current public IP address:

```bash
MY_IP=$(curl -4 -s ifconfig.me)
echo $MY_IP
# 203.0.113.42
```

**2.** Create the security group in the default VPC:

```bash
SG_ID=$(aws ec2 create-security-group \
  --group-name llm-exploration-sg \
  --description "Temporary SG for LLM exploration - SSH only" \
  --vpc-id vpc-0example \
  --region eu-west-1 \
  --query "GroupId" \
  --output text)
echo $SG_ID
# sg-0example1234abcde
```

**3.** Allow SSH (port 22) inbound from our IP only:

```bash
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr ${MY_IP}/32 \
  --region eu-west-1
```

The `/32` mask restricts acces to our single IP. `0.0.0.0/0` would expose SSH to the entire internet. All outbound traffic is allowed by default.

## 3. Find the Right AMI

We use Amazon's Deep Learning Base AMI rather than a plain Ubuntu image. It ships with NVIDIA drivers, CUDA toolkit, and cuDNN pre-installed.

```bash
AMI_ID=$(aws ec2 describe-images \
  --region eu-west-1 \
  --owners amazon \
  --filters \
    "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
    "Name=state,Values=available" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" \
  --output text)
echo $AMI_ID
# ami-0a54beb9cdef7558b
```

This resolves to `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260327`, released on 2026-03-27.

## 4. Launch the EC2 Instance

**1.** Launch the instance:

```bash
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type g5.xlarge \
  --key-name my-key-pair \
  --security-group-ids $SG_ID \
  --block-device-mappings '[{
    "DeviceName": "/dev/sda1",
    "Ebs": {
      "VolumeSize": 80,
      "VolumeType": "gp3",
      "DeleteOnTermination": true
    }
  }]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=llm-exploration}]' \
  --region eu-west-1 \
  --query "Instances[0].InstanceId" \
  --output text)
echo $INSTANCE_ID
# i-0example1234abcde
```

We use on-demand pricing here. Spot instances are cheaper (see section 14) but can be interrupted at any time, which would kill the server mid-experiment.

The 80 GB [gp3](https://docs.aws.amazon.com/ebs/latest/userguide/general-purpose.html) root volume is the minimum this AMI requires. It leaves enough room for the OS (~20 GB), the model weights (~17 GB), and the llama.cpp build. `DeleteOnTermination: true` means the volume goes away with the instance.

**2.** Wait for the instance to reach `running` state:

```bash
aws ec2 wait instance-running \
  --instance-ids $INSTANCE_ID \
  --region eu-west-1
```

**3.** Retrieve the public IP:

```bash
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --region eu-west-1 \
  --query "Reservations[0].Instances[0].PublicIpAddress" \
  --output text)
echo $PUBLIC_IP
# 198.51.100.10
```

## 5. Connect via SSH

```bash
ssh -i ~/keys/my-key-pair.pem ubuntu@$PUBLIC_IP
```

All commands from this point until the teardown section run **on the remote EC2 instance**, not on the local machine. Each remote code block is marked with `[REMOTE]` in its section title.

## 6. [REMOTE] Verify the GPU

```bash
nvidia-smi
```

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA A10G                    On  |   00000000:00:1E.0 Off |                    0 |
|  0%   21C    P8             11W /  300W |       0MiB /  23028MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

The A10G is idle at 11W with 0% utilization and no VRAM allocated.

## 7. [REMOTE] Build llama.cpp with CUDA

**1.** Install build dependencies:

```bash
sudo apt-get update && sudo apt-get install -y \
  pciutils build-essential cmake curl libcurl4-openssl-dev git tmux
```

**2.** Clone llama.cpp:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

**3.** Configure and build with CUDA support:

```bash
cmake -B build \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build --config Release -j4 \
  --clean-first \
  --target llama-server llama-cli llama-gguf-split
```

`-DGGML_CUDA=ON` enables GPU acceleration. Without it, llama.cpp runs on CPU only, which is far too slow for a 27B model. `-j4` matches the instance's 4 vCPUs.

`-DCMAKE_CUDA_ARCHITECTURES=86` tells the compiler to target only the A10G's compute capability (8.6). Without this flag, llama.cpp compiles CUDA kernels for 9 different GPU architectures (compute 50 through 121a), which takes over an hour on 4 vCPUs. Targeting just our GPU reduces this to minutes.

**4.** Copy the binaries to the home directory for convenience:

```bash
cp build/bin/llama-server build/bin/llama-cli build/bin/llama-gguf-split ~
```

## 8. [REMOTE] Download the Model

**1.** Install the [Hugging Face Hub](https://github.com/huggingface/huggingface_hub) client and the [hf_transfer](https://github.com/huggingface/hf_transfer) library for faster downloads:

```bash
pip install huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

`HF_HUB_ENABLE_HF_TRANSFER=1` switches to a Rust-based parallel download backend that saturates the network link. Without it, downloads go through a single Python thread.

**2.** Download only the files we need from the [unsloth/Qwen3.5-27B-GGUF](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) repository:

```bash
huggingface-cli download unsloth/Qwen3.5-27B-GGUF \
  --local-dir ~/models/Qwen3.5-27B-GGUF \
  --include "*UD-Q4_K_XL*"
```

This downloads `Qwen3.5-27B-UD-Q4_K_XL.gguf` (17 GB): the model weights in Unsloth Dynamic 4-bit quantization. UD-Q4_K_XL keeps important layers (attention, output) at higher precision (8/16-bit) while compressing feed-forward layers to 4-bit. This preserves more model quality than uniform Q4 quantization. According to [Unsloth's benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks), UD-Q4_K_XL has the lowest KL divergence among all 4-bit variants (mean KLD of 0.0137, 99.9th-percentile KLD of 0.41). For reference, Q8_K_XL (36 GB, near-lossless) scores 0.10 at the 99.9th percentile, so the quality gap is modest for roughly half the size.

The repository also contains `mmproj-F16.gguf` (885 MB), the vision projection weights for Qwen3.5's multimodal architecture. We skip it here since we only need text. To enable image input, download it with `--include "*mmproj-F16*"` and pass `--mmproj` to llama-server.

## 9. [REMOTE] Start llama-server

The g5.xlarge has only 16 GB of system RAM. Loading a 17 GB model briefly exceeds that (tensors are staged in RAM before being copied to GPU), so we add a swap file to avoid an OOM kill.

**1.** Create a 4 GB swap file:

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**2.** Start llama-server in a [tmux](https://github.com/tmux/tmux) session so it persists if the SSH connection drops:

```bash
tmux new-session -d -s llmserver '
~/llama-server \
  --model ~/models/Qwen3.5-27B-GGUF/Qwen3.5-27B-UD-Q4_K_XL.gguf \
  --alias "unsloth/Qwen3.5-27B" \
  -ngl 99 \
  --ctx-size 32768 \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.00 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --host 0.0.0.0 \
  --port 8080 2>&1 | tee ~/llmserver.log
'
```

Flags explained:

- `-ngl 99`: offload all 65 layers to GPU. Layers left on CPU would be far slower.
- `--ctx-size 32768`: 32K token context window. With ~5 GB of VRAM headroom and q4_0 KV cache compression, 32K fits comfortably. 64K would eat most of the remaining VRAM.
- `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.00`: [Qwen3.5 recommended sampling parameters](https://huggingface.co/Qwen/Qwen3.5-27B) for *coding tasks in thinking mode*:

<p align="center">
  <img src="/img/2026-04-11_01/qwen35_coding_tasks.png" alt="Qwen3.5 recommended sampling parameters for coding tasks" width="600" /><br>
  <b>Qwen3.5 recommended sampling parameters</b>
</p>

- `--cache-type-k q4_0 --cache-type-v q4_0`: 4-bit KV cache compression, ~4x less memory than FP16.
- `--host 0.0.0.0`: listen on all interfaces. Only `127.0.0.1` is needed for the SSH tunnel, but port 8080 is blocked by the security group anyway.

**3.** Watch the logs and wait for the server to be ready:

```bash
tmux attach -t llmserver   # Ctrl+B then D to detach
```

Loading takes several minutes: 17 GB of weights read from disk and copied to GPU through 16 GB of system RAM. Once done, the log prints:

```
main: model loaded
main: server is listening on http://0.0.0.0:8080
srv  update_slots: all slots are idle
```

With all layers on GPU, the server uses about 18.1 GB of VRAM, leaving ~5 GB headroom for the KV cache. Note that this configuration has a single processing slot: only one request is served at a time. A second concurrent request would queue behind the first. Coding agents that dispatch parallel subagents (as Claude Code does) cannot do so here; each call is serialized.

**4.** Verify the server is healthy:

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

**5.** Test with a simple completion:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.5-27B",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 4096
  }'
```

The llama-server request log reports about **24 tokens/second** generation (decode) speed on the A10G. Prompt evaluation (prefill) is much faster, typically 100+ tok/s. Decode is the bottleneck you feel: that's how fast the model writes its answer. Qwen3.5 uses a "thinking" mode by default, so the response includes both `reasoning_content` (internal chain of thought) and `content` (the visible answer). Most tokens go to thinking. A simple "Say hello" prompt burns ~1200 tokens of reasoning before producing a one-line answer. `max_tokens` covers both, so set it generously.

Thinking can be disabled two ways. `/no_think` at the start of the user prompt is a soft toggle: the model is asked not to think, but the template is still there, so it may think anyway. Passing `"chat_template": "chatml"` in the API request is harder: it strips the thinking tokens from the template entirely. Both make responses faster. For well-defined coding tasks where the instructions are already precise, disabling thinking is a reasonable trade-off.

## 10. Connect OpenCode to the Remote Server

[OpenCode](https://opencode.ai) is an open-source terminal-based coding agent, similar to Claude Code. It connects to any OpenAI-compatible endpoint. We run it on the local machine and point it at the remote llama-server through an SSH tunnel.

**1.** Start an SSH tunnel for port 8080 (run locally):

```bash
ssh -i ~/keys/my-key-pair.pem \
  -L 8080:localhost:8080 \
  ubuntu@$PUBLIC_IP \
  -N &
```

This forwards local port 8080 to the remote llama-server. `-N` tells SSH not to open a shell, it just holds the tunnel open.

**2.** Verify the tunnel works:

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

**3.** Install OpenCode if needed:

```bash
curl -fsSL https://opencode.ai/install | bash
```

**4.** Create the configuration file:

```bash
mkdir -p ~/.config/opencode
cat > ~/.config/opencode/config.json << 'EOF'
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
        }
      }
    }
  },
  "model": "local-llama/qwen3.5-27b"
}
EOF
```

This defines a custom provider called `local-llama` using the `@ai-sdk/openai-compatible` package. `baseURL` points to `localhost:8080`, which the SSH tunnel forwards to the remote llama-server. `apiKey` is required by the schema but llama-server ignores it, so any string works. If you get errors about context length being exceeded, add `"max_tokens": 20000` to the model config to cap output tokens and leave headroom for the system prompt and conversation history.

**5.** Run OpenCode from any project directory:

```bash
cd ~/your-project
opencode
```

OpenCode sends requests with `"stream": true`, so tokens appear in the terminal as they are generated (via server-sent events). The llama-server supports this natively. The curl test in section 9 uses non-streaming mode for simplicity, but in normal use you always get streaming.

## 11. [REMOTE] Validate with ToolCall-15

[ToolCall-15](https://github.com/stevibe/ToolCall-15) is a benchmark for LLM tool-calling quality: 15 tests across 5 categories. This matters for coding agents, which live or die by their ability to pick the right tool and fill in the right parameters.

**1.** In a new tmux window, install Node.js and set up the benchmark:

```bash
tmux new-window -t llmserver

# install Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# clone and set up ToolCall-15
git clone https://github.com/stevibe/ToolCall-15
cd ToolCall-15
npm install
```

**2.** Configure ToolCall-15 to use the local llama-server:

```bash
cat > .env << 'EOF'
LLAMACPP_HOST=http://localhost:8080
LLM_MODELS=llamacpp:unsloth/Qwen3.5-27B
EOF
```

**3.** Start the ToolCall-15 dashboard:

```bash
npm run dev
```

```
✓ Ready in 634ms
- Local:    http://localhost:3000
```

**4.** Access the dashboard from your **local machine** through another SSH tunnel:

```bash
# Run this on your LOCAL machine (not the EC2 instance)
ssh -i ~/keys/my-key-pair.pem \
  -L 3000:localhost:3000 \
  ubuntu@$PUBLIC_IP \
  -N &
```

Open `http://localhost:3000` in a browser and click "Run Benchmark".

The 5 categories are: tool selection, parameter precision, multi-step chains, restraint/refusal (knowing when *not* to call a tool), and error recovery.

### Results

Qwen3.5-27B scored **29/30 points (97%)** via llama.cpp, with the following breakdown by category:

| Category | Score | Tests |
|----------|-------|-------|
| A - Tool Selection | 6/6 | TC-01, TC-02, TC-03 |
| B - Parameter Precision | 6/6 | TC-04, TC-05, TC-06 |
| C - Multi-Step Chains | 6/6 | TC-07, TC-08, TC-09 |
| D - Restraint and Refusal | 5/6 | TC-10, TC-11 |
| E - Error Recovery | 6/6 | TC-12, TC-13, TC-14, TC-15 |

The only miss is TC-11 (Simple Math): the model reached for the calculator tool to compute 15% of 200 instead of just answering. Correct result, unnecessary tool call.

<p align="center">
  <img src="/img/2026-04-11_01/toolcall-15_02.png" alt="ToolCall-15 results for Qwen3.5-27B" width="600" /><br>
  <b>ToolCall-15 results for Qwen3.5-27B (29/30)</b>
</p>

## 12. Using OpenCode with Qwen3.5-27B

<p align="center">
  <img src="/img/2026-04-11_01/open_code_01.png" alt="OpenCode v1.3.17" width="600" /><br>
  <b>OpenCode v1.3.17</b>
</p>

We use OpenCode v1.3.17. With the SSH tunnel from section 10 active, run `opencode` from any project directory.

As a test task, we asked OpenCode to implement an optimized [Jaro-Winkler similarity](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance) in Python:

> Implement an optimized Jaro-Winkler similarity in Python.
>
> Requirements:
> - Time complexity better than naive O(n²)
> - No external libraries, besides Cython, NumPy, Pandas
> - Include unit tests
> - Handle edge cases (empty strings, identical strings, unicode)
> - Benchmark against a naive version
> - Vectorize for batch comparison
> - Make it NumPy-friendly
> - Avoid Python lists for matches (use arrays / bitsets)

<p align="center">
  <img src="/img/2026-04-11_01/open_code_02.png" alt="OpenCode session with Qwen3.5-27B" width="900" /><br>
  <b>OpenCode session, Jaro-Winkler implementation task</b>
</p>

The model produced 981 lines of source code across 12 files. All 22 unit tests pass. A proper code quality evaluation is beyond the scope of this post, but the harder part, keeping coherence across 12 files spanning algorithm code, tests, benchmarks, packaging, and Cython integration, worked.

## 13. Stop, Restart, and Teardown

Back on the **local machine**.

### Stopping the instance (pause billing)

Stopping keeps the EBS volume intact: the model, llama.cpp build, and all configs survive. The public IP will change on restart.

```bash
aws ec2 stop-instances \
  --instance-ids $INSTANCE_ID \
  --region eu-west-1
```

### Restarting a stopped instance

```bash
aws ec2 start-instances \
  --instance-ids $INSTANCE_ID \
  --region eu-west-1

aws ec2 wait instance-running \
  --instance-ids $INSTANCE_ID \
  --region eu-west-1

# The public IP changes after stop/start, fetch the new one
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --region eu-west-1 \
  --query "Reservations[0].Instances[0].PublicIpAddress" \
  --output text)
echo $PUBLIC_IP
```

No [Elastic IP](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/elastic-ip-addresses-eip.html), so the instance gets a new public IP on every start. Your own IP may also have changed. If SSH times out after restart, update the security group:

```bash
MY_IP=$(curl -4 -s ifconfig.me)
aws ec2 revoke-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr <old-ip>/32 \
  --region eu-west-1
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr ${MY_IP}/32 \
  --region eu-west-1
```

After reconnecting via SSH, restart the llama-server tmux session:

```bash
ssh -i ~/keys/my-key-pair.pem ubuntu@$PUBLIC_IP

# on the remote instance:
sudo swapon /swapfile
tmux new-session -d -s llmserver '
~/llama-server \
  --model ~/models/Qwen3.5-27B-GGUF/Qwen3.5-27B-UD-Q4_K_XL.gguf \
  --alias "unsloth/Qwen3.5-27B" \
  -ngl 99 \
  --ctx-size 32768 \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.00 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --host 0.0.0.0 \
  --port 8080 2>&1 | tee ~/llmserver.log
'
```

Then re-establish the SSH tunnel locally with the new IP:

```bash
ssh -i ~/keys/my-key-pair.pem -L 8080:localhost:8080 ubuntu@$PUBLIC_IP -N &
opencode
```

### Full teardown

Tear down in order: instance first, then security group. AWS will not delete a security group still attached to an instance.

**1.** Terminate the EC2 instance. The root EBS volume is automatically deleted because we set `DeleteOnTermination: true` when launching:

```bash
aws ec2 terminate-instances \
  --instance-ids $INSTANCE_ID \
  --region eu-west-1
```

**2.** Wait for termination to complete:

```bash
aws ec2 wait instance-terminated \
  --instance-ids $INSTANCE_ID \
  --region eu-west-1
echo "Instance terminated."
```

**3.** Delete the security group. This must wait for the instance to fully terminate, as AWS rejects the call if the group is still in use:

```bash
aws ec2 delete-security-group \
  --group-id $SG_ID \
  --region eu-west-1
echo "Security group deleted."
```

**4.** Kill the SSH tunnel if still running:

```bash
pkill -f "ssh.*my-key-pair"
echo "SSH tunnel killed."
```

**5.** Verify nothing is left running:

```bash
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=llm-exploration" \
            "Name=instance-state-name,Values=running,stopped,pending" \
  --region eu-west-1 \
  --query "Reservations[].Instances[].InstanceId" \
  --output text
```

If this returns nothing, all compute resources are gone and billing has stopped.

## 14. Cost

All prices below are for reference only, collected on 2026-04-11 for eu-west-1 (Ireland).

### Running costs

| Component | On-demand | Spot |
|-----------|-----------|------|
| g5.xlarge compute | \$1.12/hour | ~\$0.59/hour |
| 80 GB gp3 EBS volume | \$0.0096/hour | \$0.0096/hour |
| **Total** | **~\$1.13/hour** | **~\$0.60/hour** |

The EBS volume adds less than a cent per hour. Compute dominates.

### Idle costs

When the instance is **stopped**, compute billing stops but the EBS volume persists: 80 GB × \$0.088/GB/month = **\$7.04/month** (~\$0.23/day).

A cheaper alternative: terminate the instance and keep only an **AMI snapshot**. Snapshot storage is \$0.05/GB/month, so 80 GB = **\$4/month**. Launching a new spot instance from that snapshot takes a couple of minutes, with no need to rebuild llama.cpp or re-download the model.

### Spot instances

Spot prices for g5.xlarge in eu-west-1 have been stable over the past week, between \$0.58 and \$0.60, roughly half on-demand. Interruption risk exists but is low at these price levels. And since the setup can be baked into a custom AMI, a reclaimed spot instance can be replaced in minutes.

### Comparison with API providers

A full working day on a spot instance costs around \$5. For that budget, you can make hundreds of API calls to Claude Sonnet or GPT-4o, both of which are considerably stronger than a 27B quantized model (longer context, better reasoning, better instruction following). But the things self-hosting buys you are data privacy and no rate limits.

### Other GPU providers

AWS is not the cheapest option here. [Lambda Labs](https://lambdalabs.com/), [Vast.ai](https://vast.ai/), and [RunPod](https://www.runpod.io/) offer A10G-class GPUs at a fraction of the EC2 on-demand price.

## 15. Reflections

A 27B model quantized to 4-bit runs on a single A10G, which Nvidia launched in 2021. In daily use, the experience is a bit slower than Claude Code. The 32K context fills up fast, and OpenCode has to compact the conversation regularly to stay within the window, which adds pauses.

Still, running your own coding agent at this quality level on a single GPU is something. 27B models existed a year ago (Qwen2.5, Llama 3.1), but the combination has improved on several fronts since then: better base model quality, smarter quantization schemes, better cache compression, and more mature tooling (llama.cpp, OpenCode, GGUF ecosystem). This setup makes sense today for trying out new models, benchmarking quantization, or **working on a codebase you do not want to send to a third-party API**.
