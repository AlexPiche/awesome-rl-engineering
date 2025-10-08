# 🧠 Awesome RL Engineering

> A curated list of resources exploring **reinforcement learning training systems**, GPU kernel design, asynchronous compute, and large-scale infrastructure insights.

---

## 📚 Contents
- [🧠 Awesome RL Engineering](#-awesome-rl-engineering)
  - [📚 Contents](#-contents)
  - [🤖 LLM Inference](#-llm-inference)
  - [🧮 Low-Level Kernels \& GPU Compute](#-low-level-kernels--gpu-compute)
  - [⚙️ Asynchronous \& System Design](#️-asynchronous--system-design)
  - [Deep dive into Python's async/await implementation and its implications for ML workloads.](#deep-dive-into-pythons-asyncawait-implementation-and-its-implications-for-ml-workloads)
  - [🧱 RL Training \& Infrastructure](#-rl-training--infrastructure)
  - [🧩 Expert Parallelism \& Quantization](#-expert-parallelism--quantization)
  - [🎓 Courses \& Learning Resources](#-courses--learning-resources)
  - [🧰 Tools \& Glossaries](#-tools--glossaries)
  - [📜 License](#-license)

---

## 🤖 LLM Inference
- [LLM Inference: Continuous Batching and PagedAttention](https://insujang.github.io/2024-01-07/llm-inference-continuous-batching-and-pagedattention/)  
  Exploration of continuous batching and PagedAttention for efficient LLM inference.
- [A Deep Dive into LLM Inference Latencies](https://blog.hathora.dev/a-deep-dive-into-llm-inference-latencies/)
    Analysis of latency components in LLM inference and strategies to optimize them.

## 🧮 Low-Level Kernels & GPU Compute

- [Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels](https://www.aleksagordic.com/blog/matmul)  
  Deep dive into matmul kernel structure, memory hierarchy, and tensor core optimization.

- [JAX Pallas: Blackwell Matmul](https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html)  
  Official documentation on writing efficient matmul kernels for Blackwell GPUs using Pallas.

- [How to Scale Your Model (JAX-ML Scaling Book)](https://jax-ml.github.io/scaling-book/)  
  A systems-oriented guide to scaling models: rooflines, sharding, parallelism, profiling, and interconnects.

---

## ⚙️ Asynchronous & System Design

- [Async Compute All the Things](https://interplayoflight.wordpress.com/2025/05/27/async-compute-all-the-things/)  
  Exploration of asynchronous compute paradigms and overlapping compute/data movement in GPU workloads.
 - [How async/await works in Python](https://tenthousandmeters.com/blog/python-behind-the-scenes-12-how-asyncawait-works-in-python/)  
  Deep dive into Python's async/await implementation and its implications for ML workloads.
---

## 🧱 RL Training & Infrastructure
 - [PipelineRL — ServiceNow](https://huggingface.co/blog/ServiceNow/pipelinerl)  
  A system-level view of pipeline-based reinforcement learning training.

- [Flash RL (Notion Page)](https://fengyao.notion.site/flash-rl)  
  Notes on applying FlashAttention-style compute optimizations to reinforcement learning training loops.

  Investigation into the training-inference mismatch in LLM-RL, showing how high-speed inference can destabilize training, especially for low-probability tokens and OOD contexts. 

- [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl)
     Explains how modern RL frameworks for LLMs often introduce implicit off-policy training due to differences between inference and training policies. Highlights the importance of understanding policy mismatch and the role of importance sampling in correcting for it. Useful for diagnosing silent instability in RL training pipelines.

- [Group Sequence Policy Optimization (GSPO)](https://arxiv.org/abs/2507.18071)
     Paper introducing GSPO, a stable and efficient RL algorithm for training large language models. GSPO uses sequence-level importance ratios and clipping, outperforming previous token-level approaches (like GRPO), and stabilizes Mixture-of-Experts (MoE) RL training. Demonstrates improved efficiency and performance in Qwen3 models. 
---

## 🧩 Expert Parallelism & Quantization

- [DeepEP](https://github.com/deepseek-ai/DeepEP)  
  Expert-parallel communication library for large-scale MoE training.

- [LLMQ](https://github.com/IST-DASLab/llmq/)  
  Quantized large model training implemented in CUDA/C++, focusing on compute and memory efficiency.

---

## 🎓 Courses & Learning Resources

- [Stanford CS336 — Language Modeling from Scratch (Spring 2025)](https://stanford-cs336.github.io/spring2025/)  
  Advanced course covering model training, system optimizations, and scaling architectures.

- [The MLAI Engineer’s Starter Guide](https://multimodalai.substack.com/p/the-mlai-engineers-starter-guide)  
  A practical guide (on MultimodalAI Substack) for aspiring ML/AI engineers on tooling, infrastructure, and workflows.

- [Maximizing GPU Efficiency: The Battle](https://bytesofintelligence.substack.com/p/maximizing-gpu-efficiency-the-battle)  
  A deep commentary on GPU efficiency challenges in modern ML workloads.

---

## 🧰 Tools & Glossaries

- [GPU Glossary — Modal](https://modal.com/gpu-glossary)  
  Compact glossary explaining GPU terms, architectures, and performance metrics.

---

## 📜 License

[MIT License](LICENSE)
