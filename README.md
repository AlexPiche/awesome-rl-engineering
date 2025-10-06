# 🧠 Awesome RL Training System

> A curated list of resources exploring **reinforcement learning training systems**, GPU kernel design, asynchronous compute, and large-scale infrastructure insights.

---

## 📚 Contents
- [🧠 Awesome RL Training System](#-awesome-rl-training-system)
  - [📚 Contents](#-contents)
  - [🧮 Low-Level Kernels \& GPU Compute](#-low-level-kernels--gpu-compute)
  - [⚙️ Asynchronous \& System Design](#️-asynchronous--system-design)
  - [🧱 RL Training \& Infrastructure](#-rl-training--infrastructure)
  - [🧩 Expert Parallelism \& Quantization](#-expert-parallelism--quantization)
  - [🎓 Courses \& Learning Resources](#-courses--learning-resources)
  - [🧰 Tools \& Glossaries](#-tools--glossaries)
  - [📜 License](#-license)

---

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

---

## 🧱 RL Training & Infrastructure

- [PipelineRL — ServiceNow](https://huggingface.co/blog/ServiceNow/pipelinerl)  
  A system-level view of pipeline-based reinforcement learning training.

- [Flash RL (Notion Page)](https://fengyao.notion.site/flash-rl)  
  Notes on applying FlashAttention-style compute optimizations to reinforcement learning training loops.

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
