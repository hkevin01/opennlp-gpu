# FAQ

**Q: What is OpenNLP GPU?**
A: An extension to Apache OpenNLP for GPU-accelerated NLP tasks.

**Q: Which GPUs are supported?**
A: NVIDIA (CUDA), AMD (ROCm), Intel (OpenCL).

**Q: How do I enable GPU acceleration?**
A: Set `GpuConfig.setGpuEnabled(true)` and ensure drivers are installed.

**Q: What if no GPU is detected?**
A: The library falls back to CPU automatically.

**Q: How do I run the tests?**
A: Use `mvn test` from the project root.

**Q: Where can I get help?**
A: Open an issue on GitHub or see the user guide. 