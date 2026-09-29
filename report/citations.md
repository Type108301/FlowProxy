# 必须引用（正式技术报告附录按国标展开）

1. Ribeiro, M. D., et al. DeepCFD: Efficient Steady-State Laminar Flow Approximation with Deep Convolutional Neural Networks. arXiv:2004.08826. 代码：https://github.com/mdribeiro/DeepCFD （MIT）
2. Li, Z., et al. Fourier Neural Operator for Parametric Partial Differential Equations. ICLR 2021. arXiv:2010.08895. 参考实现：https://github.com/neuraloperator/neuraloperator
3. Ronneberger, O., Fischer, P., Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.
4. OpenFOAM Foundation / ESI-OpenCFD. OpenFOAM User Guide.
5. Schäfer, M., Turek, S., et al. Benchmark computations of laminar flow around a cylinder. Flow Simulation with High-Performance Computers II, 1996.
6. 全球校园人工智能算法精英大赛组委会. 关于举办第八届 AIC 算法大赛算法主题赛（AI+力学）的通知（第一轮）. 全智赛组委会〔2026〕19 号.

本仓库 `src/flowproxy/models/unetex.py` 改编自 DeepCFD；`fno.py` 为 Li et al. 谱卷积思想的独立精简实现，不把官方权重改名充当原创。谱权重在实现里存成实数张量，是为了在 CUDA 混合精度下做 float32 FFT，算法仍是同一套谱卷积。
