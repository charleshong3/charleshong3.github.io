---
layout: blog
title: "SALTyRN featured on the RISE Project blog"
order: 9
---

# SALTyRN featured on the RISE Project blog

#### July 27, 2026

[Charles Hong](https://charleshong3.github.io/){:target="_blank" rel="noopener"} (UC Berkeley)

### Link to the blog post: [https://riseproject.dev/2026/07/27/saltyrn-turning-neon-kernels-into-fast-verified-rvv-code-with-llms/](https://riseproject.dev/2026/07/27/saltyrn-turning-neon-kernels-into-fast-verified-rvv-code-with-llms/){:target="_blank" rel="noopener"}

We wrote about **SALTyRN** for the [RISE Project blog](https://riseproject.dev/){:target="_blank" rel="noopener"}. SALTyRN uses LLMs to translate Arm Neon SIMD kernels into RISC-V Vector (RVV) code, then searches for faster hardware-tuned variants — verifying correctness at both stages via compilation, execution on Spike, and bounded symbolic equivalence checking against the original Neon kernel. The optimization phase is driven by [Autocomp](https://github.com/ucb-bar/autocomp){:target="_blank" rel="noopener"}, which we extended with an RVV backend targeting Saturn.

> The translation from Neon to RVV is easy to get only half right. [...] Preserving the computation does not automatically mean using the target architecture effectively.

Across 35 [XNNPACK](https://github.com/google/XNNPACK){:target="_blank" rel="noopener"} microkernels, our optimized translations beat expert-written RVV code by **1.40x** geometric mean (versus 0.78x for unoptimized translations), with performance measured on a [Saturn](https://github.com/ucb-bar/saturn-vectors){:target="_blank" rel="noopener"} vector unit via FireSim. Three of our contributions have been merged upstream into XNNPACK.

Work by Keagan Chern, Charles Hong, Sophia Shao, and Alvin Cheung. Thanks to RISE for the feature and for supporting this work through a Gemini compute credit grant!
