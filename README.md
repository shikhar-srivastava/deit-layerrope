# Data-Efficient architectures and training for Image classification

> [!NOTE]
> This fork adds **LayerRoPE** to DeiT, for the ViT-Tiny and ViT-Small experiments of *LayerRoPE: Dynamic Depth-wise Magnitude & Angular Superposition* ([main repository](https://github.com/shikhar-srivastava/layer-rope)). See [LayerRoPE](#layerrope) below; everything else is upstream DeiT at [`7e160fe`](https://github.com/facebookresearch/deit/tree/7e160fe).

This repository contains PyTorch evaluation code, training code and pretrained models for the following papers:
<details>
<summary>
  <a href="README_deit.md">DeiT</a> Data-Efficient Image Transformers, ICML 2021 [<b>bib</b>]
</summary>

```
@InProceedings{pmlr-v139-touvron21a,
  title =     {Training data-efficient image transformers &amp; distillation through attention},
  author =    {Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and Jegou, Herve},
  booktitle = {International Conference on Machine Learning},
  pages =     {10347--10357},
  year =      {2021},
  volume =    {139},
  month =     {July}
}
```
</details>
<details>
<summary>
<a href="README_cait.md">CaiT</a> (Going deeper with Image Transformers), ICCV 2021  [<b>bib</b>]
</summary>

```
@InProceedings{Touvron_2021_ICCV,
    author    = {Touvron, Hugo and Cord, Matthieu and Sablayrolles, Alexandre and Synnaeve, Gabriel and J\'egou, Herv\'e},
    title     = {Going Deeper With Image Transformers},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {32-42}
}
```
</details>
<details>
<summary>
<a href="README_resmlp.md">ResMLP</a> (ResMLP: Feedforward networks for image classification with data-efficient training), TPAMI 2022 [<b>bib</b>]
</summary>

```
@article{touvron2021resmlp,
  title={ResMLP: Feedforward networks for image classification with data-efficient training},
  author={Hugo Touvron and Piotr Bojanowski and Mathilde Caron and Matthieu Cord and Alaaeldin El-Nouby and Edouard Grave and Gautier Izacard and Armand Joulin and Gabriel Synnaeve and Jakob Verbeek and Herv'e J'egou},
  journal={arXiv preprint arXiv:2105.03404},
  year={2021},
}
```
</details>
<details>
<summary>
<a href="README_patchconvnet.md">PatchConvnet</a> (Augmenting Convolutional networks with attention-based aggregation) [<b>bib</b>]
</summary>

```
@article{touvron2021patchconvnet,
  title={Augmenting Convolutional networks with attention-based aggregation},
  author={Hugo Touvron and Matthieu Cord and Alaaeldin El-Nouby and Piotr Bojanowski and Armand Joulin and Gabriel Synnaeve and Jakob Verbeek and Herve Jegou},
  journal={arXiv preprint arXiv:2112.13692},
  year={2021},
}
```
</details>
<details>
<summary>
<a href="README_3things.md">3Things</a> (Three things everyone should know about Vision Transformers), ECCV 2022 [<b>bib</b>]
</summary>

```
@article{Touvron2022ThreeTE,
  title={Three things everyone should know about Vision Transformers},
  author={Hugo Touvron and Matthieu Cord and Alaaeldin El-Nouby and Jakob Verbeek and Herve Jegou},
  journal={arXiv preprint arXiv:2203.09795},
  year={2022},
}
```
</details>
<details>
<summary>
<a href="README_revenge.md">DeiT III</a> (DeiT III: Revenge of the ViT), ECCV 2022 [<b>bib</b>]
</summary>

```
@article{Touvron2022DeiTIR,
  title={DeiT III: Revenge of the ViT},
  author={Hugo Touvron and Matthieu Cord and Herve Jegou},
  journal={arXiv preprint arXiv:2204.07118},
  year={2022},
}
```
</details>
<details>
<summary>
<a href="README_cosub.md">Cosub</a> (Co-training 2L Submodels for Visual Recognition), CVPR 2023 [<b>bib</b>]
</summary>

```
@article{Touvron2022Cotraining2S,
  title={Co-training 2L Submodels for Visual Recognition},
  author={Hugo Touvron and Matthieu Cord and Maxime Oquab and Piotr Bojanowski and Jakob Verbeek and Herv'e J'egou},
  journal={arXiv preprint arXiv:2212.04884},
  year={2022},
}
```
</details>
If you find this repository useful, please consider giving a star ⭐ and cite the relevant papers. 

# LayerRoPE

LayerRoPE replaces the per-layer normalization gains with one shared gain γ per site (the input and output of the attention and MLP blocks), scaled and rotated by a depth-conditioned complex factor:

γ<sub>ℓ</sub> = γ ⊙<sub>c</sub> exp(r(ℓ) + iθ(ℓ, j)),  r(ℓ) = α + β log(ℓ+1),  θ(ℓ, j) = exp(α<sub>rot</sub> + β<sub>rot</sub> log(ℓ+1)) · b<sup>−2j/d</sup>

where ⊙<sub>c</sub> multiplies the pairs (γ<sub>2j</sub>, γ<sub>2j+1</sub>) as complex numbers. With `--layer-rope`, the two LayerNorms of every block take their gains from these shared gains (each keeps its centering and its own bias), and the attention and MLP outputs are scaled by them before the residual additions ([`layer_rope.py`](layer_rope.py)); the final LayerNorm is unchanged. The depth slopes start at β = 0 (magnitude) and β<sub>rot</sub> = −0.5 (rotation), with α = α<sub>rot</sub> = 0 and b = 100 (`--layer-rope-*` in [`main.py`](main.py)).

We trained ViT-Tiny and ViT-Small for 300 epochs with the DeiT recipe on 4 GPUs, with and without `--layer-rope`:

```bash
torchrun --nproc_per_node=4 main.py --model deit_small_patch16_224 --batch-size 256 --data-path /path/to/imagenet --output_dir /path/to/save --layer-rope
torchrun --nproc_per_node=4 main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path /path/to/imagenet --output_dir /path/to/save --layer-rope
```

Our table reports, for each of top-1, top-5 and validation loss, the best value over the 300 epochs in `log.txt`. As in our runs, timm 0.3.2 exempts only 1-D parameters and biases from weight decay, so LayerRoPE's 16 scalars are weight-decayed. We used PyTorch 2.5.1, torchvision 0.20.1 and timm 0.3.2; under PyTorch 2, timm 0.3.2 needs `from torch._six import container_abcs` in `timm/models/layers/helpers.py` replaced by `import collections.abc as container_abcs`.

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

# Contributing
We actively welcome your pull requests! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) for more info.
