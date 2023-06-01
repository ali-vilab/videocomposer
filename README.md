# VideoComposer

Official repo for [VideoComposer: Compositional Video Synthesiswith Motion Controllability]()

Please see [Project Page](https://videocomposer.github.io/) for more examples.

<div align=center>
<img src="./source/fig01.jpg" alt="" width="100%">
</div>

VideoComposer is a controllable video diffusion model, which allows users to flexibly control the spatial and temporal patterns simultaneously within a synthesized video in various forms, such as text description, sketch sequence, reference video, or even simply handcrafted motions and handrawings.



## TODO
- [x] Release our technical papers and webpage.
- [ ] Release training and inference code.
- [ ] Release pretrained model that without watermark.
- [ ] Release pretrained model that can generate 8s videos.
- [ ] Release Gradio UI on ModelScop and Hugging Face.


## Method


![method](source/fig02_framwork.jpg "method")


## Generated Videos


<video id="video" controls="" preload="none" poster="封面">
      <source id="mp4" src="https://cloud.video.taobao.com/play/u/null/p/1/e/6/t/1/412500082019.mp4?SBizCode=xiaoer" type="video/mp4">
</videos>



## BibTeX

```bibtex
@article{2023videocomposer,
  title={VideoComposer: Compositional Video Synthesis with Motion Controllability},
  author={Wang, Xiang* and Yuan, Hangjie* and Zhang, Shiwei* and Chen, Dayou* and Wang, Jiuniu, and Zhang, Yingya, and Shen, Yujun, and Zhao, Deli and Zhou, Jingren},
  booktitle={arXiv preprint arxiv},
  year={2023}
}
```