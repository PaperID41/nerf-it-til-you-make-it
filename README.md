NeRF it 'til You Make it: Improving 3DGS with Volume Rendering Cues

Usage
1. Generate Point Cloud and Depths (nerfacc)

```python examples/pretrain_ngp.py --output "output-dir" --data_root "data-dir" --scene "scene-name"```

* It's recommended to use ```data-dir/scene-name/nerfacc_output``` as ```output-dir``` here.
2. Initialize and Supervise Model (gaussian_splatting)

```python train.py -s "data-dir" --eval```
