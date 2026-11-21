"""Export the fine-tuned D-FINE to ONNX at 640x640 (static batch 1).
Outputs: logits (1, 300, C) pre-sigmoid, pred_boxes (1, 300, 4) normalised cxcywh.

  .venv/bin/python train/nav/export_onnx.py --ckpt train/nav/runs/dfine_s_oi18 \
      --out data/nav/dfine_s_oi18_640.onnx
"""

import argparse

import torch
from transformers import DFineForObjectDetection


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        o = self.m(pixel_values=pixel_values)
        return o.logits, o.pred_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = DFineForObjectDetection.from_pretrained(a.ckpt).eval()
    torch.onnx.export(Wrap(m), (torch.rand(1, 3, 640, 640),), a.out, opset_version=17,
                      input_names=["pixel_values"], output_names=["logits", "pred_boxes"],
                      dynamo=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
