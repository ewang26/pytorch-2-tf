# pytorch-2-tf
This library is a pipeline designed to convert pretrained PyTorch image models to Tensorflow format for deployment.

## Introduction

I'm someone who loves PyTorch for its flexibility — it's really great for churning out machine learning models quickly that perform reasonably well.

However, I'm also someone who works a lot with deploying machine learning models to mobile devices. PyTorch, unfortunately, is still quite lacking in that respect.
TensorFlow is simply more flexible when it comes to deployment, with support across a variety of platforms from mobile phones to simpler embedded devices.

Thus, it's useful to be able to convert trained PyTorch model checkpoints to TensorFlow format so one can quickly deploy ML models into a real-world setting. However, current intermediary formats like ONNX often result in accuracy loss when converting between formats, which is unacceptable for some applications. This library aims to provide an easy way for developers to quickly deploy trained PyTorch models using the power of TensorFlow's support for various deployment platforms.

## Dependencies

Pytorch-2-tf uses `argparse`, which is a parser for command-line options, arguments, and sub-commands; `json`, a data-interchange format; `tqdm`, a progress bar for Python; `torch`, an open source machine learning library; and `numpy`, a library that supports large, multi-dimensional arrays and matrices.

## Contributing

See `CONTRIBUTING.md`.
