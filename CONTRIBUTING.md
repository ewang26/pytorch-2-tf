# A Few Notes on Contributions

## Comment Format

Please use [Python docstring format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for module, function, class, or method definitions. 

tl;dr: Add the following as the first thing after a module, function, class, or method definition
```
def my_method(int a):
    '''
    Describe the functionality here:
    my_method: prints the input int a

    Args:
        Here, describe each parameter:
        a: the input integer to be printed

    Returns:
        Here, describe the return type:
        None
    '''
    print(a)
```

## List of TODOs

- Adding support for more PyTorch layers. **This doesn't actually require serious knowledge of Pytorch**, though it is a fairly manual task.  I'd greatly appreciate if someone helped me with this. See the next section for more details.

- README/comment improvements

- General comments/suggestions about codebase organization or anything else are also appreciated.

### Major TODO: Added support for more PyTorch layers

The function `model_arch_conversion` performs the task of extracting the layers, or basic components, of a PyTorch model and storing the information in a dictionary so we can reconstruct the model in TensorFlow later. So far, you can see that several layer types are supported:

```
if isinstance(module, nn.Conv2d):
    model_arch_list.append({
                'name': module.__class__.__name__,
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding,
            })
```

In this example, the conditional checks if the layer is of the `nn.Conv2d` type. We then want to extract the attributes of the layer:

- The name of the layer: you can simply get this by accessing the `module.__class__.__name__` attribute.
- The other attributes `in_channels`, `out_channels`, etc. are **layer-dependent**, meaning you'll have to visit the [PyTorch docs](https://pytorch.org/docs/master/nn.html) and append the layer's specific attributes to the list. You should end up with a dictionary like:

```
{
    'name': module.__class__.__name__,
    'in_channels': module.in_channels,
    'out_channels': module.out_channels,
    'kernel_size': module.kernel_size,
    'stride': module.stride,
    'padding': module.padding,
}
```

So in summary, you need to add a new `if` statement for each new layer that extracts the attributes of the layer and adds the information to the list `model_arch_list`.