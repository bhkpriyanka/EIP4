***print(score) : [0.03689879849365975, 0.9926]***
  
**1.Convolution:** In CNNs, convolutions are elementwise products of filter weights with input layer pixels and summed to give single pixel value.Then the filter is slided and the operations are repeated.The output pixel matrix captures the input features at roughly same positions i.e. preserves spatial information unlike fully connected networks.  
**2.Filters/Kernels** These are feature extractors.Weighted matrices which when convolved with input,look for features specified by the combination of weights.  
**3.Epochs**Going over the entire training dataset ***once*** (one Forward pass + one Backwardpass for all batches and coming up with a final set of weights) is one epoch.  
**4.1x1 Convolution**Used as Feature merger.Multiplies whole channel with same number & combines existing channels to create new contextually linked channels.Used to reduce number of parameters and depth of network.  
**5.3x3 Convolution**Most commonly used filter to extract features in CNNs.A technique using a 3x3 matrix of weights,slided over input in strides to extract required features.Looks at 9 pixels at a time.One 3x3 convolution reduces size of input by 2 pixels in width and height. [e.g:(406 x 406) * 3x3 > (404 x 404)]  
**6.Feature Maps** These are the channels that are outputted after convolution of a previous input with the filter.They could patterns built from previoulsy extracted edges and gradients or objects built from previously extracted parts of objects,etc.
**7.Activation Function**  Activation functions are used to represent the linear/non linear relationship existing between the inputs and outputs.
**8.Receptive Field** The number of pixels that can been be seen by each pixel of a feature map (output matrix after every convolution).
