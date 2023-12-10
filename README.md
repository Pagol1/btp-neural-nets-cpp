# btp-neural-nets-cpp

## Refactor Plans
+ Need to separate linear layers from activation functions, different layer for thos [Inspiration: TF, Torch]
+ Add a seperate optimizer module ==> Enable Adam etc. [Inspiration: TF, Torch]

### Code Plans
+ Rename: DenseLayer -> Linear
+ ReFunc: Remove ActivaitonFunctions
+ Add Sigmoid, ReLU, SoftMax layers

#### Items Changed
- `layer_info`
- `Layer.hasBias()`
- `DenseLayer` => `Linear`
