from jacobinet.layers.convolutional import (
    BackwardConv1D,
    BackwardConv2D,
    BackwardConv3D,
    BackwardDepthwiseConv1D,
    BackwardDepthwiseConv2D,
)
from jacobinet.layers.core import BackwardActivation, BackwardDense, BackwardEinsumDense
from jacobinet.layers.custom import (
    BackwardAbs,
    get_backward_Abs,
    get_backward_Absolute,
    get_backward_Arccos,
    get_backward_Arccosh,
    get_backward_Arcsin,
    get_backward_Arcsinh,
    get_backward_Arctan,
    get_backward_Arctanh,
    get_backward_Cos,
    get_backward_Cosh,
    get_backward_Log,
    get_backward_Log1p,
    get_backward_Log2,
    get_backward_Log10,
    get_backward_Sqrt,
    get_backward_Square,
    get_backward_Tan,
)
from jacobinet.layers.normalization import BackwardBatchNormalization
from jacobinet.layers.pooling import (
    BackwardAveragePooling1D,
    BackwardAveragePooling2D,
    BackwardAveragePooling3D,
    BackwardGlobalAveragePooling1D,
    BackwardGlobalAveragePooling2D,
    BackwardGlobalAveragePooling3D,
)
from jacobinet.layers.reshaping import (
    BackwardCropping1D,
    BackwardCropping2D,
    BackwardCropping3D,
    BackwardFlatten,
    BackwardPermute,
    BackwardRepeatVector,
    BackwardReshape,
    BackwardUpSampling1D,
    BackwardUpSampling2D,
    BackwardUpSampling3D,
    BackwardZeroPadding1D,
    BackwardZeroPadding2D,
    BackwardZeroPadding3D,
)

from .activations import (
    get_backward_ELU,
    get_backward_LeakyReLU,
    get_backward_PReLU,
    get_backward_ReLU,
)
from .convolutional import (
    get_backward_Conv1D,
    get_backward_Conv2D,
    get_backward_DepthwiseConv1D,
    get_backward_DepthwiseConv2D,
    get_backward_SeparableConv1D,
    get_backward_SeparableConv2D,
)
from .core import get_backward_Activation, get_backward_Dense, get_backward_EinsumDense
from .layer import (
    BackwardBoundedLinearizedLayer,
    BackwardLayer,
    BackwardLinearLayer,
    BackwardNonLinearLayer,
)
from .merging import (  # BackwardMergeNonLinearLayer,
    BackwardAdd,
    get_backward_Add,
    get_backward_Average,
    get_backward_Concatenate,
    get_backward_Maximum,
    get_backward_Minimum,
    get_backward_Multiply,
    get_backward_Subtract,
)
from .merging.base_merge import BackwardMergeNonLinearLayer

# from .custom import get_backward_MulConstant, get_backward_PlusConstant
from .normalization import get_backward_BatchNormalization
from .pooling import (
    get_backward_AveragePooling1D,
    get_backward_AveragePooling2D,
    get_backward_AveragePooling3D,
    get_backward_GlobalAveragePooling1D,
    get_backward_GlobalAveragePooling2D,
    get_backward_GlobalAveragePooling3D,
    get_backward_GlobalMaxPooling2D,
    get_backward_MaxPooling2D,
)
from .reshaping import (
    get_backward_Cropping1D,
    get_backward_Cropping2D,
    get_backward_Cropping3D,
    get_backward_Flatten,
    get_backward_Permute,
    get_backward_RepeatVector,
    get_backward_Reshape,
    get_backward_UpSampling1D,
    get_backward_UpSampling2D,
    get_backward_UpSampling3D,
    get_backward_ZeroPadding1D,
    get_backward_ZeroPadding2D,
    get_backward_ZeroPadding3D,
)
