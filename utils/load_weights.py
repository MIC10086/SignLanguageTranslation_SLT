import h5py
import numpy as np

def loadweights(model_pretrained, path_weights):
    weights = []
    if model_pretrained == 'c3d':
        c3d_weights = h5py.File(path_weights, 'r')
        for layer in ['layer_0','layer_2','layer_4','layer_7',]:
            weights.append([
                np.moveaxis(np.r_[c3d_weights[layer]['param_0']], (0,1),(4,3)), #Cambio los ejes porque c3d estan con canales primero
                np.r_[c3d_weights[layer]['param_1']]
                   ])
        return weights

    elif model_pretrained == 'vgg16':
        vgg16 = h5py.File(path_weights, 'r')
        for j, block in enumerate(['block1_conv1','block2_conv1','block3_conv1', 'block4_conv1' ]):
            num = block[-1]
            num_block = block[5]
            w = vgg16[block]['block'+num_block+'_conv'+num+'_W_1:0']
            b = vgg16[block]['block'+num_block+'_conv'+num+'_b_1:0']
            w_expand = np.expand_dims(w,axis=2)
            weights.append([(np.concatenate([w_expand, w_expand, w_expand], axis=2)/3.),b])
        return weights
    elif model_pretrained == None:
        return [None,None,None,None]
    else:
        raise ValueError('The model to load weights is invalid. Valid models are "c3d", '
        '"vgg16", None. Value given: '+str(model_pretrained))