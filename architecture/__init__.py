import torch
from pathlib import Path
from . import srcnn 
from . import fsrcnn 
from . import edsr 
from . import espcn 
from . import srgan 
from . import esrgan 
from . import prosr 

class arch(object):
	def __init__(self, model, upscale_factor = 2, grad = False):
		super(arch, self).__init__()
		torch.set_grad_enabled(False)
		if model == 'srcnn':
			tmp = srcnn.SRCNN(1, 64, upscale_factor)
		
		if model == 'fsrcnn':
			tmp = fsrcnn.FSRCNN(1, upscale_factor)
		
		if model == 'srgan':
			tmp = srgan.SRGAN(upscale_factor)
		
		if model == 'edsr':
			tmp = edsr.EDSR(upscale_factor)
		
		if model == 'espcn':
			tmp = espcn.ESPCN(upscale_factor)
		
		if model == 'prosr':
			tmp = prosr.ProSR(residual_denseblock = True, num_init_features = 160, growth_rate = 40, level_config = [[8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8], [8]], max_num_feature = 312, ps_woReLU = False, level_compression = -1, bn_size = 4, res_factor = 0.2, max_scale = 8)

		if model == 'esrgan':
			tmp = esrgan.ESRGAN(
				3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                mode='CNA', res_scale=1, upsample_mode='upconv'
			)

		pth = Path.cwd() / 'architecture' / model / 'pretrained_models' / ('x' + str(upscale_factor) + '.pth')
		tmp.load_state_dict(torch.load(pth))
		self.model = tmp