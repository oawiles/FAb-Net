# Add an encoder/decoder
from tensorboardX import SummaryWriter

import sys
from utils_scheduler.TrackLoss import TrackLoss
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import numpy as np

from models_multiview import FrontaliseModelMasks_wider

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse

BASE_LOCATION = os.environ['BASE_LOCATION']


arguments = argparse.ArgumentParser()
arguments.add_argument('--lr', type=float, default=0.0001)
arguments.add_argument('--momentum', type=float, default=0.9)
arguments.add_argument('--load_old_model', action='store_true', default=False)
arguments.add_argument('--num_views', type=int, default=2, help='Number of source views + 1 (e.g. the target view) so set = 2 for 1 source view')
arguments.add_argument('--continue_epoch', type=int, default=0)
arguments.add_argument('--crop_size', type=int, default=180)
arguments.add_argument('--num_additional_ids', type=int, default=32)
arguments.add_argument('--use_landmark_supervision', action='store_true', default=False)
arguments.add_argument('--use_landmark_mask_supervision', action='store_true', default=False)
arguments.add_argument('--num_workers', type=int, default=1)
arguments.add_argument('--max_percentile', type=float, default=0.85)
arguments.add_argument('--diff_percentile', type=float, default=0.1)
arguments.add_argument('--batch_size', type=int, default=128)
arguments.add_argument('--log_dir', type=str, default=BASE_LOCATION+'/code_faces/runs/')
arguments.add_argument('--embedding_size', type=int, default=256)
arguments.add_argument('--run_dir', type=str, default='curriculumwidervox2%d_fabnet%s/lr_%.4f_lambda%.4f_nv%d_addids%d_cropsize%d')
arguments.add_argument('--old_model', type=str, default=BASE_LOCATION + '')
arguments.add_argument('--model_epoch_path', type=str, default=BASE_LOCATION + '/faces/models/disentangling/curriculumwidervox2%.4f_emb%d_bs%d_lambda%.4f_photomask%s_nv%d_addids%d_cropsize%d')
arguments.add_argument('--learn_mask', action='store_true')
opt = arguments.parse_args()


opt.run_dir = opt.run_dir % (opt.embedding_size, str(opt.use_landmark_supervision), opt.lr, 0, opt.num_views, opt.num_additional_ids, opt.crop_size)
opt.model_epoch_path = opt.model_epoch_path % (opt.lr, opt.embedding_size, opt.batch_size, 0, str(opt.use_landmark_supervision), opt.num_views, opt.num_additional_ids, opt.crop_size)

opt.model_epoch_path = opt.model_epoch_path + 'epoch%d.pth'

model = FrontaliseModelMasks_wider(3, inner_nc=opt.embedding_size, num_masks=0, num_additional_ids=opt.num_additional_ids)

model.lr = opt.lr
model.momentum = opt.momentum
writer = SummaryWriter('/%s/%s' % (opt.log_dir, opt.run_dir))

model = model.cuda()

criterion_reconstruction = nn.L1Loss(reduce=False).cuda()


from VoxCelebData_withmask import VoxCeleb2 as VoxCeleb2

if opt.num_views > 2:
	optimizer = optim.SGD([{'params' : model.decoder.parameters(), 'lr' : 0},
							{'params' : model.encoder.parameters(), 'lr' : opt.lr},
							{'params' : model.mask.parameters(), 'lr' : opt.lr}], lr=opt.lr, momentum=opt.momentum)
else:
	optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)


def multiview_maskfaceloss(input_image, model, front_images):
	xc = model.encoder(input_image)

	err_recons = None
	all_samplers = [0] * len(front_images)
	all_masks = [0] * len(front_images)

	xcs = [0] * len(front_images)
	weighting_masks = [0] * len(front_images)

	for i, front_image in enumerate(front_images):
		xcs[i] = model.encoder(front_image)

	xc_sampled_images = [0] * len(front_images)
	xcs_samplers = [0] * len(front_images)
	attention_maps = [0] * len(front_images)

	for i in range(0, len(front_images)):
		mask = model.mask(torch.cat((xcs[i], xc), 1))


		input_vector = torch.cat((xcs[i], xc), 1)
		samplers = model.decoder(input_vector) 
		xs = np.linspace(-1,1,samplers.size(2))
		xs = np.meshgrid(xs, xs)
		xs = np.stack(xs, 2)
		xs = torch.Tensor(xs).unsqueeze(0).repeat(xc.size(0), 1,1,1).cuda()
		xs = Variable(xs, requires_grad=False)
		samplers = (samplers.permute(0,2,3,1) + xs).clamp(min=-1,max=1)


		sampled_image = F.grid_sample(front_images[i].detach(), samplers).unsqueeze(4)


		xc_sampled_images[i] = sampled_image
		all_samplers[i] = samplers
		attention_maps[i] = mask.unsqueeze(4)
		all_masks[i] = mask.detach().squeeze().unsqueeze(1)
			

	sampled_images = torch.cat(xc_sampled_images, 4)
	attention_maps = torch.cat(attention_maps, 4)
	result_image = (sampled_images * attention_maps.exp()).sum(dim=4) / attention_maps.exp().sum(dim=4)

	for i in range(0, len(all_masks)):
		all_masks[i] = all_masks[i].exp() / attention_maps.detach().exp().sum(dim=4)

	err = Variable(torch.zeros(1,)).cuda()
	return result_image, all_samplers, all_masks


		

def train(epoch, model, criterion, optimizer, num_additional_ids=5, minpercentile=0, maxpercentile=50):
	train_set = VoxCeleb2(opt.num_views, epoch, 1, jittering=True)
	training_data_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True)
    
	t_loss = 0


	for iteration, batch in enumerate(training_data_loader, 1):
		optimizer.zero_grad()
		input_image = Variable(batch[0]).cuda()
		
		offset = 1

		t_front_images = batch[offset:]
		front_images = [0] * (len(t_front_images) / offset)
		weightings = [0] * (len(t_front_images) / offset)
		for i in range(0, len(t_front_images) / offset):
			front_images[i] = Variable(t_front_images[i*offset]).cuda()
		
		output_image, samplers, masks  = multiview_maskfaceloss(input_image, model, front_images)

		loss_pre_sort = criterion_reconstruction(output_image, input_image).view(output_image.size(0), -1).mean(dim=1)
		loss = loss_pre_sort.sort()[0][int(minpercentile * input_image.size(0)):int(maxpercentile * input_image.size(0))].mean()
		loss.backward()


		t_loss += loss.cpu().data[0]
		
		if iteration == 1:
			writer.add_image('Image_train/%d_input' % iteration, torchvision.utils.make_grid(input_image[0:10,:,:,:].data), epoch)
			writer.add_image('Image_train/%d_output' % iteration, torchvision.utils.make_grid(output_image[0:10,:,:,:].data), epoch)

			loss_image = output_image * 0 + loss_pre_sort.detach().unsqueeze(1).unsqueeze(1).unsqueeze(1) / loss_pre_sort.max()
			writer.add_image('Image_train/%d_weighting' % iteration, torchvision.utils.make_grid(loss_image[0:10,:,:,:].data), epoch)

			for i, mask in enumerate(masks):
				print(mask.size())
				
				writer.add_image('Image_train/%d_inputmask_%d' % (iteration, i), torchvision.utils.make_grid(mask[0:10,:,:,:].data), epoch)


		optimizer.step()

		if iteration % 100 == 0:
			print("Train: Epoch {}: {}/{} with error {:.4f}". \
				format(epoch, iteration, len(training_data_loader), t_loss / float(iteration)))

	return {'reconstruction_error' : t_loss / float(iteration)}

def val(epoch, model, criterion, optimizer, minpercentile=0, maxpercentile=50):
	val_set = VoxCeleb2(opt.num_views, 0, 2, jittering=True) 

	val_data_loader = DataLoader(dataset=val_set, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=False)
    
	t_loss = 0


	for iteration, batch in enumerate(val_data_loader, 1):
		input_image = Variable(batch[0]).cuda()
		
		offset = 1

		t_front_images = batch[offset:]
		front_images = [0] * (len(t_front_images) / offset)
		for i in range(0, len(t_front_images) / offset):
			front_images[i] = Variable(t_front_images[i*offset]).cuda()

		output_image, samplers, masks = multiview_maskfaceloss(input_image, model, front_images)
		


		loss_pre_sort = criterion_reconstruction(output_image, input_image).view(output_image.size(0), -1).mean(dim=1)
		loss = loss_pre_sort.sort()[0][int(minpercentile * input_image.size(0)):int(maxpercentile * input_image.size(0))].mean()

		t_loss += loss.cpu().data[0]
		
		if iteration % 1000 == 0 or iteration == 1:
			writer.add_image('Image_val/%d_input' % iteration, torchvision.utils.make_grid(input_image[0:10,:,:,:].data), epoch)
			
			writer.add_image('Image_val/%d_output' % iteration, torchvision.utils.make_grid(output_image[0:10,:,:,:].data), epoch)
			
			if opt.use_landmark_mask_supervision:
				for i, (a_mask, t_a_masks) in enumerate(additional_masks, 0):
					print(a_mask.size())
					
					writer.add_image('Image_val/%d_amask_%d' % (iteration, i), torchvision.utils.make_grid(a_mask[0:10,:,:,:].data), epoch)
					for j, t_a_mask in enumerate(t_a_masks, 0):
						writer.add_image('Image_val/%d_tamask_%d_%d' % (iteration, i, j), torchvision.utils.make_grid(res_masks[i][j].data), epoch)
			for i, mask in enumerate(front_images):
				writer.add_image('Image_val/%d_inputfront_%d' % (iteration, i), torchvision.utils.make_grid(mask[0:10,:,:,:].data), epoch)


		if iteration % 100 == 0:
			print("Val: Epoch {}: {}/{} with error {:.4f}".format(epoch, iteration,
				len(val_data_loader), t_loss / float(iteration)))


	return {'reconstruction_error' : t_loss / float(iteration)} 

def checkpoint(model, save_path):
	checkpoint_state = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict(), 'epoch' : model.epoch, 
							'lr' : model.lr, 'momentum' : model.momentum, 'opts' : opt}

	torch.save(checkpoint_state, save_path)

def run(minpercentile=0, maxpercentile=0.5):
	scheduler = TrackLoss()

	plateauscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
	if opt.continue_epoch > 0:
		past_state = torch.load(opt.model_epoch_path % (opt.continue_epoch - 1))
		model.load_state_dict(torch.load(opt.model_epoch_path % (opt.continue_epoch - 1))['state_dict'])
		optimizer.load_state_dict(torch.load(opt.model_epoch_path % (opt.continue_epoch - 1))['optimizer'])

		percentiles = past_state['opts']
		minpercentile = percentiles.minpercentile
		maxpercentile = percentiles.maxpercentile

	for epoch in range(opt.continue_epoch, 10000):
		model.epoch = epoch
		model.optimizer_state = optimizer.state_dict()
		model.train()
		train_loss = train(epoch, model, criterion_reconstruction, optimizer, minpercentile=minpercentile, maxpercentile=maxpercentile)
		model.eval()
		with torch.no_grad():
			loss = val(epoch, model, criterion_reconstruction, optimizer, minpercentile=0, maxpercentile=1)

		scheduler.update(loss['reconstruction_error'], epoch)

		if scheduler.drop_learning_rate(epoch):
			if maxpercentile < opt.max_percentile:
				maxpercentile += opt.diff_percentile
				minpercentile += opt.diff_percentile
				scheduler = TrackLoss()
			else:
				plateauscheduler.step(loss['reconstruction_error'])

		writer.add_scalars('loss_recon/train_val', {'train' : train_loss['reconstruction_error'], 'val' : loss['reconstruction_error']}, epoch)

		if epoch % 10 == 0:
			checkpoint(model, opt.model_epoch_path % epoch)

			for i in range(1,15):
				if os.path.exists(opt.model_epoch_path % (epoch - i)):
					os.remove(opt.model_epoch_path % (epoch - i))

		opt.minpercentile = minpercentile
		opt.maxpercentile = maxpercentile


if __name__ == '__main__':
	if opt.load_old_model:
		model.load_state_dict(torch.load(opt.old_model)['state_dict'])
		percentiles = torch.load(opt.old_model)['opts']
		minpercentile = percentiles.minpercentile
		maxpercentile = percentiles.maxpercentile

		run(minpercentile, maxpercentile)
	
	else:
		run()





