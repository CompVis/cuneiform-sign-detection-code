import time
from collections import OrderedDict
import copy

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.module import _addindent

import torchvision
from torchvision.transforms import *

import matplotlib.pyplot as plt


# HELPER FUNCTIONS


def weights_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)  # 0.005
        m.bias.data.zero_()


def torch_summarize(model, show_weights=True, show_parameters=True):
    # code found here: https://stackoverflow.com/questions/42480111/model-summary-in-pytorch
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr


def summary(mymodule, input_size):
    # code from PR by isaykatsman https://github.com/pytorch/pytorch/pull/3043
    def register_hook(module):
        def hook(module, input, output):
            if module._modules:  # only want base layers
                return
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)
            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = None
            if output.__class__.__name__ == 'tuple':
                summary[m_key]['output_shape'] = list(output[0].size())
            else:
                summary[m_key]['output_shape'] = list(output.size())
            summary[m_key]['output_shape'][0] = None

            params = 0
            # iterate through parameters and count num params
            for name, p in module._parameters.items():
                params += torch.numel(p.data)
                summary[m_key]['trainable'] = p.requires_grad

            summary[m_key]['nb_params'] = params

        if not isinstance(module, torch.nn.Sequential) and \
           not isinstance(module, torch.nn.ModuleList) and \
           not (module == mymodule):
            hooks.append(module.register_forward_hook(hook))

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1, *in_size)) for in_size in input_size]
    else:
        x = Variable(torch.randn(1, *input_size))

    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    mymodule.apply(register_hook)
    # make a forward pass
    mymodule(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    # print out neatly
    def get_names(module, name, acc):
        if not module._modules:
            acc.append(name)
        else:
            for key in module._modules.keys():
                p_name = key if name == "" else name + "." + key
                get_names(module._modules[key], p_name, acc)
    names = []
    get_names(mymodule, "", names)

    col_width = 25  # should be >= 12
    summary_width = 61

    def crop(s):
        return s[:col_width] if len(s) > col_width else s

    print('_' * summary_width)
    print('{0: <{3}} {1: <{3}} {2: <{3}}'.format(
        'Layer (type)', 'Output Shape', 'Param #', col_width))
    print('=' * summary_width)
    total_params = 0
    trainable_params = 0
    for (i, l_type), l_name in zip(enumerate(summary), names):
        d = summary[l_type]
        total_params += d['nb_params']
        if 'trainable' in d and d['trainable']:
            trainable_params += d['nb_params']
        print('{0: <{3}} {1: <{3}} {2: <{3}}'.format(
            crop(l_name + ' (' + l_type[:-2] + ')'), crop(str(d['output_shape'])),
            crop(str(d['nb_params'])), col_width))
        if i < len(summary) - 1:
            print('_' * summary_width)
    print('=' * summary_width)
    print('Total params: ' + str(total_params))
    print('Trainable params: ' + str(trainable_params))
    print('Non-trainable params: ' + str((total_params - trainable_params)))
    print('_' * summary_width)


def visualize_model(model, dataloader, re_transform, device, num_images=6):
    was_training = model.training
    images_so_far = 0
    fig = plt.figure(figsize=(10, 10))

    # switch to bachnorm and dropout to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # compute predictions using the model
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(preds[j]))

                ax.imshow(re_transform(inputs.cpu().data[j].clone()), cmap=plt.cm.Greys_r)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def prepare_embedding(model_feature, dataloader, re_transform, device):

    # switch to bachnorm and dropout to eval mode
    model_feature.eval()

    f_list = []
    i_list = []
    l_list = []

    with torch.no_grad():
        # inputs, labels = next(iter(dataloaders['train']))
        for inputs, labels in dataloader:
            em_sz = inputs.shape[0]

            # append labels
            l_list.append(labels)

            # undo transform, convert to RGB, and convert back to tensor
            t_list = []
            for t in inputs:
                t_list.append(torchvision.transforms.ToTensor()(re_transform(t.clone()).convert('RGB')))

                # append images
            i_list.append(torch.stack(t_list))

            # compute feature
            inputs = inputs.to(device)

            # append features
            f_list.append(model_feature(inputs).view(em_sz, -1).data)

    return torch.cat(f_list), torch.cat(l_list).numpy(), torch.cat(i_list)


def prepare_prcurves(model, dataloader, device):
    # create softmax
    softmax = nn.Softmax()
    # loop over dataset with dataloader
    p_list = []
    l_list = []
    with torch.no_grad():
        # inputs, labels = next(iter(dataloaders['train']))
        for inputs, labels in dataloader:
            # append labels
            l_list.append(labels)
            # prepare input
            inputs = inputs.to(device)
            # apply network model
            output = model(inputs)
            # compute softmax
            predicted = softmax(output)
            # append features
            p_list.append(predicted.data.cpu())

    # concat to tensors
    return torch.cat(p_list), torch.cat(l_list)


def preprocess_tablet_im(pil_im, scale, shift=5.0):
    # compute scaled size
    imw, imh = pil_im.size
    imw = int(imw * scale)
    imh = int(imh * scale)
    # determine crop size
    crop_sz = [int(imh - shift), int(imw - shift)]
    # tensor-space transforms
    ts_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[1]),  # normalize
    ])
    # compose transforms
    tablet_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x.convert('L')),  # convert to gray
            Resize((imh, imw)),  # resize according to scale
            FiveCrop((crop_sz[0], crop_sz[1])),  # oversample
            torchvision.transforms.Lambda(
                lambda crops: torch.stack([ts_transform(crop) for crop in crops])),  # returns a 4D tensor
        ])
    # apply transforms
    im_list = tablet_transform(pil_im)
    return im_list


def predict(model, im_list, device, use_bbox_reg=False):
    inputs = im_list

    with torch.no_grad():  # faster, less memory usage
        # prepare input
        inputs = inputs.to(device)

        # apply network model
        # output = model(inputs) # consumes to much memory
        output = []
        for in_im in inputs:
            output.append(model(in_im.unsqueeze(0)))
        output = torch.cat(output, dim=0)

        # convert to numpy
        predicted = output.data.cpu().numpy()
        # free memory?!
        # del output

    # TODO: integrate bbox regression
    result_roi = []
    # stack detections to single tensor
    predicted_roi = []
    if use_bbox_reg:
        predicted_roi = np.stack(result_roi).squeeze()

    return predicted, predicted_roi


# TRAINER HELPER


def get_tensorboard_writer(logs_folder='runs_new', comment=''):
    # init logger
    import os
    import socket
    from datetime import datetime
    from tensorboardX import SummaryWriter

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(logs_folder, current_time + '_' + socket.gethostname() + comment)
    writer = SummaryWriter(log_dir=log_dir)  # comment='_{}'.format(weights_path.split('/')[1].split('.')[0])
    return writer


# TRAINER FUNCTIONS

def train_model(model, criterion, optimizer, scheduler, writer, dataloaders, dataset_sizes, device, num_epochs=25, test_every=10):
    ''' generic trainer function '''
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase

        phases = ['train', 'dev']
        if epoch % test_every != 0:
            phases = ['train']

        for phase in phases:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # else:
                        # for name, param in model.named_parameters():
                        #     writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

                # statistics
                running_loss += loss.item()  # * inputs.size(0)  # uncomment this to fix a legacy bug XXX
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / float(dataset_sizes[phase])

            # write to logger
            writer.add_scalar('data/{}/loss'.format(phase), epoch_loss, epoch)
            writer.add_scalar('data/{}/acc'.format(phase), epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} Number correct: {} '.format(phase, running_corrects))

            # deep copy the model
            if phase == 'dev' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f} at {}'.format(best_acc, best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
