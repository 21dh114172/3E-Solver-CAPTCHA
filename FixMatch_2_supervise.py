import argparse
import torch
from torch import optim
import matplotlib
from torch.autograd import Variable
import pprint
import sys

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_datasets_mean_teacher, get_label_dict, get_vocab
from models import CNNSeq2Seq
from util import compute_seq_acc, Seq2SeqLoss, ConsistentLoss, ConsistentLoss_MT_Temperature, \
    get_current_consistency_weight, SaveBestModel, save_model
import random
import time
import os

parser = argparse.ArgumentParser(description='PyTorch Captcha Training Using Mean-Teacher')

parser.add_argument('--dataset', default='google', type=str, help="the name of dataset")
parser.add_argument('--label', default="500.txt", type=str, help='the labels of captcha images used for training')
parser.add_argument('--batch-size', default=32, type=int, help='batch size for training and test')
parser.add_argument('--secondary-batch-size', default=64, type=int, help='batch size for unlabel')
parser.add_argument('--unlabeled-number', default=5000, type=int, help='the number of unlabeled images')
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--epoch', default=700, type=int, help='the number of training epochs')
parser.add_argument('--t', default=0.5, type=float, help='temperature of MT')
parser.add_argument('--weight', default=100.0, type=float, help='the weight of consistency loss')
parser.add_argument('--teachforce', action="store_false", help='whether to use teaching force(Default: True)')
parser.add_argument('--lr', default=0.02, type=float, help='learning rate')
parser.add_argument('--seed', default=42, type=int, help='running seed')
parser.add_argument('--save-epoch', default=50, type=int, help='save model per epoch')
parser.add_argument('--wait-save-best-epoch', default=30, type=int, help='wait until epoch to save best model')
parser.add_argument('--delimiter-label', default=' ', type=str, help='delimiter in label file')
parser.add_argument('--hidden-size', default=128, type=int, help='hidden size for prediction layer')
parser.add_argument('--TARGET-HEIGHT', default=64, type=int, help='resize image height to target height')
parser.add_argument('--TARGET-WIDTH', default=128, type=int, help='resize image width to target width')
parser.add_argument('--use-abi-group', default=True, type=bool, help='use ABI Group strategy')
parser.add_argument('--load-model', default='', type=str, help='path to previous model')
parser.add_argument('--load-model-ema', default='', type=str, help='path to previous ema model')
parser.add_argument('--use-new-optimizer', action="store_true", help='create new optimizer for loaded model (Default: False)')
parser.add_argument('--use-new-vocab', action="store_true", help='create new label dict for new dataset from loaded model, might cause mismatch output layer (Default: False)')
parser.add_argument('--vocab', default='', type=str, help='Provide vocab for current training session, might cause mismatch output layer (Default: False)')
parser.add_argument('--wandb-api-key', default='', type=str, help='Provide wandb api key for log tracking')
parser.add_argument('--wandb-run-name', default='', type=str, help='Provide wandb run name')


args = parser.parse_args()

args.vocab = os.environ.get("my_vocab", args.vocab)
args.delimiter_label = os.environ.get("delimiter_label", args.delimiter_label)
SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = True

pprint.pprint(args)
USE_CUDA = torch.cuda.is_available()

LR = args.lr
NUM_EPOCHS = args.epoch

is_model_path_empty = args.load_model == '' and args.load_model_ema == ''
is_model_path_exist = not is_model_path_empty if os.path.exists(args.load_model) and os.path.exists(args.load_model_ema) else False
if (not is_model_path_empty and not is_model_path_exist):
    raise Exception(f"Could not find previous model from this path, \n\tBase: {args.load_model}\n\tEma:{args.load_model_ema}\n")

dataloader_train_labeled, dataloader_train_nolabeled, dataloader_test, id2token, MAXLEN, _ = load_datasets_mean_teacher(
    args)

vocab = get_vocab(get_label_dict(args))
token = "".join(list(id2token.values()))
print(f"token:{token}")
print(f"vocab:{vocab}")

model = CNNSeq2Seq(vocab_size=len(id2token), max_len=MAXLEN, hidden_size=args.hidden_size)
model_ema = CNNSeq2Seq(vocab_size=len(id2token), max_len=MAXLEN, hidden_size=args.hidden_size)

class_criterion = Seq2SeqLoss()
consistent_criterion = ConsistentLoss(args.threshold)
consistent_criterion_mt = ConsistentLoss_MT_Temperature(args.t)

if USE_CUDA:
    model = model.cuda()
    model_ema = model_ema.cuda()
    class_criterion = class_criterion.cuda()
    consistent_criterion = consistent_criterion.cuda()
    consistent_criterion_mt = consistent_criterion_mt.cuda()

for param_main, param_ema in zip(model.parameters(), model_ema.parameters()):
    param_ema.data.copy_(param_main.data)  # initialize
    param_ema.requires_grad = False  # not update by gradient

params = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)

if (not is_model_path_empty and is_model_path_exist):
    checkpoint = torch.load(args.load_model)
    checkpoint_ema = torch.load(args.load_model_ema)
    model.load_state_dict(checkpoint["model_state_dict"])
    model_ema.load_state_dict(checkpoint_ema["model_state_dict"])
    if not args.use_new_optimizer:
        print(f"Loaded optimizer from model {args.load_model} \n")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print(f"Create new optimizer \n")

    model.train()
    model_ema.train()
    print(f"Loaded previous model from {args.load_model}")
else:
    print(f"Create model from scratch \n")

train_loss_class = []
train_loss_consistency = []
train_loss_consistency_mt = []
train_accclevel = []
train_accuracy = []

test_class_loss = []
test_accclevel = []
test_accuracy = []

test_class_loss_ema = []
test_accclevel_ema = []
test_accuracy_ema = []
save_best_model = SaveBestModel()
save_best_model_ema = SaveBestModel()
for epoch in range(NUM_EPOCHS):
    time_epoch = time.time()
    loss_1 = loss_2 = loss_mt = accuracy = accclevel = 0

    iter_label = dataloader_train_labeled.__iter__()
    
    for num_iter, (inputs_x, targets_x) in enumerate(dataloader_train_labeled):
        if USE_CUDA:
            targets_x = targets_x.cuda()
            inputs_x = inputs_x.cuda()

        batch_size = inputs_x.size(0)
        if args.teachforce:
            logits_x = model.forward_train(inputs_x, targets_x)
        else:
            logits_x = model.forward_test(inputs_x)

        optimizer.zero_grad()

        Lx = class_criterion(logits_x, targets_x)

        max_len = targets_x.size(1)
        acccl, acc = compute_seq_acc(logits_x, targets_x, max_len)
        acccl /= args.batch_size
        acc /= args.batch_size

        loss_all = Lx
        accclevel += acccl
        accuracy += acc
    
    train_loss_class.append(loss_1 / len(dataloader_train_nolabeled))
    train_loss_consistency.append(loss_2 / len(dataloader_train_nolabeled))
    train_loss_consistency_mt.append(loss_mt / len(dataloader_train_nolabeled))
    train_accclevel.append(accclevel / len(dataloader_train_nolabeled))
    train_accuracy.append(accuracy / len(dataloader_train_nolabeled))
    print("{} epoch train\n"
          "class loss: {} consistent loss {} consistent loss mt {}\n"
          "accuracy {} accclevel {}".format(epoch, train_loss_class[-1], train_loss_consistency[-1],
                                            train_loss_consistency_mt[-1], train_accuracy[-1],
                                            train_accclevel[-1]))

    model = model.eval()
    loss = accuracy = accclevel = total = 0
    for num_iter, (x, y) in enumerate(dataloader_test):
        x = Variable(x)
        y = Variable(y)
        if USE_CUDA:
            x = x.cuda()
            y = y.cuda()

        outputs = model.forward_test(x)

        loss_batch = class_criterion(outputs, y)

        max_len = y.size(1)
        acccl, acc = compute_seq_acc(outputs, y, max_len)

        loss += loss_batch.item()
        accclevel += acccl
        accuracy += acc
        total += y.size(0)

    test_class_loss.append(loss / len(dataloader_test))
    test_accclevel.append(accclevel / total)
    test_accuracy.append(accuracy / total)
    print("test loss: {}\n"
          "accuracy {} accclevel {}".format(test_class_loss[-1], test_accuracy[-1], test_accclevel[-1]))
    print(f"Model predict match groundtruth: {accuracy}/{total}")
    model = model.train()

    model_ema = model_ema.eval()
    loss = accuracy = accclevel = total = 0
    for num_iter, (x, y) in enumerate(dataloader_test):
        x = Variable(x)
        y = Variable(y)
        if USE_CUDA:
            x = x.cuda()
            y = y.cuda()

        outputs = model_ema.forward_test(x)

        loss_batch = class_criterion(outputs, y)

        max_len = y.size(1)
        acccl, acc = compute_seq_acc(outputs, y, max_len)

        loss += loss_batch.item()
        accclevel += acccl
        accuracy += acc
        total += y.size(0)

    test_class_loss_ema.append(loss / len(dataloader_test))
    test_accclevel_ema.append(accclevel / total)
    test_accuracy_ema.append(accuracy / total)
    print("test loss ema: {}\n"
          "accuracy {} accclevel {}".format(test_class_loss_ema[-1], test_accuracy_ema[-1], test_accclevel_ema[-1]))
    print(f"Model_ema predict match groundtruth: {accuracy}/{total}")
    model_ema = model_ema.train()

    print(f"epoch time {time.time()-time_epoch}\n")
    if (epoch + 1) >= args.wait_save_best_epoch:
        save_best_model(
                test_class_loss[-1], epoch, model, optimizer, class_criterion, vocab=vocab, id2token=id2token, image_height=args.TARGET_HEIGHT, image_width=args.TARGET_WIDTH 
        )
        save_best_model_ema(
                test_class_loss_ema[-1], epoch, model_ema, optimizer, class_criterion, vocab=vocab, id2token=id2token, image_height=args.TARGET_HEIGHT, image_width=args.TARGET_WIDTH, model_name="ema_best_model.pth"
        )
        
    
    if (epoch + 1) % int(args.save_epoch) == 0:
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.plot(train_loss_class, 'r', label='train_class_loss')
        ax1.plot(train_loss_consistency, 'y', label='train_consistency_loss')
        ax1.plot(train_loss_consistency_mt, 'm', label='train_loss_consistency_mt')
        ax1.plot(test_class_loss, 'b', label='test_class_loss')
        ax1.plot(test_class_loss_ema, 'g', label='test_class_loss_ema')

        ax1.legend()
        ax2.plot(test_accuracy, 'b', label='test_accuracy')
        ax2.plot(test_accuracy_ema, 'g', label='test_accuracy_ema')
        ax2.plot(train_accuracy, 'r', label='train_accuracy')
        ax2.legend()
        test_acc_array = np.array(test_accuracy_ema)
        max_indx = np.argmax(test_acc_array)
        show_max = '[' + str(max_indx) + " " + str(test_acc_array[max_indx].item()) + ']'
        ax2.annotate(show_max, xytext=(max_indx, test_acc_array[max_indx].item()),
                     xy=(max_indx, test_acc_array[max_indx].item()))

        path = "FixMatch_2_" + args.dataset + "_" + str(args.label) + "_" + str(args.unlabeled_number) + "_" + str(
            args.teachforce) + "_" + str(args.weight) + "_" + str(args.threshold) + "_" + str(args.t) + "_" + str(
            args.seed)
        fig.savefig("result/" + path + ".png")
        np.save("result/" + path + "_test_accuracy_ema.npy", np.array(test_accuracy_ema))
        np.save("result/" + path + "_train_accuracy.npy", np.array(train_accuracy))
        np.save("result/" + path + "_test_class_loss_ema.npy", np.array(test_class_loss_ema))
        np.save("result/" + path + "_train_loss_class.npy", np.array(train_loss_class))

fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(train_loss_class, 'r', label='train_class_loss')
ax1.plot(train_loss_consistency, 'y', label='train_consistency_loss')
ax1.plot(train_loss_consistency_mt, 'm', label='train_loss_consistency_mt')
ax1.plot(test_class_loss, 'b', label='test_class_loss')
ax1.plot(test_class_loss_ema, 'g', label='test_class_loss_ema')

ax1.legend()
ax2.plot(test_accuracy, 'b', label='test_accuracy')
ax2.plot(test_accuracy_ema, 'g', label='test_accuracy_ema')
ax2.plot(train_accuracy, 'r', label='train_accuracy')
ax2.legend()
test_acc_array = np.array(test_accuracy_ema)
max_indx = np.argmax(test_acc_array)
show_max = '[' + str(max_indx) + " " + str(test_acc_array[max_indx].item()) + ']'
ax2.annotate(show_max, xytext=(max_indx, test_acc_array[max_indx].item()),
             xy=(max_indx, test_acc_array[max_indx].item()))

path = "FixMatch_2_" + args.dataset + "_" + str(args.label) + "_" + str(args.unlabeled_number) + "_" + str(
    args.teachforce) + "_" + str(args.weight) + "_" + str(args.threshold) + "_" + str(args.t) + "_" + str(args.seed)
fig.savefig("result/" + path + ".png")
np.save("result/" + path + "_test_accuracy_ema.npy", np.array(test_accuracy_ema))
np.save("result/" + path + "_train_accuracy.npy", np.array(train_accuracy))
np.save("result/" + path + "_test_class_loss_ema.npy", np.array(test_class_loss_ema))
np.save("result/" + path + "_train_loss_class.npy", np.array(train_loss_class))

save_model(args.epoch, model, optimizer, class_criterion, vocab=vocab, id2token=id2token, image_height=args.TARGET_HEIGHT, image_width=args.TARGET_WIDTH)
save_model(args.epoch, model_ema, optimizer, class_criterion, vocab=vocab, id2token=id2token, image_height=args.TARGET_HEIGHT, image_width=args.TARGET_WIDTH, model_name="ema_final_model.pth")
