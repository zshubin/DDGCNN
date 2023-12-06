import sys
from DDGCNN.config import OptInit
from DDGCNN.model import DenseDDGCNN
from utils.ckpt_util import save_checkpoint
import logging
from data_loader.data_loaders import *
import math

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep  # 40*iter
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)  #0:1.5e-4

    iters = np.arange(epochs * niter_per_ep - warmup_iters) #(epoch-warmup_epoch)*iter
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule

def main(logger, sub):
    opt_class = OptInit(sub, logger)
    opt = opt_class.get_args()
    logger.info('===> Creating dataloader ...')
    train_loader, valid_data_loader = \
        data_generator_np(opt.data_dir, opt.batch_size, win_train=opt.time_win, down_sample=opt.down_sample, sample_freq=opt.sample_freq, device=opt.device)

    logger.info('===> Loading the network ...')
    model = DenseDDGCNN([opt.batch_size, opt.in_channels, opt.eeg_channel], opt.k_adj, opt.n_filters, opt.dropout,
                     n_blocks=opt.n_blocks,
                     nclass=opt.class_num, bias=opt.bias, norm=opt.norm, act=opt.act,trans_class=opt.trans_class, device=opt.device).to(opt.device)
    logger.info('===> Init the optimizer ...')
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=opt.lr_decay_rate, patience=opt.optim_patience, verbose=True, eps=1e-08)

    logging.info('===> Init Metric ...')
    opt.vali_loss  = 0.0
    opt.vali_acc   = 0.0
    opt.best_acc   = 0.0
    opt.total_loss = 0.0
    opt.total_acc  = 0.0
    opt.start_steps = 0

    logging.info('===> Start training ...')
    for e in range(opt.total_epochs):
        opt.epoch += 1
        train(model, train_loader, valid_data_loader, optimizer, scheduler, criterion, opt, logger)

        print(opt.best_acc)
        if opt.best_acc == 1. and opt.total_acc > 0.95:
            break

    logger.info('Saving the final model.Finish!')
    return opt.best_acc


def train(model, train_loader, valid_data_loader, optimizer, scheduler, criterion, opt, logger):
    model.train()
    total_loss = 0.
    total_acc  = 0.
    total_num  = 0.
    opt.iter   = 0.
    for i, (data, target) in enumerate(train_loader):
        opt.iter += 1
        opt.start_steps += 1
        data, target = data.to(opt.device), target.to(opt.device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target.long())
        total_loss += loss
        pred = torch.argmax(out, dim=1)
        total_acc += torch.sum(pred == target).item()
        loss.backward()
        optimizer.step()
        total_num += data.shape[0]
    opt.total_loss = total_loss/opt.iter
    opt.total_acc = total_acc/total_num
    scheduler.step(total_acc)
    logger.info('Epoch:{}\t Batch Loss: {}\t Batch Accuracy: {}\t Current_lr: {:5f}'.format(opt.epoch, opt.total_loss, opt.total_acc, optimizer.param_groups[0]['lr']))
    test(model, valid_data_loader, criterion, opt)
    opt.writer.add_scalars("Loss", {"Train": total_loss/opt.iter}, opt.epoch)
    opt.writer.add_scalars("Accuracy", {"Train": total_acc/total_num}, opt.epoch)
    opt.writer.add_scalars("Loss", {"Validation": opt.vali_loss}, opt.epoch)
    opt.writer.add_scalars("Accuracy", {"Validation": opt.vali_acc}, opt.epoch)
    if opt.vali_acc >= opt.best_acc:
        opt.best_acc = opt.vali_acc
        save_checkpoint({
            'epoch': opt.epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_value': opt.best_acc,
        }, opt.save_dir)


def test(model, test_loader, criterion, opt):
    model.eval()
    correct = 0
    loss = 0
    total_num = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(opt.device), target.to(opt.device)
            out = model(data)
            loss += criterion(out, target.long())
            pred = torch.argmax(out, dim=1)
            correct += torch.sum(pred == target).item()
            total_num += data.shape[0]
    valid_accuracy = correct / total_num
    valid_loss = loss / (i+1)
    logging.info('Epoch: [{}], Test loss: {}, Test_accuracy: {})\t'.format(opt.epoch, valid_loss, valid_accuracy))
    opt.vali_acc = valid_accuracy
    opt.vali_loss = valid_loss

def make_logger():
    loglevel = "info"
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(loglevel))
    log_format = logging.Formatter('%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    file_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logging.root = logger
    return logger

if __name__ == '__main__':
    # data_path = './processed_ssvep_data/'
    # sub_list = os.listdir(data_path)
    # logger = make_logger()
    # acc_list = []
    # for sess_index, sess in enumerate(sub_list):
    #     sub_list = os.listdir(os.path.join(data_path, sess))
    #     for sub_index, sub in enumerate(sub_list):
    #         print(sub)
    #         acc = main(logger, sub, sess)
    #         acc_list.append(acc)
    #         record_file.write(sub+' ' + str(acc) + '\n')
    #         print(acc_list)

    data_path = './40targetdata/'
    logger = make_logger()
    record_file = open('./record_DDGCNN_0.2_40target_3_3_128_0.5.txt', 'w')
    sub_list = os.listdir(data_path)
    # sub_list = sub_list[0:1]+sub_list[2:3]+sub_list[4:5]+sub_list[6:7]+sub_list[8:9]+sub_list[12:13]+sub_list[16:17]+sub_list[26:27] #3 2 64
    # sub_list = sub_list[10:11]+sub_list[15:17]+sub_list[20:22]+sub_list[23:24]+sub_list[26:27]+sub_list[12:13]+sub_list[32:34] #3 2 32
    # sub_list = sub_list[1:2]+sub_list[7:8] #3 3 64
    # sub_list = sub_list[11:12] #3 3 32
    # sub_list = sub_list[27:28]+sub_list[31:32]+sub_list[34:35] #3 2 16
    # sub_list = sub_list[9:10] #2 2 32 bottle drop0.5
    sub_list = sub_list[13:14] #3 3 128 bottle drop0.5.
    print(sub_list)
    for sub_index, sub in enumerate(sub_list):
        print(sub)
        acc = main(logger, sub)
        record_file.write(sub + ' ' + str(acc) + '\n')

