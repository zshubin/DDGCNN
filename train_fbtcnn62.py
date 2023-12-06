import sys
from config_sess1_62 import OptInit
from fbtcnn62 import tCNN
from utils.ckpt_util import save_checkpoint
import logging
from cnn_data_loader_62_fb.data_loaders import *


def get_confusion_matrix(ground_truth, prediction, num_classes):
    values = torch.ones(1, dtype=ground_truth.dtype, device=ground_truth.device).expand_as(ground_truth)
    conf_mat = torch.zeros(num_classes, num_classes, dtype=ground_truth.dtype, device=ground_truth.device)
    return conf_mat.index_put_((prediction, ground_truth), values, accumulate=True)


def get_pr(gt, pd, class_num):
    conf_mat = get_confusion_matrix(gt, pd, class_num)
    TP = conf_mat.diagonal()
    FP = conf_mat.sum(1) - TP
    FN = conf_mat.sum(0) - TP
    TP = TP.float()
    FP = FP.float()
    FN = FN.float()
    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    precision = precision.mean()
    recall = recall.mean()
    return precision, recall


def main(logger, sub,sess):
    opt_class = OptInit(sub, logger,sess)
    opt = opt_class.get_args()
    logger.info('===> Creating dataloader ...')
    train_loader, valid_data_loader = \
        data_generator_np(opt.data_dir, opt.batch_size, win_train=opt.time_win, down_sample=opt.down_sample, sample_freq=opt.sample_freq, device=opt.device)
    opt.n_classes = 4

    logger.info('===> Loading the network ...')
    model = tCNN(int(opt.time_win*opt.sample_freq/opt.down_sample)).to(opt.device)
    logger.info('===> Init the optimizer ...')
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=0.01)
    logging.info('===> Init Metric ...')
    opt.vali_loss  = 0.0
    opt.vali_acc   = 0.0
    opt.best_acc   = 0.0
    opt.total_loss = 0.0
    opt.total_acc  = 0.0
    opt.test_precision = 0.0
    opt.test_recall = 0.0

    logging.info('===> Start training ...')
    for e in range(opt.total_epochs):
        opt.epoch += 1
        train(model, train_loader, valid_data_loader, optimizer,  criterion, opt, logger)
        print(opt.best_acc)
        if opt.best_acc == 1. and opt.total_acc > 0.95:
            break

    logger.info('Saving the final model.Finish!')
    return opt.best_acc


def train(model, train_loader, valid_data_loader, optimizer, criterion, opt, logger):
    model.train()
    total_loss = 0.
    total_acc  = 0.
    total_num  = 0.
    opt.iter   = 0.
    for i, (data1, data2, data3, data4, target) in enumerate(train_loader):
        opt.iter += 1
        data1, data2, data3, data4, target = data1.to(opt.device), data2.to(opt.device), data3.to(opt.device), data4.to(opt.device),target.to(opt.device)
        optimizer.zero_grad()
        out = model(data1,data2, data3, data4)
        loss = criterion(out, target.long())
        total_loss += loss
        pred = torch.argmax(out, dim=1)
        total_acc += torch.sum(pred == target).item()
        loss.backward()
        optimizer.step()
        total_num += data1.shape[0]
    opt.total_loss = total_loss/opt.iter
    opt.total_acc = total_acc/total_num
    logger.info('Epoch:{}\t Batch Loss: {}\t Batch Accuracy: {}\t'.format(opt.epoch, opt.total_loss, opt.total_acc))
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
    precision = 0
    recall = 0
    with torch.no_grad():
        for i, (data1, data2, data3, data4, target) in enumerate(test_loader):
            data1, data2, data3, data4, target = data1.to(opt.device), data2.to(opt.device), data3.to(
               opt.device), data4.to(opt.device), target.to(opt.device)
            out = model(data1, data2, data3, data4)
            loss += criterion(out, target.long())
            pred = torch.argmax(out, dim=1)
            correct += torch.sum(pred == target).item()
            total_num += data1.shape[0]
            p, r = get_pr(target.long(), pred, opt.n_classes)
            precision += p
            recall += r
    valid_accuracy = correct / total_num
    valid_loss = loss / (i+1)
    logging.info('Epoch: [{}], Test loss: {}, Test_accuracy: {})\t'.format(opt.epoch, valid_loss, valid_accuracy))
    opt.vali_acc = valid_accuracy
    opt.vali_loss = valid_loss
    opt.test_precision = precision/(i+1)
    opt.test_recall = recall/(i+1)


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
    # data_path = './40targetdata/'
    # sess_list = os.listdir(data_path)
    # logger = make_logger()
    # record_file = open('./record_fbtcnn_1.0_40target.txt', 'w')
    # sub_list = os.listdir(data_path)[7:]
    # for sub_index, sub in enumerate(sub_list):
    #     print(sub)
    #     acc = main(logger, sub)
    #     record_file.write(sub + ' ' + str(acc) + '\n')
    data_path = './processed_ssvep_data/'
    sess_list = os.listdir(data_path)
    logger = make_logger()
    acc_list = []
    for sess_index, sess in enumerate(sess_list):
        if sess_index==0:
            continue
        else:
            start_index = 19
        record_file = open('./record_{}_fbtcnn_0.8.txt'.format(sess), 'w')
        sub_list = os.listdir(os.path.join(data_path, sess))[:start_index]
        for sub_index, sub in enumerate(sub_list):
            print(sub)
            acc = main(logger, sub, sess)
            acc_list.append(acc)
            record_file.write(sub + ' ' + str(acc) + '\n')
            print(acc_list)
