import argparse
import math
from datetime import datetime
#import h5pyprovider
import numpy as np
import tensorflow.compat.v1 as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR) # model
sys.path.append(ROOT_DIR) # provider
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
sys.path.append(os.path.join(ROOT_DIR, 'data_prep'))
import scannet_dataset

tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1, help='GPU to use; set -1 for CPU [default: CPU]')
parser.add_argument('--model', default='model', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--data_root', default=None, help='ScanNet data root [default: data/scannet_data_pointnet2]')
parser.add_argument('--num_classes', type=int, default=4, help='Number of semantic classes [default: 4]')
parser.add_argument('--eval_interval', type=int, default=5, help='Evaluate every N epochs [default: 5]')
parser.add_argument('--save_interval', type=int, default=10, help='Save latest checkpoint every N epochs [default: 10]')
parser.add_argument('--max_train_batches', type=int, default=None, help='Limit train batches for smoke tests')
parser.add_argument('--max_eval_batches', type=int, default=None, help='Limit eval batches for smoke tests')
parser.add_argument('--whole_scene_eval', action='store_true', help='Also run whole-scene evaluation')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_FEATURES = 4
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
EVAL_INTERVAL = FLAGS.eval_interval
SAVE_INTERVAL = FLAGS.save_interval
MAX_TRAIN_BATCHES = FLAGS.max_train_batches
MAX_EVAL_BATCHES = FLAGS.max_eval_batches

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (__file__, LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = FLAGS.num_classes
CLASS_NAMES = ['not_glass', 'glass', 'ground', 'interior']
if NUM_CLASSES > len(CLASS_NAMES):
    CLASS_NAMES += ['class_%d' % idx for idx in range(len(CLASS_NAMES), NUM_CLASSES)]
CLASS_NAMES = CLASS_NAMES[:NUM_CLASSES]

# Dataset splits are created by the manifest scripts.
DATA_PATH = FLAGS.data_root or os.path.join(ROOT_DIR,'data','scannet_data_pointnet2')
TRAIN_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train')
TEST_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test')
VAL_SPLIT_PATH = os.path.join(DATA_PATH, 'scannet_val.pickle')
if os.path.exists(VAL_SPLIT_PATH):
    VAL_DATASET = scannet_dataset.ScannetDataset(root=DATA_PATH, npoints=NUM_POINT, split='val')
    VAL_SPLIT_NAME = 'val'
else:
    VAL_DATASET = TEST_DATASET
    VAL_SPLIT_NAME = 'test'
TEST_DATASET_WHOLE_SCENE = None
if FLAGS.whole_scene_eval:
    TEST_DATASET_WHOLE_SCENE = scannet_dataset.ScannetDatasetWholeScene(root=DATA_PATH, npoints=NUM_POINT, split='test')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def update_confusion_matrix(confusion, pred, label, valid):
    pred = pred[valid].astype(np.int64)
    label = label[valid].astype(np.int64)
    in_range = (label >= 0) & (label < NUM_CLASSES) & (pred >= 0) & (pred < NUM_CLASSES)
    label = label[in_range]
    pred = pred[in_range]
    if label.size == 0:
        return
    encoded = label * NUM_CLASSES + pred
    confusion += np.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def log_segmentation_metrics(split_name, confusion):
    eps = 1e-12
    tp = np.diag(confusion).astype(np.float64)
    gt_count = np.sum(confusion, axis=1).astype(np.float64)
    pred_count = np.sum(confusion, axis=0).astype(np.float64)
    union = gt_count + pred_count - tp

    per_class_acc = np.divide(tp, gt_count + eps)
    precision = np.divide(tp, pred_count + eps)
    recall = np.divide(tp, gt_count + eps)
    f1 = np.divide(2.0 * precision * recall, precision + recall + eps)
    iou = np.divide(tp, union + eps)

    present = gt_count > 0
    mean_class_acc = np.mean(per_class_acc[present]) if np.any(present) else 0.0
    mean_precision = np.mean(precision[present]) if np.any(present) else 0.0
    mean_recall = np.mean(recall[present]) if np.any(present) else 0.0
    mean_f1 = np.mean(f1[present]) if np.any(present) else 0.0
    mean_iou = np.mean(iou[present]) if np.any(present) else 0.0

    log_string('%s mean class accuracy: %f' % (split_name, mean_class_acc))
    log_string('%s mean precision: %f' % (split_name, mean_precision))
    log_string('%s mean recall: %f' % (split_name, mean_recall))
    log_string('%s mean F1: %f' % (split_name, mean_f1))
    log_string('%s mean IoU: %f' % (split_name, mean_iou))
    log_string('%s per-class metrics:' % split_name)
    for class_idx in range(NUM_CLASSES):
        log_string(
            '  class %d %s: count=%d precision=%f recall=%f f1=%f iou=%f accuracy=%f'
            % (
                class_idx,
                CLASS_NAMES[class_idx],
                int(gt_count[class_idx]),
                precision[class_idx],
                recall[class_idx],
                f1[class_idx],
                iou[class_idx],
                per_class_acc[class_idx],
            )
        )
    log_string('%s confusion matrix rows=true cols=pred:' % split_name)
    for class_idx in range(NUM_CLASSES):
        row = ' '.join(str(int(value)) for value in confusion[class_idx])
        log_string('  class %d %s: %s' % (class_idx, CLASS_NAMES[class_idx], row))

    return {
        'mean_class_acc': mean_class_acc,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1,
        'mean_iou': mean_iou,
        'per_class_iou': iou,
        'per_class_f1': f1,
    }

def train():
    with tf.Graph().as_default():
        device_name = '/cpu:0' if GPU_INDEX < 0 else '/gpu:' + str(GPU_INDEX)
        with tf.device(device_name):
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, smpws_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'val'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        #sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
           'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_metric = -1
        best_model_path = None
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            if EVAL_INTERVAL > 0 and epoch % EVAL_INTERVAL == 0:
                metric = eval_one_epoch(sess, ops, val_writer, VAL_DATASET, VAL_SPLIT_NAME)
                if FLAGS.whole_scene_eval:
                    metric = eval_whole_scene_one_epoch(sess, ops, val_writer, VAL_DATASET, VAL_SPLIT_NAME)
                if metric > best_metric:
                    best_metric = metric
                    best_model_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                    log_string("Best validation mIoU improved to %f" % best_metric)
                    log_string("Model saved in file: %s" % best_model_path)

            # Save the variables to disk.
            if SAVE_INTERVAL > 0 and epoch % SAVE_INTERVAL == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

        if best_model_path is not None:
            log_string('Restoring best validation checkpoint: %s' % best_model_path)
            saver.restore(sess, best_model_path)
        log_string('---- FINAL TEST EVALUATION ----')
        eval_one_epoch(sess, ops, test_writer, TEST_DATASET, 'test', increment_epoch=False)

def get_batch_wdp(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, NUM_FEATURES))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps,seg,smpw = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
        batch_smpw[i,:] = smpw

        dropout_ratio = np.random.random()*0.875 # 0-0.875
        drop_idx = np.where(np.random.random((ps.shape[0]))<=dropout_ratio)[0]
        batch_data[i,drop_idx,:] = batch_data[i,0,:]
        batch_label[i,drop_idx] = batch_label[i,0]
        batch_smpw[i,drop_idx] *= 0
    return batch_data, batch_label, batch_smpw

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, NUM_FEATURES))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps,seg,smpw = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
        batch_smpw[i,:] = smpw
    return batch_data, batch_label, batch_smpw

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET) // BATCH_SIZE
    if MAX_TRAIN_BATCHES is not None:
        num_batches = min(num_batches, MAX_TRAIN_BATCHES)
    
    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_label, batch_smpw = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx)
        # Augment batched point clouds by rotation
        aug_data = provider.rotate_point_cloud_z(batch_data)
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']:batch_smpw,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0

# evaluate on a split
def eval_one_epoch(sess, ops, summary_writer, dataset, split_name, increment_epoch=True):
    global EPOCH_CNT
    is_training = False
    eval_idxs = np.arange(0, len(dataset))
    num_batches = len(dataset) // BATCH_SIZE
    if MAX_EVAL_BATCHES is not None:
        num_batches = min(num_batches, MAX_EVAL_BATCHES)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d %s EVALUATION ----' % (EPOCH_CNT, split_name.upper()))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data, batch_label, batch_smpw = get_batch(dataset, eval_idxs, start_idx, end_idx)
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        summary_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        valid = batch_smpw > 0
        total_correct += np.sum((pred_val == batch_label) & valid)
        total_seen += np.sum(valid)
        loss_sum += loss_val
        update_confusion_matrix(confusion, pred_val, batch_label, valid)

    mean_loss = loss_sum / float(max(num_batches, 1))
    accuracy = total_correct / float(max(total_seen, 1))
    log_string('%s mean loss: %f' % (split_name, mean_loss))
    log_string('%s point accuracy: %f' % (split_name, accuracy))
    metrics = log_segmentation_metrics(split_name, confusion)
    if increment_epoch:
        EPOCH_CNT += 1
    return metrics['mean_iou']


# evaluate on whole scenes to generate numbers provided in the paper
def eval_whole_scene_one_epoch(sess, ops, summary_writer, dataset, split_name):
    # Use the same lightweight metric as chopped-scene eval.
    return eval_one_epoch(sess, ops, summary_writer, dataset, split_name)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
