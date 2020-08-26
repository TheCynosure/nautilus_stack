from model import FullNet, EmbeddingNet, LCCNet, DistanceNet, StructuredEmbeddingNet, ScanMatchNet, ScanConvNet, ScanTransformNet, ScanSingleConvNet, ScanUncertaintyNet
from data_processing.dataset import LCTripletDataset, LCCDataset, LCTripletStructuredDataset, LCLaserDataset, MergedDataset, LUDataset
import time
import torch
import os

log_file = None
def initialize_logging(start_time, file_prefix='train_'):
    global log_file
    log_file = open('./logs/' + file_prefix + start_time + '.log', 'w+')

def print_output(*args):
    print(args)
    if log_file:
        log_file.write(' '.join([str(a) for a in args]) + '\n')
        log_file.flush()
    else:
        print("warning: log file not initialized. Please call initialize_logging")

def close_logging():
    global log_file
    log_file.close()
    log_file = None

def load_structured_dataset(root, split, distance_cache, exhaustive=False, evaluation=False, threshold=0.5):
    print_output("Loading data into memory...", )
    dataset = LCTripletStructuredDataset(
        root=root,
        split=split,
        threshold=threshold)
    dataset.load_data()
    dataset.load_distances(distance_cache)
    if exhaustive:
        dataset.load_all_triplets()
    else:
        dataset.load_triplets()

    if dataset.computed_new_distances:
        dataset.cache_distances()
    print_output("Finished loading data.")
    return dataset

def load_dataset(root, split, distance_cache, exhaustive=False, evaluation=False):
    print_output("Loading data into memory...", )
    dataset = LCTripletDataset(
        root=root,
        split=split,
        evaluation=True)
    dataset.load_data()
    dataset.load_distances(distance_cache)
    if exhaustive:
        dataset.load_all_triplets()
    else:
        dataset.load_triplets()

    if dataset.computed_new_distances:
        dataset.cache_distances()
    print_output("Finished loading data.")
    return dataset

def load_lcc_dataset(root, timestamps):
    print_output("Loading data into memory...")
    dataset = LCCDataset(root=root, timestamps=timestamps)
    print_output("Finished loading data.")
    return dataset

def load_laser_dataset(config, distance_cache=None, use_overlap=False):
    print_output("Loading data into memory...", )
    dataset = LCLaserDataset(config, use_overlap)
    if use_overlap:
        dataset.load_distances(distance_cache)
    dataset.load_data()
    if use_overlap:
        dataset.cache_distances()
    print_output("Finished loading data.")
    return dataset

def load_uncertainty_dataset(bag_file, stats_file):
    print_output("Loading data into memory...", )
    dataset = LUDataset(bag_file, stats_file)
    dataset.load_data()
    print_output("Finished loading data.")
    return dataset

def load_merged_laser_dataset(config, distance_cache=None, use_overlap=False):
    print_output("Loading data into memory...", )
    datasets = []
    for bag_file in config.bag_files:
        dataset = LCLaserDataset(config)
        if use_overlap:
            dataset.load_distances(distance_cache)
            dataset.load_data()
            dataset.cache_distances()
        datasets.append(dataset)
    
    merged = MergedDataset(datasets, name)

    print_output("Finished loading data.")
    return merged

def create_embedder(embedding_model=''):
    embedder = EmbeddingNet()
    if embedding_model != '':
        embedder.load_state_dict(torch.load(embedding_model))
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        embedder = torch.nn.DataParallel(embedder)

    embedder.cuda()
    return embedder

def create_structured_embedder(embedding_model=''):
    embedder = StructuredEmbeddingNet()
    if embedding_model != '':
        embedder.load_state_dict(torch.load(embedding_model))
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        embedder = torch.nn.DataParallel(embedder)

    embedder.cuda()
    return embedder

def create_distance_model(embedding_model='', model=''):
    embedder = EmbeddingNet()
    if embedding_model != '':
        embedder.load_state_dict(torch.load(embedding_model))

    distanceModel = DistanceNet(embedder)

    if model != '':
        distanceModel.load_state_dict(torch.load(model))

    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        distanceModel = torch.nn.DataParallel(distanceModel)

    distanceModel.cuda()
    return distanceModel

def create_classifier(embedding_model='', model=''):
    embedder = EmbeddingNet()
    if embedding_model != '':
        embedder.load_state_dict(torch.load(embedding_model))
    classifier = FullNet(embedder)

    if model != '':
        classifier.load_state_dict(torch.load(model))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        classifier = torch.nn.DataParallel(classifier)

    classifier.cuda()
    return classifier

def create_lcc(embedding_model='', model=''):
    embedder = EmbeddingNet()
    if embedding_model != '':
        embedder.load_state_dict(torch.load(embedding_model))

    lcc = LCCNet(embedder)

    if model != '':
        lcc.load_state_dict(torch.load(model))
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        lcc = torch.nn.DataParallel(lcc)
    
    lcc.cuda()
    return lcc
    
def create_lu_networks(model_dir, model_epoch):
    scan_conv = ScanSingleConvNet()
    if model_dir:
        scan_conv.load_state_dict(torch.load(os.path.join(model_dir, 'model_conv_' + model_epoch + '.pth')))

    scan_uncertainty = ScanUncertaintyNet()
    if model_dir:
        scan_uncertainty.load_state_dict(torch.load(os.path.join(model_dir, 'model_uncertainty_' + model_epoch + '.pth')))

    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        scan_conv = torch.nn.DataParallel(scan_conv)
        scan_uncertainty = torch.nn.DataParallel(scan_uncertainty)

    scan_conv.cuda()
    scan_uncertainty.cuda()
    return scan_conv, scan_uncertainty

def create_laser_networks(model_dir, model_epoch, multi_gpu=True):
    scan_conv = ScanConvNet()
    if model_dir:
        scan_conv.load_state_dict(torch.load(os.path.join(model_dir, 'model_conv_' + model_epoch + '.pth')))

    scan_transform = ScanTransformNet()
    if model_dir:
        transform_path = os.path.join(model_dir, 'model_transform_' + model_epoch + '.pth')
        if os.path.exists(transform_path):
            scan_transform.load_state_dict(torch.load(transform_path))
        else:
            print("Warning: no `transform` network found for provided model_dir and epoch")

    scan_match = ScanMatchNet()
    if model_dir:
        scan_match.load_state_dict(torch.load(os.path.join(model_dir, 'model_match_' + model_epoch + '.pth')))
    
    if multi_gpu and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        scan_conv = torch.nn.DataParallel(scan_conv)
        scan_match = torch.nn.DataParallel(scan_match)
        scan_transform = torch.nn.DataParallel(scan_transform)

    scan_conv.cuda()
    scan_match.cuda()
    scan_transform.cuda()
    return scan_conv, scan_match, scan_transform

def save_model(model, outf, epoch, name_prefix=''):
    to_save = model
    if isinstance(model, torch.nn.DataParallel):
        to_save = model.module
    torch.save(to_save.state_dict(), '%s/model_%s_%d.pth' % (outf, name_prefix, epoch))

def get_model_type(model):
    model_to_check = model
    if isinstance(model, torch.nn.DataParallel):
        model_to_check = model.module
    
    if isinstance(model_to_check, EmbeddingNet) or isinstance(model_to_check, StructuredEmbeddingNet):
        return "embedder"
    elif isinstance(model_to_check, FullNet):
        return "full"
    else:
        raise Exception('Unexpected model', model_to_check)

def get_predictions_for_model(model, clouds, similar, distant, threshold=None):
    model_type = get_model_type(model)
    if model_type == 'embedder':
        distances = get_distances_for_model(model, clouds, similar, distant)
        predictions = (distances < threshold).int()
    elif model_type == 'full':
        scores = get_distances_for_model(model, clouds, similar, distant)
        predictions = torch.argmax(scores, dim=1).cpu()

    return predictions

def get_distances_for_model(model, clouds, similar, distant):
    model_type = get_model_type(model)

    if model_type == 'embedder':
        model_to_check = model
        if isinstance(model, torch.nn.DataParallel):
            model_to_check = model.module

        if isinstance(model_to_check, EmbeddingNet):
            anchor_embeddings, _, _ = model(clouds)
            similar_embeddings, _, _ = model(similar)
            distant_embeddings, _, _ = model(distant)
        elif isinstance(model_to_check, StructuredEmbeddingNet):
            anchor_embeddings = model(clouds[0], clouds[1])
            similar_embeddings = model(similar[0], similar[1])
            distant_embeddings = model(distant[0], distant[1])

        distance_pos = torch.norm(anchor_embeddings - similar_embeddings, p=2, dim=1)
        distance_neg = torch.norm(anchor_embeddings - distant_embeddings, p=2, dim=1)
        
        distances = torch.cat([distance_pos, distance_neg])
        return distances
    elif model_type == 'full':
        scores, _, _ = model(torch.cat([clouds, clouds], dim=0), torch.cat([similar, distant], dim=0))
        
        return scores

def update_metrics(metrics, predictions, labels):
    for i in range(len(predictions)):
        label = labels[i].item()
        prediction = predictions[i].item()
        if label and prediction:
            metrics[0] += 1 # True Positive
        elif not label and not prediction:
            metrics[1] += 1 # True Negative
        elif not label and prediction:
            metrics[2] += 1 # False Positive
        elif label and not prediction:
            metrics[3] += 1 # False Negative
        else:
            raise Exception('This is bad')
