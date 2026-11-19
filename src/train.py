from torchmetrics.functional import f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
import torch
import yaml
import os
import sys
import pandas as pd
from utils import get_dataset_dir, get_subset
from model_utils import Config, AudioTextDataset, AudioLLM
from projector_train_text_audio import train_projector_all_dataset
from checkpoint_utils import capture_rng_state, parse_feature_paths, restore_rng_state
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np
import random
import argparse
import json

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python built-in hash seed
    random.seed(seed)                         # Python random module
    np.random.seed(seed)                      # NumPy
    torch.manual_seed(seed)                   # PyTorch CPU
    torch.cuda.manual_seed(seed)              # PyTorch single GPU
    torch.cuda.manual_seed_all(seed)          # PyTorch multi-GPU
    torch.backends.cudnn.deterministic = True # Enforce deterministic behavior
    torch.backends.cudnn.benchmark = False    # Disable benchmarking to avoid non-deterministic conv algorithms

def train(model, dataloader, optimizer, criterion, device, global_step=0):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch", unit="batch")
    for audio_filenames, x_audio_encoder, x_prompt, x_context, labels in progress_bar:
        if isinstance(x_audio_encoder, (tuple, list)):
            x_audio_encoder = tuple(item.to(device) for item in x_audio_encoder)
        else:
            x_audio_encoder = x_audio_encoder.to(device)
        labels = labels.to(device)
        model_outputs = model(audio_filenames, x_audio_encoder, x_prompt, x_context)
        outputs = model_outputs.logits
        outputs = outputs.squeeze(-1)  
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        global_step += 1
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(dataloader), global_step

def evaluate(model, dataloader, dataset_name, device, configs):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for x_audio_filename, x_audio_encoder, x_prompt, x_context, labels in tqdm(dataloader, desc=f"Evaluating {dataset_name}", unit="batch"):
            if isinstance(x_audio_encoder, (tuple, list)):
                x_audio_encoder = tuple(item.to(device) for item in x_audio_encoder)
            else:
                x_audio_encoder = x_audio_encoder.to(device)
            labels = labels.to(device)
            outputs = model(x_audio_filename, x_audio_encoder, x_prompt, x_context).logits
            probabilities = torch.softmax(outputs, dim=-1)  
            _, predictions = torch.max(outputs, dim=-1)  

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy()[:, 1])  

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    F1score = f1_score(all_labels, all_preds)

    print("LLM Model Selected: ", configs.llm_model)
    print(f"{dataset_name}: AUC: {roc_auc:.4f}    F1 score: {F1score:.4f}    Accuracy: {accuracy:.4f}")
    return accuracy, roc_auc, all_labels, all_probs

def main(train_datasets, test_datasets, diag_disease='all', retrain_projector=True,
         audio_mode="baseline", multiscale_checkpoint=None, train_feature_paths=None,
         test_feature_paths=None, output_dir=None, resume_path=None, num_epochs=None):
    config_path = './src/config.yaml'
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    configs = Config(configs)
    configs.audio_mode = audio_mode
    if multiscale_checkpoint:
        configs.multiscale_checkpoint = multiscale_checkpoint
    print("LLM Model Selected: ", configs.llm_model)

    batch_size = configs.batch_size
    train_feature_paths = train_feature_paths or {}
    test_feature_paths = test_feature_paths or {}

    # Define Training Dataset
    all_train_dataset = []
    for dataset_name in train_datasets:
        dataset_root_dir, model_type, train_path, test_path = get_dataset_dir(dataset_name)
        train_path = train_feature_paths.get(dataset_name, train_path)
        train_dataset = AudioTextDataset(configs, train_path, dataset_name)
        all_train_dataset.append(train_dataset)

    combined_training_dataset = ConcatDataset(all_train_dataset)
    if configs.dataset_subset_fraction < 1:
        combined_training_dataset = get_subset(combined_training_dataset, fraction=configs.dataset_subset_fraction, seed=42)
    train_loader = DataLoader(combined_training_dataset, batch_size=batch_size, shuffle=True)

    all_dataset_projector_train = all_train_dataset
    # Define Test Dataset
    all_test_loader = []
    for dataset_name in test_datasets:
        dataset_root_dir, model_type, train_path, test_path = get_dataset_dir(dataset_name)
        test_path = test_feature_paths.get(dataset_name, test_path)
        test_dataset = AudioTextDataset(configs, test_path, dataset_name)
        if configs.dataset_subset_fraction < 1:
            test_dataset = get_subset(test_dataset, fraction=configs.dataset_subset_fraction,
                                               seed=42)
        all_dataset_projector_train.append(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        all_test_loader.append(test_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if retrain_projector:
        if configs.multimodal_config != 3 and configs.aligner != "projection":
            train_projector_all_dataset(
                train_datasets, diag_disease, num_epochs=500, audio_mode=audio_mode,
                feature_paths=train_feature_paths, checkpoint_path=multiscale_checkpoint,
            )

    model = AudioLLM(configs, diag_disease=diag_disease).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,  
        betas=(0.9, 0.95),  
        eps=1e-8,  
        weight_decay=0.1  
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10000,  
        eta_min=1e-6  
    )
    criterion = torch.nn.CrossEntropyLoss()

    print("🔥 Training Started....")
    num_epoch = num_epochs or 20
    best_avg = 0
    global_step = 0
    start_epoch = 0
    model_output_dir = output_dir or os.environ.get(
        "RESPIRAMFM_MODEL_OUTPUT_DIR", "/local/scratch1/siam/saved_models/acl_2026/respiramfm",
    )
    os.makedirs(model_output_dir, exist_ok=True)
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        restore_rng_state(checkpoint.get("rng_state"))
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        best_avg = checkpoint.get("best_avg", 0)

    for epoch in range(start_epoch, num_epoch):
        print(f"Epoch {epoch + 1} of {num_epoch}")
        current_results_dict = {name: 0 for name in test_datasets}
        train_loss, global_step = train(model, train_loader, optimizer, criterion, device, global_step)
        scheduler.step()

        # Evaluation
        save_flag = audio_mode == "baseline"
        for i, test_loader in enumerate(all_test_loader):
            dataset_name = test_datasets[i]
            acc, roc, all_labels, all_probs = evaluate(model, test_loader, dataset_name, device, configs)
            current_results_dict[dataset_name] = round(roc, 4)

        current_avg = np.mean(list(current_results_dict.values()))
        current_results_dict['epoch'] = epoch
        current_results_dict['train_loss'] = train_loss
        current_results_dict['Agerage'] = current_avg
        if current_avg > best_avg:
            save_flag = True
            best_avg = current_avg

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": capture_rng_state(),
            "best_avg": best_avg,
            "config": vars(configs),
        }
        torch.save(checkpoint, os.path.join(model_output_dir, "phase2_last.resume.pth"))

        if save_flag:
            model_saved_path = os.path.join(
                model_output_dir,
                f"audio_llm_model_{diag_disease}_{configs.llm_model}_{configs.aligner}_{configs.multimodal_config}_{configs.llm_peft}_epoch{epoch}.pth",
            )
            torch.save(model.state_dict(), model_saved_path)
            print(f"Full model with projector saved in directory: {model_saved_path}")

        print(current_results_dict)
        results_log_dir = os.environ.get("RESPIRAMFM_RESULTS_LOG_DIR", "./results_log/final")
        os.makedirs(results_log_dir, exist_ok=True)
        out_save_file_json = os.path.join(
            results_log_dir,
            f"RespiraMFM_{configs.llm_model}_{diag_disease}_{configs.aligner}_{configs.multimodal_config}.jsonl",
        )
        with open(out_save_file_json, 'a') as f:
            json.dump(current_results_dict, f)
            f.write('\n')

if __name__ == "__main__":
    set_seed(seed=420)
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", default="all")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--audio-mode", choices=("baseline", "multiscale"), default="baseline")
    parser.add_argument("--multiscale-checkpoint")
    parser.add_argument("--train-feature-path", action="append", default=[], help="DATASET=/path/to/train_features.h5")
    parser.add_argument("--test-feature-path", action="append", default=[], help="DATASET=/path/to/test_features.h5")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    diag_disease = args.disease
    retrain_projector = args.retrain
    if args.audio_mode == "multiscale" and not args.multiscale_checkpoint:
        parser.error("--multiscale-checkpoint is required for multiscale Phase 2")

    if diag_disease == "covid":
        train_datasets = ["ukcovid19", "coughvid"]
        test_datasets = ["ukcovid19", "coughvid", "coswara"]
    elif diag_disease == "TB":
        train_datasets = ["TBscreen"]
        test_datasets = ["codaTB", "TBscreen"] # TBscreen or codaTB
    elif diag_disease == "copd":
        train_datasets = ["icbhi"]
        test_datasets = ["icbhi", "kauh_copd", "kauh_asthma", "kauh_pneumonia"]
    elif diag_disease == "all":
        train_datasets = ["ukcovid19", "coughvid", "TBscreen", "icbhi"]
        test_datasets = ["ukcovid19", "coughvid", "TBscreen", "icbhi", "coswara", "codaTB",  "kauh_copd", "kauh_asthma", "kauh_pneumonia"]

    print(f'Model Training on {diag_disease} : {train_datasets} dataset')
    print(f'Model will be evaluated on {diag_disease} : {test_datasets} dataset')
    main(train_datasets, test_datasets, diag_disease=diag_disease, retrain_projector=retrain_projector,
         audio_mode=args.audio_mode, multiscale_checkpoint=args.multiscale_checkpoint,
         train_feature_paths=parse_feature_paths(args.train_feature_path),
         test_feature_paths=parse_feature_paths(args.test_feature_path),
         output_dir=args.output_dir, resume_path=args.resume, num_epochs=args.epochs)
