import numpy as np
import random
import argparse
import torch
import yaml
import os
import sys
import csv
import json
from torchmetrics.functional import f1_score
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import get_dataset_dir
from model_utils import Config, AudioTextDataset, AudioLLM
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python built-in hash seed
    random.seed(seed)                         # Python random module
    np.random.seed(seed)                      # NumPy
    torch.manual_seed(seed)                   # PyTorch CPU
    torch.cuda.manual_seed(seed)              # PyTorch single GPU
    torch.cuda.manual_seed_all(seed)          # PyTorch multi-GPU
    torch.backends.cudnn.deterministic = True # Enforce deterministic behavior
    torch.backends.cudnn.benchmark = False    # Disable benchmarking to avoid non-deterministic conv algorithms


def evaluate(model, dataloader, dataset_name, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x_audio_filename, x_audio_encoder, x_prompt, x_context, labels in tqdm(dataloader, desc=f"Evaluating {dataset_name}", unit="batch"):
            x_audio_encoder, labels = x_audio_encoder.to(device), labels.to(device)

            # Forward pass
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

    print(f"{dataset_name}: AUC: {roc_auc:.4f}    F1 score: {F1score:.4f}    Accuracy: {accuracy:.4f}")
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy),
        "auroc": float(roc_auc),
        "f1": float(F1score),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "n_samples": len(all_labels),
    }
    return metrics, all_labels, all_preds, all_probs

def main(test_datasets, projector_type, diag_disease='all', checkpoint=None,
         aligner_checkpoint=None, results_dir="results"):
    config_path = './src/config.yaml'
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    configs = Config(configs)
    if aligner_checkpoint:
        configs.aligner_checkpoint = aligner_checkpoint
    print("LLM Model Selected: ", configs.llm_model)

    batch_size = configs.batch_size
    configs.aligner = projector_type

    all_test_loader = []
    all_test_dataset = []
    for dataset_name in test_datasets:
        dataset_root_dir, model_type, train_path, test_path = get_dataset_dir(dataset_name)
        test_dataset = AudioTextDataset(configs, test_path, dataset_name)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        all_test_loader.append(test_loader)
        all_test_dataset.append(test_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioLLM(configs, diag_disease=diag_disease).to(device)
    model_saved_path = checkpoint or f"/local/scratch1/siam/saved_models/acl_2026/final_models/audio_llm_model_{diag_disease}_{configs.llm_model}_{projector_type}_{configs.multimodal_config}_final.pth"
    print("Model will be loaded from: ", model_saved_path)
    checkpoint_state = torch.load(model_saved_path, map_location=device)
    if isinstance(checkpoint_state, dict) and "model_state_dict" in checkpoint_state:
        checkpoint_state = checkpoint_state["model_state_dict"]
    model.load_state_dict(checkpoint_state)

    os.makedirs(results_dir, exist_ok=True)
    results_by_dataset = {}
    prediction_rows = []

    # Evaluation
    for i, test_loader in enumerate(all_test_loader):
        dataset_name = test_datasets[i]
        test_dataset = all_test_dataset[i]
        metrics, all_labels, all_preds, all_probs = evaluate(model, test_loader, dataset_name, device)
        results_by_dataset[dataset_name] = metrics
        for filename, label, prediction, score in zip(test_dataset.audio_filenames, all_labels, all_preds, all_probs):
            prediction_rows.append([dataset_name, filename, label, prediction, score])

    print(results_by_dataset)
    with open(os.path.join(results_dir, "predictions.csv"), mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "filename", "label", "prediction", "score_class_1"])
        writer.writerows(prediction_rows)
    with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump({"datasets": results_by_dataset}, file, indent=2)
    with open(os.path.join(results_dir, "evaluation.log"), "w", encoding="utf-8") as file:
        file.write(json.dumps(results_by_dataset, indent=2) + "\n")

if __name__ == "__main__":
    set_seed(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", default="all")
    parser.add_argument("--projector", default="contrastive")
    parser.add_argument("--datasets", nargs="+", default=["ukcovid19", "coughvid", "TBscreen", "icbhi", "coswara", "codaTB", "kauh_copd", "kauh_asthma", "kauh_pneumonia"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--aligner-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results-dir", default="results/evaluation")
    args = parser.parse_args()

    diag_disease = args.disease
    projector_type = args.projector
    os.environ["RESPIRAMFM_DATA_ROOT"] = args.data_root
    test_datasets = args.datasets
    print(f'Model will be evaluated on {diag_disease} : {test_datasets} dataset')
    main(test_datasets, projector_type, diag_disease=diag_disease,
         checkpoint=args.checkpoint, aligner_checkpoint=args.aligner_checkpoint,
         results_dir=args.results_dir)
