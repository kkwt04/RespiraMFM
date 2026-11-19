import torch
import torch.nn as nn
import torch.nn.functional as F
# from nemo.collections.common.data import ConcatDataset
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import yaml
import h5py
from transformers import AutoTokenizer, LlamaModel, AutoModel
from model_utils import get_prompt, Config
from model_projector import ContrastiveProjectionHead, ContrastiveProjectionHead_new
from multiscale import MultiScaleFusion, MultiScaleProjectionModel
from utils import get_dataset_dir, get_subset
from checkpoint_utils import capture_rng_state, parse_feature_paths, restore_rng_state
import os
import argparse

class AudioTextDataset(Dataset):
    def __init__(self, configs, dataset_path, dataset_name, d_llm, suffix= 'none', cache_path="text_embeddings_cache.npy"):
        with h5py.File(dataset_path, "r") as hf:
            self.audio_mode = getattr(configs, "audio_mode", "baseline")
            if self.audio_mode == "multiscale":
                windows_ms = getattr(configs, "multiscale_windows_ms", [20, 100, 400])
                self.audio_embeddings = (
                    hf[f"audio_embeddings_{windows_ms[0]}ms"][:],
                    hf[f"audio_embeddings_{windows_ms[1]}ms"][:],
                    hf[f"audio_embeddings_{windows_ms[2]}ms"][:],
                )
            else:
                self.audio_embeddings = hf["audio_embeddings"][:]
            self.contexts = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in hf["metadata"][:]]
            self.labels = hf["labels"][:]

        self.prompt = get_prompt(configs, dataset=dataset_name, label="covid", modality="cough")

        self.cache_path = f'./cached_files/{dataset_name}_{suffix}_{d_llm}.npy'
        if os.path.exists(self.cache_path):
            print("Loading cached text embeddings...")
            self.text_embeddings = np.load(self.cache_path)
        else:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            print("Generating and caching text embeddings...")
            if d_llm==2048:
                model_id = "meta-llama/Llama-3.2-1B"
            elif d_llm == 3072:
                model_id = "meta-llama/Llama-3.2-3B"
            elif d_llm == 4096:
                model_id = "ContactDoctor/Bio-Medical-Llama-3-8B-CoT-012025"
            elif d_llm==2560:
                model_id = "microsoft/phi-2"
            elif d_llm==1536:
                model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            elif d_llm==1024:
                model_id = "openai-community/gpt2-medium"
            elif d_llm==1280:
                model_id = "openai-community/gpt2-large"
            else:
                raise NotImplementedError('invalid d_llm value : d_llm = ', d_llm)

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token
            # model = LlamaModel.from_pretrained(model_id).eval().cuda()
            model = AutoModel.from_pretrained(model_id).eval().cuda()
            self.text_embeddings = []
            for context in tqdm(self.contexts, desc="Encoding text"):
                with torch.no_grad():
                    tokens = tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=125).to("cuda")
                    output = model(**tokens)
                    text_embed = output.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
                    self.text_embeddings.append(text_embed)
                    # print(text_embed.shape)
            self.text_embeddings = np.array(self.text_embeddings)
            np.save(self.cache_path, self.text_embeddings)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.audio_mode == "multiscale":
            audio_embed = tuple(torch.tensor(item[idx], dtype=torch.float32) for item in self.audio_embeddings)
        else:
            audio_embed = torch.tensor(self.audio_embeddings[idx], dtype=torch.float32)
        text_embed = torch.tensor(self.text_embeddings[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return audio_embed, text_embed, label

def contrastive_loss(audio_proj, text_embed, temperature=0.07):
    audio_proj = F.normalize(audio_proj, dim=1)
    text_embed = F.normalize(text_embed, dim=1)
    symmetric_loss = False
    if symmetric_loss:
        logits_a2t = torch.matmul(audio_proj, text_embed.T) / temperature
        labels = torch.arange(audio_proj.size(0)).to(audio_proj.device)
        loss_a2t = F.cross_entropy(logits_a2t, labels)

        # Backward loss: text → audio
        logits_t2a = torch.matmul(text_embed, audio_proj.T) / temperature
        loss_t2a = F.cross_entropy(logits_t2a, labels)
        loss = (loss_a2t + loss_t2a) / 2
        return loss
    else:
        # print('shape check: ', audio_proj.shape, text_embed.shape)
        logits = torch.matmul(audio_proj, text_embed.T) / temperature
        labels = torch.arange(audio_proj.size(0)).to(audio_proj.device)
        return F.cross_entropy(logits, labels)

def train(model, dataloader, optimizer, device, global_step=0):
    model.train()
    total_loss = 0
    for audio_embed, text_embed, _ in dataloader:
        text_embed = text_embed.to(device)
        if isinstance(audio_embed, (tuple, list)):
            audio_embed = tuple(item.to(device) for item in audio_embed)
            projected = model(*audio_embed)
        else:
            projected = model(audio_embed.to(device))
        loss = contrastive_loss(projected, text_embed)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        global_step += 1
    return total_loss / len(dataloader), global_step

def train_projector_all_dataset(train_datasets, diag_disease, num_epochs=100, learning_rate=0.001,
                                audio_mode="baseline", feature_paths=None, checkpoint_path=None,
                                resume_path=None):
    config_path = './src/config.yaml'
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    configs = Config(configs)
    configs.audio_mode = audio_mode

    if configs.llm_model in ["llama3-8b", "contactdoctor-8b", "openbiollm", "llama3-8b-instruct", "GPT-j"]:
        d_llm = 4096
    elif configs.llm_model in ["llama3-1b", "contactdoctor-1b"]:
        d_llm = 2048
    elif configs.llm_model in ["contactdoctor-3b"]:
        d_llm = 2048
    elif configs.llm_model in ["phi-2"]:
        d_llm = 2560
    elif configs.llm_model in ["deepseek-1.5b"]:
        d_llm = 1536
    elif configs.llm_model in ["GPT2-m"]:
        d_llm = 1024
    elif configs.llm_model == "GPT2-l":
        d_llm = 1280
    elif configs.llm_model == "llama3-3b":
        d_llm = 3072
    else:
        raise NotImplementedError(f'🚨 No d_llm defined for {configs.llm_model}.')

    all_datasets = []
    feature_paths = feature_paths or {}
    for dataset_name in train_datasets:
        dataset_root_dir, model_type, train_path, test_path = get_dataset_dir(dataset_name)
        train_path = feature_paths.get(dataset_name, train_path)
        train_dataset = AudioTextDataset(configs, train_path, dataset_name, d_llm, suffix='train')
        all_datasets.append(train_dataset)
    all_datasets_concat = ConcatDataset(all_datasets)
    data_loader = DataLoader(all_datasets_concat, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aligner = ContrastiveProjectionHead(out_dim=d_llm)
    if getattr(configs, "audio_mode", "baseline") == "multiscale":
        fusion = MultiScaleFusion(
            embedding_dim=configs.enc_dim,
            hidden_dim=configs.multiscale_fusion_hidden,
            dropout=configs.multiscale_fusion_dropout,
        )
        model = MultiScaleProjectionModel(fusion, aligner).to(device)
    else:
        model = aligner.to(device)
    # model = ContrastiveProjectionHead_new(out_dim=d_llm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    projector_saved_path = checkpoint_path or os.environ.get(
        "RESPIRAMFM_ALIGNER_CHECKPOINT",
        f'/local/scratch1/siam/saved_models/acl_2026/contrastive_audio_to_text_{diag_disease}_{d_llm}_final.pth',
    )
    resume_saved_path = projector_saved_path + ".resume.pth"
    start_epoch = 0
    global_step = 0
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_rng_state(checkpoint.get("rng_state"))
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
    os.makedirs(os.path.dirname(projector_saved_path) or ".", exist_ok=True)
    print(f'Projector will be saved at: {projector_saved_path}')
    for epoch in tqdm(range(start_epoch, num_epochs)):
        loss, global_step = train(model, data_loader, optimizer, device, global_step)
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "rng_state": capture_rng_state(),
            "config": vars(configs),
        }, resume_saved_path)
        if num_epochs >=10:
            if (epoch) % (num_epochs//10) == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    if getattr(configs, "audio_mode", "baseline") == "multiscale":
        torch.save({
            "fusion_state_dict": model.fusion.state_dict(),
            "aligner_state_dict": model.aligner.state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "epoch": num_epochs - 1,
            "global_step": global_step,
            "rng_state": capture_rng_state(),
            "config": vars(configs),
        }, projector_saved_path)
    else:
        torch.save(model.state_dict(), projector_saved_path)
    print('✅️ Projector model saved successfully...')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the original Phase 1 contrastive aligner")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--disease", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--audio-mode", choices=("baseline", "multiscale"), default="baseline")
    parser.add_argument("--feature-path", action="append", default=[], help="DATASET=/path/to/train_features.h5")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.audio_mode == "multiscale" and not args.checkpoint_path:
        parser.error("--checkpoint-path is required for multiscale Phase 1")
    train_projector_all_dataset(
        args.datasets, args.disease, num_epochs=args.epochs, learning_rate=args.learning_rate,
        audio_mode=args.audio_mode, feature_paths=parse_feature_paths(args.feature_path),
        checkpoint_path=args.checkpoint_path, resume_path=args.resume,
    )
