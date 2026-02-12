import torch
from asteroid.models import ConvTasNet, DPRNNTasNet
from tqdm import tqdm
from LossFunction.SI_SNR import si_snr
from DataLoader.WSJDataLoader import MakeDataLoader as MakeWSJLoader


def eval_asteroid_model(model_name, loader, device):
    print(f"Loading {model_name} from Asteroid Hub...")
    # Asteroid 会自动下载 HuggingFace 上的预训练模型
    if model_name == "ConvTasNet":
        model = ConvTasNet.from_pretrained("mpariente/ConvTasNet_WHAM!_sepclean")
    elif model_name == "DPRNN":
        model = DPRNNTasNet.from_pretrained("mpariente/DPRNNTasNet-ks2_WHAM!_sepclean")

    model.to(device)
    model.eval()

    scores = []
    with torch.no_grad():
        for mix, targets in tqdm(loader, desc=f"Eval {model_name}"):
            mix = mix.to(device)
            targets = targets.to(device)

            # Asteroid 模型通常直接接受 (Batch, Time)
            est = model(mix)

            # 计算 Loss (注意 Asteroid 可能输出顺序不同，最好用 PIT 计算)
            # 这里简单复用你的 pit_si_snr_loss 逻辑，或者直接算 si_snr
            # 你的 pit_si_snr_loss 在 trainTasNet.py 里，这里简写一下 PIT 逻辑

            # ... (此处省略 PIT 代码，可以使用你现有的 loss function)
            # 假设你已经 import 了 pit_si_snr_loss
            from LossFunction.SI_SNR import pit_si_snr_loss
            loss = pit_si_snr_loss(est, targets)
            scores.append(-loss.item())  # loss is negative snr

    print(f"Average SI-SNR for {model_name}: {sum(scores) / len(scores):.2f} dB")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = "../../Dataset/WSJ0-2Mix/wsj0_2mix/2speakers/wav8k/min"
    _, _, test_loader = MakeWSJLoader(root=root, batch_size=8, num_workers=4, segment=4.0, sample_rate=8000)

    # 1. 对比标准 ConvTasNet (验证你的实现是否达标)
    eval_asteroid_model("ConvTasNet", test_loader, device)

    # 2. 对比更强的 DPRNN (作为 Advanced Baseline)
    eval_asteroid_model("DPRNN", test_loader, device)