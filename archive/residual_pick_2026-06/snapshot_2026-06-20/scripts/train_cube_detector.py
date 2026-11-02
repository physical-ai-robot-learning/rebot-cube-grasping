"""STEP 2 of the sim-trained cube detector: train CubeDetector (CSP+FPN+heatmap) on the labeled
TOP-CAM dataset from step 1 (scripts/gen_perception_data.py -> data/percep_top.npz).

Online PHOTOMETRIC augmentation (brightness/contrast/gamma/per-channel gain/gaussian-noise/random-erase)
gives the lighting+sensor+occlusion robustness the brittle HSV detector lacked; the geometric variety
(arm/cube/distractor poses) is already baked into the dataset. Loss = heatmap MSE (gaussian target,
masked by visibility) + visibility BCE + a small soft-argmax (u,v) L1 to sharpen localization. The
SAME model runs on sim renders and the real camera -> a clutter/lighting-robust drop-in for HSV that
feeds the policy's (u,v) token (project (b): raw (u,v) in obs, no privileged cube_pos).

    uv run --no-sync python scripts/train_cube_detector.py --data data/percep_top.npz --out ckpt/cube_detector.pt
"""
import os, sys, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/percep_top.npz")
ap.add_argument("--out", default="ckpt/cube_detector.pt")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--val-frac", type=float, default=0.1)
ap.add_argument("--width", type=int, default=32, help="CubeDetector base channels")
ap.add_argument("--sigma", type=float, default=1.5, help="heatmap gaussian sigma (in /stride grid cells)")
ap.add_argument("--w-vis", type=float, default=0.5, help="visibility BCE weight")
ap.add_argument("--w-uv", type=float, default=0.01, help="soft-argmax (u,v) L1 weight (input px)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch
import torch.nn.functional as F
from rebot_sim2real.perception_cnn import CubeDetector, heatmap_target

torch.manual_seed(args.seed)
dev = torch.device(args.device if torch.cuda.is_available() else "cpu")


def photometric_aug(x):
    """x [B,3,H,W] in 0..1 -> augmented, uv-PRESERVING (no geometric change). Vectorized on device."""
    B = x.shape[0]
    r = lambda lo, hi: torch.empty(B, 1, 1, 1, device=x.device).uniform_(lo, hi)
    x = x * r(0.6, 1.4)                                   # brightness
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * r(0.6, 1.4) + m                         # contrast
    x = x.clamp(1e-4, 1.0) ** r(0.7, 1.4)                 # gamma
    gain = torch.empty(B, 3, 1, 1, device=x.device).uniform_(0.85, 1.15)
    x = x * gain                                          # per-channel (color/white-balance) jitter
    x = x + torch.randn_like(x) * r(0.0, 0.04)            # sensor noise
    return x.clamp(0.0, 1.0)


def random_erase(x, uv, vis, p=0.5, max_frac=0.25):
    """Paint random gray rectangles (occlusion robustness). If a box covers the cube label, flip that
    row's visibility to 0 and zero its heatmap (so 'occluded' is a real, labeled state)."""
    B, _, H, W = x.shape
    for i in range(B):
        if torch.rand(1).item() > p:
            continue
        bw = int(torch.empty(1).uniform_(0.08, max_frac).item() * W)
        bh = int(torch.empty(1).uniform_(0.08, max_frac).item() * H)
        x0 = int(torch.randint(0, max(1, W - bw), (1,)).item())
        y0 = int(torch.randint(0, max(1, H - bh), (1,)).item())
        x[i, :, y0:y0 + bh, x0:x0 + bw] = torch.rand(1, device=x.device)
        if vis[i] > 0.5:
            u, v = float(uv[i, 0]), float(uv[i, 1])
            if x0 <= u < x0 + bw and y0 <= v < y0 + bh:
                vis[i] = 0.0
    return x, vis


def main():
    d = np.load(args.data)
    imgs = d["images"]                 # [N,H,W,3] uint8
    uv = d["uv"].astype(np.float32)    # [N,2] input-res px (-1 if not visible)
    vis = d["visible"].astype(np.float32)
    N, H, W = imgs.shape[:3]
    print(f"[train] data={args.data} N={N} img={W}x{H} visible_frac={vis.mean():.3f}", flush=True)

    # to tensors (keep images uint8 on CPU; move per-batch -> fits any GPU). NCHW.
    imgs_t = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).contiguous()  # uint8
    uv_t = torch.from_numpy(uv)
    vis_t = torch.from_numpy(vis)

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)
    n_val = int(N * args.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = CubeDetector(w=args.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # heatmap grid size (forward once)
    with torch.no_grad():
        hm0, _ = model(torch.zeros(1, 3, H, W, device=dev))
    _, _, gh, gw = hm0.shape
    stride = model.stride
    print(f"[train] heatmap grid {gw}x{gh} (stride {stride})  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    def batches(idx, bs, shuffle):
        order = idx[torch.randperm(len(idx))] if shuffle else idx
        for i in range(0, len(order), bs):
            yield order[i:i + bs]

    best_px = 1e9
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for b in batches(tr_idx, args.batch, shuffle=True):
            x = imgs_t[b].to(dev).float().div(255.0)
            u = uv_t[b].to(dev).clone()
            v = vis_t[b].to(dev).clone()
            x = photometric_aug(x)
            x, v = random_erase(x, u, v)
            tgt = heatmap_target(u, gh, gw, stride, sigma=args.sigma) * v.view(-1, 1, 1)
            hm, vlogit = model(x)
            l_hm = F.mse_loss(hm.squeeze(1), tgt)
            l_vis = F.binary_cross_entropy_with_logits(vlogit, v)
            uv_pred = model.uv_from_heatmap(hm)
            mvis = v > 0.5
            l_uv = (uv_pred[mvis] - u[mvis]).abs().mean() if mvis.any() else torch.zeros((), device=dev)
            loss = l_hm + args.w_vis * l_vis + args.w_uv * l_uv
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += float(loss) * len(b)
        sched.step()

        # ---- validation: mean (u,v) pixel error on VISIBLE samples + visibility accuracy (no aug) ----
        model.eval()
        px_err, n_vis, vis_correct = 0.0, 0, 0
        with torch.no_grad():
            for b in batches(val_idx, args.batch, shuffle=False):
                x = imgs_t[b].to(dev).float().div(255.0)
                u = uv_t[b].to(dev); v = vis_t[b].to(dev)
                hm, vlogit = model(x)
                uv_pred = model.uv_from_heatmap(hm)
                mvis = v > 0.5
                if mvis.any():
                    px_err += float(((uv_pred[mvis] - u[mvis]) ** 2).sum(-1).sqrt().sum())
                    n_vis += int(mvis.sum())
                vis_correct += int(((torch.sigmoid(vlogit) > 0.5).float() == v).sum())
        mpx = px_err / max(1, n_vis)
        vacc = vis_correct / max(1, len(val_idx))
        tag = ""
        if mpx < best_px:
            best_px = mpx
            torch.save({"state_dict": model.state_dict(), "width": args.width, "stride": stride,
                        "input_hw": [H, W]}, args.out)
            tag = "  *best -> saved"
        print(f"[train] ep {ep+1:02d}/{args.epochs}  loss={tr_loss/len(tr_idx):.4f}  "
              f"val_px_err={mpx:.2f}px  vis_acc={vacc:.3f}{tag}", flush=True)

    msg = f"[train] DONE best_val_px_err={best_px:.2f}px -> {args.out}"
    print(msg, flush=True)
    with open("/tmp/train_cube_detector.txt", "w") as fh:
        fh.write(msg + "\n"); fh.flush(); os.fsync(fh.fileno())


if __name__ == "__main__":
    main()
