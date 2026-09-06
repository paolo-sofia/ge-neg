import gc

import numpy as np
import pygad
import torch

# =====================================================================
# 1. FUNZIONI DI ELABORAZIONE E IMMAGINI SULLA GPU (PYTORCH)
# =====================================================================


def apply_log_logistic_curve_torch(
    img: torch.Tensor, x0: torch.Tensor, k: torch.Tensor, h: torch.Tensor
) -> torch.Tensor:
    """Implementazione identica alla formula matematica originale senza inversioni di h."""
    img_safe = torch.clamp(img, 1e-6, 1.0 - 1e-6)
    x0_safe = torch.clamp(x0, 1e-3, 0.999)

    # 1. Calcolo potenze base
    x_k = torch.pow(img_safe, k)
    x0_k = torch.pow(x0_safe, k)

    # 2. Curva log-logistica base: s(x)
    s_shaped = x_k / (x0_k + x_k + 1e-6)

    # 3. Normalizzazione min-max su [0, 1]
    # Valore di s_shaped quando x = 1.0
    y_max = 1.0 / (x0_k + 1.0 + 1e-6)

    # Scalatura lineare per garantire che x=1 corrisponda a y=1
    normalized = s_shaped / (y_max + 1e-6)

    # 4. Applicazione diretta dell'esponente h (come in NumPy)
    img_boosted = torch.pow(torch.clamp(normalized, 1e-6, 1.0), h)

    return torch.clamp(img_boosted, 0.0, 1.0)


def torch_image_entropy(
    img_batch: torch.Tensor, normalize: bool = True
) -> torch.Tensor:
    """Calcola l'entropia media sui canali per un BATCH di immagini (B, C, H, W) o (B, H, W, C).

    Returns: Tensor 1D di forma (B,) con l'entropia di ogni immagine nel batch.
    """
    # 1. Uniformiamo la forma a (B, C, H, W)
    if img_batch.ndim == 3:
        img_batch = img_batch.unsqueeze(1)
    elif img_batch.ndim == 4 and img_batch.shape[-1] in (1, 3):
        img_batch = img_batch.permute(0, 3, 1, 2)

    B, C, H, W = img_batch.shape
    num_bins = 256

    flat_images = img_batch.reshape(B, C, -1)

    # 2. Mappatura deterministica sui 256 bin (gestisce sia range 0-1 che 0-255)
    # Se i valori sono in [0, 1], scala per 255.0; se sono già in [0, 255], li mantiene.
    is_float_range = flat_images.detach().abs().max() <= 1.0
    scale = 255.0 if is_float_range else 1.0

    # Arrotondamento corretto a intero per la quantizzazione nei 256 bin
    bin_indices = torch.floor(flat_images * scale).clamp(0, num_bins - 1).long()

    # 3. Calcolo degli offset per separare (Batch, Canale) -> (B, C, 1)
    batch_offsets = (torch.arange(B, device=img_batch.device) * (C * num_bins)).view(
        B, 1, 1
    )
    channel_offsets = (torch.arange(C, device=img_batch.device) * num_bins).view(
        1, C, 1
    )
    offsets = batch_offsets + channel_offsets

    global_indices = (bin_indices + offsets).reshape(-1)

    # 4. Istogramma isolato per ciascun canale di ciascuna immagine
    histograms = torch.zeros(
        B * C * num_bins, device=img_batch.device, dtype=torch.float64
    )
    histograms.scatter_add_(
        0,
        global_indices,
        torch.ones_like(global_indices, dtype=torch.float64),
    )

    histograms = histograms.view(B, C, num_bins)

    # 5. Calcolo della probabilità e dell'entropia
    p = histograms / (H * W)

    # Zero-out per i bin vuoti (p == 0) per evitare NaN con log2
    log_p = torch.zeros_like(p)
    nonzero_mask = p > 0
    log_p[nonzero_mask] = torch.log2(p[nonzero_mask])

    entropy_per_channel = -torch.sum(p * log_p, dim=-1)

    if normalize:
        entropy_per_channel = entropy_per_channel / 8.0  # log2(256)

    # 6. Media sui canali colore -> (B,)
    return torch.mean(entropy_per_channel, dim=-1)


def torch_rgb_to_hue(img: torch.Tensor) -> torch.Tensor:
    """Estrae il canale Hue (0-1) da un tensor RGB (B, C, H, W)."""
    r, g, b = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
    max_c, _ = torch.max(img[:, :3, :, :], dim=1)
    min_c, _ = torch.min(img[:, :3, :, :], dim=1)
    chroma = max_c - min_c + 1e-7

    hue = torch.zeros_like(max_c)

    mask_r = (max_c == r) & (chroma > 1e-5)
    hue[mask_r] = ((g[mask_r] - b[mask_r]) / chroma[mask_r]) % 6.0

    mask_g = (max_c == g) & (chroma > 1e-5)
    hue[mask_g] = ((b[mask_g] - r[mask_g]) / chroma[mask_g]) + 2.0

    mask_b = (max_c == b) & (chroma > 1e-5)
    hue[mask_b] = ((r[mask_b] - g[mask_b]) / chroma[mask_b]) + 4.0

    return hue / 6.0


def torch_compute_hue_shift(
    orig_img: torch.Tensor, new_img: torch.Tensor, film_type: str
) -> torch.Tensor:
    """Calcola la deviazione media della tonalità su PyTorch per l'intero batch."""
    if film_type == "bw" or orig_img.shape[1] < 3:
        return torch.zeros(orig_img.shape[0], device=orig_img.device)

    orig_hue = torch_rgb_to_hue(orig_img)
    new_hue = torch_rgb_to_hue(new_img)

    diff = torch.abs(orig_hue - new_hue)
    hue_dist = torch.minimum(diff, 1.0 - diff)

    return torch.mean(hue_dist, dim=(-2, -1))


def zonal_system_fitness_penalty_torch(
    img_batch: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.5,
    gamma: float = 3.0,
    tau_ext: float = 0.015,
) -> torch.Tensor:
    """Traduzione vettorializzata 1:1 della funzione NumPy 'zonal_system_fitness_penalty'.

    Accetta: Tensore (B, C, H, W) oppure (B, H, W) in range [0.0, 1.0].
    Restituisce: Tensore 1D di forma (B,) con le penalità per ciascun elemento del
    batch.
    """
    B = img_batch.shape[0]
    num_bins = 11

    # Flatten dello spazio spaziale/canali per ogni elemento del batch -> (B, N)
    flat_img = img_batch.reshape(B, -1)
    N = flat_img.shape[1]

    # 1. Discretizzazione nei 11 bin zonali [0..10]
    # Mappiamo i valori [0.0, 1.0] su indici interi da 0 a 10
    bin_indices = torch.floor(flat_img * num_bins).clamp(0, num_bins - 1).long()

    # Offset per calcolare gli istogrammi di tutto il batch in parallelo su GPU
    batch_offsets = (torch.arange(B, device=img_batch.device) * num_bins).unsqueeze(1)
    global_indices = (bin_indices + batch_offsets).reshape(-1)

    # Accumulo istogramma zonale globale (B * 11)
    histograms = torch.zeros(B * num_bins, device=img_batch.device, dtype=torch.float32)
    histograms.scatter_add_(
        0,
        global_indices,
        torch.ones_like(global_indices, dtype=torch.float32),
    )

    # Reshape all'istogramma zonale del batch -> (B, 11)
    p = histograms.view(B, num_bins) / float(N)

    # 2. Termine 1: Smoothness (Differenze tra zone adiacenti adiacenti) -> p[k+1] - p[k]
    diffs = p[:, 1:] - p[:, :-1]  # Forma (B, 10)
    l_smooth = torch.sum(diffs**2, dim=1)  # Forma (B,)

    # 3. Termine 2: Concentrazione (Indice HHI) -> sum(p^2)
    l_conc = torch.sum(p**2, dim=1)  # Forma (B,)

    # 4. Termine 3: Ancoraggio alle estremità (Zone 0 e Zone 10)
    p_zone_0_excess = torch.clamp(p[:, 0] - tau_ext, min=0.0)
    p_zone_10_excess = torch.clamp(p[:, 10] - tau_ext, min=0.0)
    l_ext = p_zone_0_excess + p_zone_10_excess  # Forma (B,)

    # 5. Penalità totale combinata (Saturazione esponenziale)
    unweighted_score = l_smooth + l_conc + l_ext
    total_penalty = 1.0 - torch.exp(-unweighted_score)

    return total_penalty  # (B,)


# =====================================================================
# 2. EVALUATOR E BATCH FITNESS FUNCTION PER PYGAD
# =====================================================================
def fitness_batch(
    img: torch.Tensor,
    batch_size: int,
    x0: torch.Tensor,
    k: torch.Tensor,
    h: torch.Tensor,
    film_type: str,
    verbose=True,
) -> list[float]:
    torch.cuda.empty_cache()
    # 2. Espansione immagine per il batch: (1, C, H, W) -> (B, C, H, W)
    orig_batch = img.expand(batch_size, -1, -1, -1)

    # 3. Trasformazione vettorializzata dell'immagine
    new_batch = apply_log_logistic_curve_torch(orig_batch, x0, k, h)

    # 4. Metriche di valutazione
    sigmas = torch.std(new_batch, dim=(1, 2, 3))
    sigma_scores = torch.exp(-15.0 * torch.abs(sigmas - 0.21))

    medians = torch.median(new_batch.reshape(batch_size, -1), dim=1).values
    median_scores = torch.exp(-15.0 * torch.abs(medians - 0.48))

    k_penalty = 5.0
    shadows_threshold = 0.05
    highlight_threshold = 0.98

    # Penalità Ombre e Luci
    orig_shadows = torch.mean((orig_batch < shadows_threshold).float(), dim=(1, 2, 3))
    new_shadows = torch.mean((new_batch < shadows_threshold).float(), dim=(1, 2, 3))
    shadow_diff = new_shadows - orig_shadows
    shadow_penalties = torch.exp(k_penalty * torch.clamp(shadow_diff, min=0.0)) - 1.0

    orig_highlights = torch.mean(
        (orig_batch > highlight_threshold).float(), dim=(1, 2, 3)
    )
    new_highlights = torch.mean(
        (new_batch > highlight_threshold).float(), dim=(1, 2, 3)
    )
    highlight_diff = new_highlights - orig_highlights
    highlight_penalties = (
        torch.exp(k_penalty * torch.clamp(highlight_diff, min=0.0)) - 1.0
    )

    # Penalità Entropia
    orig_entropy = torch_image_entropy(orig_batch)
    new_entropy = torch_image_entropy(new_batch)
    entropy_diff = orig_entropy - new_entropy
    entropy_penalties = torch.exp(k_penalty * torch.clamp(entropy_diff, min=0.0)) - 1.0

    # Altre penalità
    zonal_system_penalties = zonal_system_fitness_penalty_torch(
        new_batch, alpha=1.0, beta=1.2, gamma=2.5
    )
    hue_shift_penalties = torch_compute_hue_shift(orig_batch, new_batch, film_type)

    # 1. Calcolo del 1° e 99° percentile dell'immagine per ogni elemento del batch
    flat_batch = new_batch.reshape(batch_size, -1)
    p01 = torch.quantile(flat_batch, 0.01, dim=1)
    p99 = torch.quantile(flat_batch, 0.99, dim=1)

    # 2. Score per l'ancoraggio del punto di nero e di bianco
    black_point_scores = torch.exp(
        -k_penalty * torch.abs(p01 - shadows_threshold)
    )  # Spinge le ombre profonde a sfiorare 0.03
    white_point_scores = torch.exp(
        -k_penalty * torch.abs(p99 - highlight_threshold)
    )  # Spinge le luci alte a sfiorare 0.96

    # 5. Fitness Finale per l'intero batch
    fitness_values = (
        sigma_scores
        + median_scores
        + black_point_scores
        + white_point_scores
        - shadow_penalties
        - highlight_penalties
        - entropy_penalties
        - zonal_system_penalties
        - hue_shift_penalties
    )
    if verbose:
        argmax = fitness_values.argmax()
        print(f"""Fitness evaluated using pytorch
x0, k, h:               {torch.round(x0[argmax], decimals=4).cpu().tolist()}-{torch.round(k[argmax], decimals=4).cpu().tolist()}-{torch.round(h[argmax], decimals=4).cpu().tolist()}
fitness_values:         {torch.round(fitness_values[argmax], decimals=4).cpu().tolist()}
sigma_scores:           {torch.round(sigma_scores[argmax], decimals=4).cpu().tolist()}
median_scores:          {torch.round(median_scores[argmax], decimals=4).cpu().tolist()}
black_point_scores:     {torch.round(black_point_scores[argmax], decimals=4).cpu().tolist()}
white_point_scores:     {torch.round(white_point_scores[argmax], decimals=4).cpu().tolist()}
shadow_penalties:       {torch.round(shadow_penalties[argmax], decimals=4).cpu().tolist()}
highlight_penalties:    {torch.round(highlight_penalties[argmax], decimals=4).cpu().tolist()}
entropy_penalties:      {torch.round(entropy_penalties[argmax], decimals=4).cpu().tolist()}
zonal_system_penalties: {torch.round(zonal_system_penalties[argmax], decimals=4).cpu().tolist()}
hue_shift_penalties:    {torch.round(hue_shift_penalties[argmax], decimals=4).cpu().tolist()}
            """)
        print("=" * 150)

    del new_batch, orig_batch, flat_batch
    torch.cuda.empty_cache()
    gc.collect()

    return fitness_values.cpu().tolist()


class PyTorchGeneticEvaluator:
    def __init__(
        self,
        img_np: np.ndarray,
        bounds: list[list[float]],
        film_type: str = "color",
        device: str = "cuda",
    ):
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.bounds = bounds
        self.film_type = film_type

        # Convertiamo l'immagine (H, W, C) in Tensor (1, C, H, W) e carichiamo in memoria GPU
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=-1)

        tensor_img = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        self.orig_img = tensor_img.to(self.device, dtype=torch.float64)

    @torch.no_grad()
    def fitness_batch(
        self, genetic_optimizer: pygad.GA, solutions: np.ndarray, solution_idx: int
    ) -> list[float]:
        """Calcola la fitness per l'INTERA popolazione in parallelo (Batch) su GPU."""
        B = solutions.shape[0]

        # 1. Soluzioni normalizzate -> Tensor GPU
        solutions_tensor = torch.from_numpy(solutions).to(
            self.device, dtype=torch.float64
        )

        norm_x0 = solutions_tensor[:, 0].view(-1, 1, 1, 1)
        norm_k = solutions_tensor[:, 1].view(-1, 1, 1, 1)
        norm_h = solutions_tensor[:, 2].view(-1, 1, 1, 1)

        x0 = self.bounds[0][0] + norm_x0 * (self.bounds[0][1] - self.bounds[0][0])
        k = self.bounds[1][0] + norm_k * (self.bounds[1][1] - self.bounds[1][0])
        h = self.bounds[2][0] + norm_h * (self.bounds[2][1] - self.bounds[2][0])

        return fitness_batch(
            self.orig_img,
            batch_size=B,
            x0=x0,
            k=k,
            h=h,
            film_type=self.film_type,
            verbose=False,
        )
