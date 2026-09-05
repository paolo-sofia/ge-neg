import math

import numpy as np
import pygad
import torch

# =====================================================================
# 1. FUNZIONI DI ELABORAZIONE E IMMAGINI SULLA GPU (PYTORCH)
# =====================================================================


def apply_log_logistic_curve_torch(
    img_tensor: torch.Tensor, x0: torch.Tensor, k: torch.Tensor, h: torch.Tensor
) -> torch.Tensor:
    """Applica la curva Log-Logistica con 3 parametri su un batch di immagini in PyTorch.

    - img_tensor: (B, C, H, W)
    - x0, k, h: Tensori di forma (B, 1, 1, 1)
    """
    img_safe = torch.clamp(img_tensor, 1e-6, 1.0 - 1e-6)
    x0_safe = torch.clamp(x0, 1e-3, 0.999)

    x_k = torch.pow(img_safe, k)
    x0_k = torch.pow(x0_safe, k)

    s_shaped = x_k / (x0_k + x_k + 1e-6)

    y_min = 0.0
    y_max = 1.0 / (x0_k + 1.0 + 1e-6)

    normalized = (s_shaped - y_min) / (y_max - y_min + 1e-6)
    img_boosted = torch.pow(torch.clamp(normalized, 1e-6, 1.0), h)

    return torch.clamp(img_boosted, 0.0, 1.0)


def torch_image_entropy(img_batch: torch.Tensor, bins: int = 64) -> torch.Tensor:
    """Calcola l'entropia normalizzata [0, 1] per ogni immagine del batch (B, C, H, W)."""
    B = img_batch.shape[0]
    flattened = img_batch.reshape(B, -1)

    entropies = []
    log2_bins = torch.log2(
        torch.tensor(bins, dtype=torch.float32, device=img_batch.device)
    )

    for i in range(B):
        hist = torch.histc(flattened[i], bins=bins, min=0.0, max=1.0)
        prob = hist / (hist.sum() + 1e-7)
        prob = prob[prob > 0]
        ent = -torch.sum(prob * torch.log2(prob))
        entropies.append(ent / log2_bins)

    return torch.stack(entropies)


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
    img_batch: torch.Tensor, alpha: float = 1.0, beta: float = 1.2, gamma: float = 2.5
) -> torch.Tensor:
    """Calcola la penalità basata sul sistema zonale per un batch di immagini."""
    # Placeholder per la funzione zonale: restituisce 0.0 per ogni elemento del batch
    return torch.zeros(img_batch.shape[0], device=img_batch.device)


# =====================================================================
# 2. EVALUATOR E BATCH FITNESS FUNCTION PER PYGAD
# =====================================================================


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
        self.orig_img = tensor_img.to(self.device, dtype=torch.float32)

    def fitness_batch(
        self, genetic_optimizer: pygad.GA, solutions: np.ndarray, solution_idx: int
    ) -> list[float]:
        """Calcola la fitness per l'INTERA popolazione in parallelo (Batch) su GPU."""
        B = solutions.shape[0]

        # 1. Soluzioni normalizzate -> Tensor GPU
        solutions_tensor = torch.from_numpy(solutions).to(
            self.device, dtype=torch.float32
        )

        norm_x0 = solutions_tensor[:, 0]
        norm_k = solutions_tensor[:, 1]
        norm_h = solutions_tensor[:, 2]

        x0 = (
            self.bounds[0][0] + norm_x0 * (self.bounds[0][1] - self.bounds[0][0])
        ).view(B, 1, 1, 1)
        k = (self.bounds[1][0] + norm_k * (self.bounds[1][1] - self.bounds[1][0])).view(
            B, 1, 1, 1
        )
        h = (self.bounds[2][0] + norm_h * (self.bounds[2][1] - self.bounds[2][0])).view(
            B, 1, 1, 1
        )

        # 2. Espansione immagine per il batch: (1, C, H, W) -> (B, C, H, W)
        orig_batch = self.orig_img.expand(B, -1, -1, -1)

        # 3. Trasformazione vettorializzata dell'immagine
        new_batch = apply_log_logistic_curve_torch(orig_batch, x0, k, h)

        # 4. Metriche di valutazione
        sigmas = torch.std(new_batch, dim=(1, 2, 3))
        sigma_scores = torch.exp(-15.0 * torch.abs(sigmas - 0.21))

        medians = torch.median(new_batch.reshape(B, -1), dim=1).values
        median_scores = torch.exp(-15.0 * torch.abs(medians - 0.48))

        k_penalty = 3.0
        shadows_threshold = 0.05
        highlight_threshold = 0.98

        # Penalità Ombre e Luci
        orig_shadows = torch.mean(
            (orig_batch < shadows_threshold).float(), dim=(1, 2, 3)
        )
        new_shadows = torch.mean((new_batch < shadows_threshold).float(), dim=(1, 2, 3))
        shadow_diff = new_shadows - orig_shadows
        shadow_penalties = (
            torch.exp(k_penalty * torch.clamp(shadow_diff, min=0.0)) - 1.0
        )

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
        entropy_penalties = (
            torch.exp(k_penalty * torch.clamp(entropy_diff, min=0.0)) - 1.0
        )

        # Altre penalità
        zonal_system_penalties = zonal_system_fitness_penalty_torch(
            new_batch, alpha=1.0, beta=1.2, gamma=2.5
        )
        hue_shift_penalties = torch_compute_hue_shift(
            orig_batch, new_batch, self.film_type
        )

        # 5. Fitness Finale per l'intero batch
        fitness_values = (
            sigma_scores
            + median_scores
            - shadow_penalties
            - highlight_penalties
            - entropy_penalties
            - zonal_system_penalties
            - hue_shift_penalties
        )

        return fitness_values.cpu().tolist()


# =====================================================================
# 3. DETEZIONE BORDI VETTORIALIZZATA NUMPY (PER IL PRE-PROCESSING)
# =====================================================================


def analyze_all_slices_at_once(
    img: np.ndarray, limit_px: int, step_size_px: int, direction: str = "bottom"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estrarre e calcolare le metriche di tutte le slice contemporaneamente."""
    num_slices: int = math.ceil(limit_px / step_size_px)
    exact_limit: int = num_slices * step_size_px

    if direction == "left":
        crop = img[:, :exact_limit]
    elif direction == "right":
        crop = img[:, -exact_limit:]
        crop = crop[:, ::-1]
    elif direction == "top":
        crop = img[:exact_limit, :]
    elif direction == "bottom":
        crop = img[-exact_limit:, :]
        crop = crop[::-1, :]
    else:
        raise ValueError(f"Direzione non valida: {direction}")

    has_channels = crop.ndim == 3

    if direction in ("left", "right"):
        height = crop.shape[0]
        if has_channels:
            reshaped = crop.reshape(height, num_slices, step_size_px, crop.shape[2])
            slices_tensor = reshaped.transpose(1, 0, 2, 3)
        else:
            reshaped = crop.reshape(height, num_slices, step_size_px)
            slices_tensor = reshaped.transpose(1, 0, 2)
    else:
        width = crop.shape[1]
        if has_channels:
            reshaped = crop.reshape(num_slices, step_size_px, width, crop.shape[2])
            slices_tensor = reshaped
        else:
            reshaped = crop.reshape(num_slices, step_size_px, width)
            slices_tensor = reshaped

    reduce_axes = tuple(range(1, slices_tensor.ndim))

    means = np.mean(slices_tensor, axis=reduce_axes)
    stds = np.std(slices_tensor, axis=reduce_axes)
    medians = np.median(slices_tensor, axis=reduce_axes)

    return means, stds, medians


def find_edge_by_gradient(
    means: np.ndarray,
    stds: np.ndarray,
    min_gradient_significance: float = 0.05,
) -> int | None:
    """Rileva l'indice del bordo tramite la combinazione dei gradienti di Mean e Std."""
    mean_grad = np.gradient(means)
    abs_mean_grad = np.abs(mean_grad)
    std_grad = np.gradient(stds)

    combined_signal = abs_mean_grad + std_grad

    max_idx = int(np.argmax(combined_signal))
    max_value = combined_signal[max_idx]

    if max_value < min_gradient_significance:
        return None

    return max_idx


# =====================================================================
# 4. ESEMPIO DI ESECUZIONE
# =====================================================================

if __name__ == "__main__":
    # Generazione di un'immagine di prova (H=1000, W=800, C=3) in range 0-1
    test_image = np.random.uniform(0.0, 1.0, (1000, 800, 3)).astype(np.float32)

    # Istanziamento dell'Evaluator PyTorch GPU
    bounds = [(0.3, 0.6), (1.0, 5.0), (0.7, 1.3)]
    evaluator = PyTorchGeneticEvaluator(
        img_np=test_image,
        bounds=bounds,
        film_type="color",
        device="cuda",
    )

    # Configurazione e avvio di PyGAD con il supporto Batch Fitness
    pop_size = 32
    ga_instance = pygad.GA(
        num_generations=50,
        num_parents_mating=8,
        fitness_batch_size=pop_size,
        fitness_func=evaluator.fitness_batch,
        sol_per_pop=pop_size,
        num_genes=3,
        init_range_low=0,
        init_range_high=1,
        mutation_probability=0.2,
    )

    print("Avvio dell'ottimizzazione genetica su PyTorch GPU...")
    ga_instance.run()

    best_solution, best_fitness, _ = ga_instance.best_solution()
    print(f"Miglior Fitness Trovata: {best_fitness:.4f}")
    print(f"Migliori Parametri Normalizzati: {best_solution}")
