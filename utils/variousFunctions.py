import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
from typing import Union
from pathlib import Path
from PIL import Image


def calcRSquared(yTrue, yPred):
    """
    :param yTrue: The true/validation value of the function that we test
    :param yPred: The prediction values for the function that our model/algorithm gives
    :return:
    """
    RSquared = 1 - torch.div(
        torch.sum(torch.linalg.norm(torch.reshape(yTrue - yPred, [yTrue.size(0), -1]), dim=-1) ** 2, dim=0),
        torch.sum(torch.linalg.norm(torch.reshape(yTrue - torch.mean(yTrue, dim=0), [yTrue.size(0), -1]), dim=-1) ** 2,
                  dim=0))
    return RSquared

def calcEpsilon(yTrue, yPred):
    """
    :param yTrue: The true/validation value of the function that we test
    :param yPred: The prediction values for the function that our model/algorithm gives
    :return:
    """
    epsilon = 1/yTrue.size(0) * torch.sum(torch.div(
        torch.linalg.norm(torch.reshape(yTrue - yPred, [yTrue.size(0), -1]), dim=-1),
        torch.linalg.norm(torch.reshape(yTrue, [yTrue.size(0), -1]), dim=-1)), dim=0)
    return epsilon

def calcEpsilon(yTrue, yPred):
    """
    :param yTrue: The true/validation value of the function that we test
    :param yPred: The prediction values for the function that our model/algorithm gives
    :return:
    """
    if len(yTrue.size()) < 3:
        yTrue = yTrue.unsqueeze(0)
    if len(yPred.size()) < 3:
        yPred = yPred.unsqueeze(0)
    epsilon = 1/yTrue.size(0) * torch.sum(torch.div(
        torch.linalg.norm(torch.reshape(yTrue - yPred, [yTrue.size(0), -1]), dim=-1),
        torch.linalg.norm(torch.reshape(yTrue, [yTrue.size(0), -1]), dim=-1)), dim=0)
    return epsilon

def calc_pixel_accuracy(predicted: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Computes the pixel-wise accuracy between predicted and ground truth binary images using PyTorch.

    Parameters:
    - predicted (torch.Tensor): Binary predicted image (0s and 1s)
    - ground_truth (torch.Tensor): Binary ground truth image (0s and 1s)

    Returns:
    - float: Pixel-wise accuracy between 0 and 1
    """
    highValue = torch.max(ground_truth)
    lowValue = torch.min(ground_truth)
    # Check that both tensors have the same shape
    if predicted.shape != ground_truth.shape:
        raise ValueError("Predicted and ground truth images must have the same shape.")

    predicted = torch.where(predicted > (highValue+lowValue)/2, 1, 0)
    ground_truth = torch.where(ground_truth > (highValue+lowValue)/2, 1, 0)
    # Ensure both are binary
    if not torch.all((predicted == 0) | (predicted == 1)):
        raise ValueError("Predicted image must be binary (0 or 1).")
    if not torch.all((ground_truth == 0) | (ground_truth == 1)):
        raise ValueError("Ground truth image must be binary (0 or 1).")

    # Count correct predictions
    correct = (predicted == ground_truth).sum().item()

    # Total number of pixels
    total = predicted.numel()

    # Accuracy
    accuracy = correct / total

    return accuracy

def makeCGProjection(pde, sampX, sampSolFenics, sampYCoeff, load=False):
        if load == False:
            pde.sampX
            xTest = sampX
            yTest = sampSolFenics
            pde.shapeFunc.createShapeFuncsConstraint()
            rbfRefSol = pde.shapeFunc.cTrialSolutionParallel(sampYCoeff)
            pde.shapeFunc.createShapeFuncsFree()
            sampYCoeff = pde.shapeFunc.findRbfCoeffs(closedForm=True, y=rbfRefSol, N=30000)
            YProj = torch.einsum('ij,...j->...i', torch.linalg.inv(pde.shapeFunc.aInvB.t() @ pde.shapeFunc.aInvB) @ pde.shapeFunc.aInvB.t(), sampYCoeff)
            yFProj = torch.einsum('ij,...j->...i', torch.linalg.inv(pde.shapeFunc.aInvBCO.t() @ pde.shapeFunc.aInvBCO) @ pde.shapeFunc.aInvBCO.t(), sampYCoeff)
            yProj = torch.einsum('ij,...j->...i',pde.shapeFunc.aInvB, YProj)
            yProjTotal = torch.einsum('ij,...j->...i',pde.shapeFunc.aInvB, YProj) + torch.einsum('...ij,...j->...i', pde.shapeFunc.aInvBCO, yFProj)
            yProj = pde.shapeFunc.cTrialSolutionParallel(yProj)
            yProjTotal = pde.shapeFunc.cTrialSolutionParallel(yProjTotal)
            tensorsToSave = {'xTest': xTest, 'yTest': yTest, 'yProj': yProj, 'yProjTotal': yProjTotal}
            torch.save(tensorsToSave, "./projectionTensors.pt")
            ### Process for fixing the over-constraint situation of the YProj that the CG model gives which isn't flexible enough for masked trial shape functions
            t0 = torch.nn.functional.interpolate(rbfRefSol[0].unsqueeze(0).unsqueeze(0), size=(17, 17), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
            t1 = torch.nn.functional.interpolate(t0.unsqueeze(0).unsqueeze(0), size=(65, 65), mode='bilinear', align_corners=True).squeeze(0).squeeze(0)
            t2 = pde.shapeFunc.findRbfCoeffs(closedForm=True, y=t1.unsqueeze(0), N=30000).squeeze(0)
            t3 = torch.einsum('ij,...j->...i', torch.linalg.inv(pde.shapeFunc.aInvB.t() @ pde.shapeFunc.aInvB) @ pde.shapeFunc.aInvB.t(), t2)
            return xTest, yTest, yProj, yProjTotal
        else:
            tensorsToSave = torch.load("./projectionTensors.pt")
            return tensorsToSave['xTest'], tensorsToSave['yTest'], tensorsToSave['yProj'], tensorsToSave['yProjTotal']

def createFolderIfNotExists(folder_path):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at: {folder_path}")
        return
    else:
        return
        

def setupDevice(cudaIndex, device, dataType):
    np.set_printoptions(formatter={'float': '{: 0.14f}'.format})
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cudaIndex)

    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

    if device == 'cpu':
        use_cuda=False
        torch.set_default_device('cpu')
    else:
        use_cuda=True
        torch.set_default_device('cuda:0')

    if dataType == 'float':
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.float64)
    torch.cuda.set_device("cuda:0")
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        print("CUDA is available")
        print("PyTorch version:", torch.__version__)
    else:
        print("CUDA is not available")

    print("Using device", device)
    return device

def get_tensor_size(tensor):
    """Returns the memory size of a tensor in bytes."""
    return tensor.element_size() * tensor.nelement()

def autocorrelation(x, max_lag=None):
    """Compute autocorrelation function for a 1D torch tensor."""
    n = x.shape[0]
    x = x - x.mean()
    result = torch.zeros(max_lag)
    var = x.var(unbiased=False)

    for lag in range(1, max_lag + 1):
        cov = (x[:-lag] * x[lag:]).mean()
        result[lag - 1] = cov / var
    return result

def effective_sample_size(samples, max_lag=100):
    """
    samples: Tensor of shape (num_samples, num_dims)
    returns: ESS per dimension, shape (num_dims,)
    """
    num_samples, num_dims = samples.shape
    ess = torch.zeros(num_dims)

    for d in range(num_dims):
        x = samples[:, d]
        rho = autocorrelation(x, max_lag=max_lag)

        # Truncate sum where autocorrelation becomes negative
        positive_rhos = rho[rho > 0]
        tau = 1 + 2 * positive_rhos.sum()
        ess[d] = num_samples / tau

    return ess

def create_latent_grid(dimz=50, blueprint_z=None, N=10, z1_idx=5, z2_idx=6, 
                       interp_range=(-5, 5), device=None):
    """
    Creates an NxN grid of latent vectors for interpolation along two dimensions.

    Args:
        z_dim (int): Dimensionality of the latent space.
        N (int): Number of interpolation steps per axis (grid will be NxN).
        z1_idx (int): Index of the first dimension to interpolate.
        z2_idx (int): Index of the second dimension to interpolate.
        interp_range (tuple): (min, max) range to interpolate over.
        device (torch.device or str): Device to create tensors on.

    Returns:
        torch.Tensor: A tensor of shape (N*N, z_dim) containing the latent vectors.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sample one base vector from N(0,1)
    if blueprint_z is None:
        base_z = torch.randn(dimz, device=device)
    else:
        base_z = blueprint_z.squeeze(0)

    # Generate linspace values for the interpolation range
    lin_vals = torch.linspace(interp_range[0], interp_range[1], N, device=device)

    # Allocate space for grid
    z_grid = torch.zeros(N * N, dimz, device=device)

    for i in range(N):
        for j in range(N):
            z = base_z.clone()
            z[z1_idx] = lin_vals[i]
            z[z2_idx] = lin_vals[j]
            z_grid[i * N + j] = z

    return z_grid

def interpolate_latents(Z1, Z2, steps=10):
    """
    Linearly interpolates between two latent vectors Z1 and Z2.

    Args:
        Z1 (torch.Tensor): Starting latent vector of shape (z_dim,).
        Z2 (torch.Tensor): Ending latent vector of shape (z_dim,).
        steps (int): Number of interpolation steps (including Z1 and Z2).

    Returns:
        torch.Tensor: Tensor of shape (steps, z_dim) containing interpolated vectors.
    """
    assert Z1.shape == Z2.shape, "Z1 and Z2 must have the same shape"
    assert Z1.dim() == 1, "Z1 and Z2 must be 1D tensors (single vectors)"

    t = torch.linspace(0, 1, steps, device=Z1.device).unsqueeze(1)  # shape: (steps, 1)
    interpolated = (1 - t) * Z1 + t * Z2  # shape: (steps, z_dim)
    return interpolated

def calcUncertaintyMetric(meanSol, meanRef, stdSol, sIndex):
        """
        :param meanSol: The mean prediction of the trained surrogate
        :param meanRef: The mean solution, calculated by a numerical solver (Reference solution)
        :param stdSol: The std at each point of the domain, calculated from actual samples from the posterior
        :param sIndex: The factor with which we multiply the standard deviation (e.g. +- 2 sigma)
        :return: The Envelope Metric
        """
        x = 1. - torch.abs(meanSol - meanRef) / (sIndex * (stdSol)+10**(-6))
        #mask = torch.logical_and(x>=-1., x<=1.)
        #torch.where(mask, torch.abs(x), 1 - torch.abs(x))
        return x

def patched_getitem(self, index):
    data = tuple(tensor[index] for tensor in self.tensors)
    return data, index

def image_to_binary_tensor(image_path: Union[str, Path], size=(65, 65), threshold=128) -> torch.Tensor:
    """
    Converts a black and white image to a binary (0 and 1) torch tensor.

    Parameters:
    - image_path (str or Path): Path to the image file.
    - size (tuple): Desired size for the output tensor (width, height).
    - threshold (int): Threshold for binarization (default 128).

    Returns:
    - torch.Tensor: A tensor of shape (height, width) with values 0 or 1.
    """
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize(size)  # Resize to desired dimensions
    image_np = np.array(image)
    binary_np = (image_np > threshold).astype(np.uint8)  # Binarize
    return torch.tensor(binary_np, dtype=torch.uint8)

def list_gpu_tensors(num=None):
    """Lists all tensors in GPU memory sorted by their size."""
    tensors = []
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            tensors.append((obj, get_tensor_size(obj)))
    
    # Sort by tensor size (from largest to smallest)
    tensors = sorted(tensors, key=lambda x: x[1], reverse=True)
    i=0
    sum = 0
    if i is not None:
        num=0
    for tensor, size in tensors:
        i = i + 1
        sum = sum + size / 1024 ** 2
        print(f"Tensor: {tensor.size()}, Memory: {size / 1024 ** 2:.2f} MB")
        if i == num:
            break
    print(f"Total size of N Tensors: {sum:.2f} MB")

# Call the function to list all GPU tensors by memory usage
list_gpu_tensors()


from pathlib import Path

def clean_torch_file(input_path, output_path=None):
    """
    Loads a torch file, detaches + clones all tensors (removes graph and shared storage),
    moves them to CPU, and saves a cleaned version.
    Works for:
        - Single torch.Tensor
        - dict containing tensors (nested)
        - lists / tuples containing tensors
    """
    
    def clean(obj):
        if torch.is_tensor(obj):
            # detach -> remove graph
            # clone  -> remove oversized backing storage
            # cpu    -> remove CUDA dependency
            return obj.detach().clone().cpu()
        elif isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(clean(v) for v in obj)
        else:
            return obj

    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path

    # Load safely to CPU
    data = torch.load(input_path, map_location="cpu")

    # Clean
    cleaned_data = clean(data)

    # Save
    torch.save(cleaned_data, output_path)


def find_latest_checkpoint(pde_type, root='checkpoints'):
    """Return the most recently modified .ckpt for pde_type.

    Searches two locations (most-recently-modified wins):
      1. Flat pretrained files directly in root/ containing the pde_type name
         (e.g. checkpoints/darcy10k_pretrained.ckpt)
      2. Nested checkpoints under root/genPANIS_{pde_type}_*/ produced by training
    """
    import glob
    candidates = []
    candidates += glob.glob(os.path.join(root, f'*{pde_type}*.ckpt'))
    candidates += glob.glob(os.path.join(root, f'genPANIS_{pde_type}_*', '**', '*.ckpt'), recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for '{pde_type}' under '{root}'.\n"
            f"Expected a flat file like '{root}/{pde_type}*_pretrained.ckpt' or a trained "
            f"checkpoint under '{root}/genPANIS_{pde_type}_*'.\n"
            f"Train a model first: python experiments/demo/train_{pde_type}.py"
        )
    return max(candidates, key=os.path.getmtime)

    print(f"Saved cleaned file to: {output_path}")
