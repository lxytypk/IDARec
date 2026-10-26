import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment
import torch.distributed as dist

class ContinualLoRALinear(nn.Module):
    """
    Single-adapter continual LoRA used by PESO-style variants.

    At each block we keep one active LoRA adapter `(A, B)` and store snapshots from
    previous blocks in `A_matrices` / `B_matrices`. In the default inherited setting,
    training resumes from the previously saved adapter state; `noinherit` explicitly
    disables that inheritance by reinitializing the current adapter.
    """
    def __init__(self,
                 orig_linear: nn.Linear,
                 r: int,
                 alpha: float,
                 dropout: float = 0.0,
                 num_blocks: int = 5,
                 option: str = "lora"):
        super().__init__()
        self.orig = orig_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.num_blocks = num_blocks
        self.current_block = 0
        self.merged = False
        self.option = option

        # Get device and dtype from original linear layer
        in_f, out_f = orig_linear.in_features, orig_linear.out_features
        device, dtype = orig_linear.weight.device, orig_linear.weight.dtype

        # One active LoRA adapter for the current block
        self.A = nn.Parameter(torch.zeros(r, in_f, device=device, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(out_f, r, device=device, dtype=dtype))
        
        # Initialize A with kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        # Snapshots from previous blocks used by PESO / ablation penalties
        self.A_matrices = nn.ParameterList([])
        self.B_matrices = nn.ParameterList([])

        # Weights for continual-regularization variants:
        # L2 proximity, orthogonal separation, and KL proximity (PESO).
        self.l2_weight = 1.0
        self.ortho_weight = 1.0
        self.kl_weight = 1.0



    def reinit_current_parameters(self):
        """Reinitialize the current LoRA adapter instead of inheriting the previous block."""
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def set_current_block(self, block_id: int):
        """Switch to a new chronological block and configure the active adapter."""
        assert 0 <= block_id < self.num_blocks, f"Block ID {block_id} out of range [0, {self.num_blocks})"
        
        self.current_block = block_id

        if 'noinherit' in self.option:
            self.reinit_current_parameters()
            
        self._update_parameter_gradients()

    def _initialize_with_previous_knowledge(self):
        """Copy the most recent saved adapter into the current adapter state."""
        
        # Inherited initialization uses only the latest previous block.
        idx = self.current_block - 1
        
        # Safety checks
        if idx < 0 or idx >= len(self.A_matrices):
            print(f"Warning: Block {self.current_block} trying to access saved matrices at index {idx}, but only {len(self.A_matrices)} matrices are saved. Skipping initialization.")
            
        A_prev = self.A_matrices[idx]
        B_prev = self.B_matrices[idx]
        self.A.data.copy_(A_prev)
        self.B.data.copy_(B_prev)

    def _update_parameter_gradients(self):
        """Make only the current adapter trainable and freeze stored history."""
        self.A.requires_grad = True
        self.B.requires_grad = True

        # Stored adapters are historical references and stay frozen.
        for i in range(len(self.A_matrices)):
            self.A_matrices[i].requires_grad = False
            self.B_matrices[i].requires_grad = False


    def compute_l2_loss(self):
        """L2 proximity variant against previous adapter states."""
        if len(self.A_matrices) == 0:
            return torch.tensor(0.0, device=self.A.device, dtype=self.A.dtype)
        
        reg_loss = 0.0
        for i in range(len(self.A_matrices)):
            if 'latest' in self.option and i != len(self.A_matrices) - 1:
                continue
            
            A_prev = self.A_matrices[i].detach()  # Detach to prevent gradient flow
            B_prev = self.B_matrices[i].detach()  # Detach to prevent gradient flow
            
            # L2 proximity between the current adapter and stored previous adapters.
            A_reg = torch.norm(self.A - A_prev, p=2)
            B_reg = torch.norm(self.B - B_prev, p=2)
            
            reg_loss += A_reg + B_reg
        
        if 'latest' in self.option:
            reg_loss = reg_loss 
        else:
            reg_loss = reg_loss / len(self.A_matrices)
        
        return reg_loss

    def compute_orthogonal_loss(self):
        """Orthogonal-separation variant against previous adapter states."""
        if len(self.A_matrices) == 0:
            return torch.tensor(0.0, device=self.A.device, dtype=self.A.dtype)
        
        ortho_loss = 0.0
        for i in range(len(self.A_matrices)):
            if 'latest' in self.option and i != len(self.A_matrices) - 1:
                continue
            
            A_prev = self.A_matrices[i].detach()  # Detach to prevent gradient flow
            B_prev = self.B_matrices[i].detach()  # Detach to prevent gradient flow

            # Encourage the current adapter to move in a direction different from history.
            A_ortho = torch.matmul(self.A, A_prev.t())
            A_ortho_loss = torch.sum(torch.square(A_ortho))
            
            # Apply the same separation idea to B.
            B_ortho = torch.matmul(self.B.t(), B_prev)
            B_ortho_loss = torch.sum(torch.square(B_ortho))
            
            ortho_loss += A_ortho_loss + B_ortho_loss
        
        if 'latest' in self.option:
            ortho_loss = ortho_loss 
        else:
            ortho_loss = ortho_loss / len(self.A_matrices)
        
        return ortho_loss

    def compute_kl_divergence_loss(self):
        """
        KL-based proximity penalty used by PESO-style variants.

        We flatten the current and previous LoRA parameters, convert them to
        distributions with softmax, and penalize divergence from previous blocks.
        """
        if len(self.A_matrices) == 0:
            return torch.tensor(0.0, device=self.A.device, dtype=self.A.dtype)
        
        kl_loss = 0.0
        for i in range(len(self.A_matrices)):
            if 'latest' in self.option and i != len(self.A_matrices) - 1:
                continue
            
            A_prev = self.A_matrices[i].detach()  # Detach to prevent gradient flow
            B_prev = self.B_matrices[i].detach()  # Detach to prevent gradient flow
    
            # Turn LoRA parameters into distributions for the KL-based PESO penalty.
            A_current_logits = self.A.flatten()
            A_prev_logits = A_prev.flatten()
            B_current_logits = self.B.flatten()
            B_prev_logits = B_prev.flatten()
            
            # Softmax produces normalized parameter distributions.
            A_current_probs = F.softmax(A_current_logits, dim=0)
            A_prev_probs = F.softmax(A_prev_logits, dim=0)
            B_current_probs = F.softmax(B_current_logits, dim=0)
            B_prev_probs = F.softmax(B_prev_logits, dim=0)
            
            # Add small epsilon to avoid log(0)
            eps = 1e-8
            A_current_probs = A_current_probs + eps
            A_prev_probs = A_prev_probs + eps
            B_current_probs = B_current_probs + eps
            B_prev_probs = B_prev_probs + eps
            
            # Renormalize
            A_current_probs = A_current_probs / A_current_probs.sum()
            A_prev_probs = A_prev_probs / A_prev_probs.sum()
            B_current_probs = B_current_probs / B_current_probs.sum()
            B_prev_probs = B_prev_probs / B_prev_probs.sum()
            
            # KL(current || previous): proximity to earlier adapters in distribution space.
            A_kl = torch.sum(A_current_probs * torch.log(A_current_probs / A_prev_probs))
            B_kl = torch.sum(B_current_probs * torch.log(B_current_probs / B_prev_probs))
            
            # Ensure non-negative values
            A_kl = torch.clamp(A_kl, min=0.0)
            B_kl = torch.clamp(B_kl, min=0.0)
            
            kl_loss += A_kl + B_kl
        
        if 'latest' in self.option:
            kl_loss = kl_loss 
        else:
            kl_loss = kl_loss / len(self.A_matrices)

        return kl_loss


    def compute_continual_loss(self):
        """Dispatch to the continual-regularization variant requested by `option`."""

        # The trainer adds this block-wise continual penalty to the main ranking loss.
        total_loss = 0.0
        
        if 'l2' in self.option:
            reg_loss = self.compute_l2_loss()
            total_loss += self.l2_weight * reg_loss
        if 'orthogonal' in self.option:
            ortho_loss = self.compute_orthogonal_loss()
            total_loss += self.ortho_weight * ortho_loss
        if 'kldiv' in self.option:
            kl_loss = self.compute_kl_divergence_loss()
            total_loss += self.kl_weight * kl_loss
        
        return total_loss, {
            'reg_loss': reg_loss.item() if 'l2' in self.option else 0.0,
            'ortho_loss': ortho_loss.item() if 'orthogonal' in self.option else 0.0,
            'kl_loss': kl_loss.item() if 'kldiv' in self.option else 0.0,
            'total_cl_loss': total_loss.item()
        }


    def save_current_matrices(self):
        """Store the current adapter as a historical snapshot for later blocks."""
        # Safety checks before saving
        if self.A is None or self.B is None:
            print(f"Warning: A or B matrices are None in block {self.current_block}. Cannot save.")
            return
            
        if not hasattr(self.A, 'data') or not hasattr(self.B, 'data'):
            print(f"Warning: A or B matrices don't have .data attribute in block {self.current_block}. Cannot save.")
            return
            
        # Store a detached copy of the current block adapter.
        A_clone = nn.Parameter(self.A.data.clone())
        B_clone = nn.Parameter(self.B.data.clone())
        
        self.A_matrices.append(A_clone)
        self.B_matrices.append(B_clone)


    def forward(self, x):   
        if self.merged:
            return self.orig(x)
        
        # Frozen backbone projection.
        base = self.orig(x)
        dropped = self.dropout(x)

        # Optional cumulative baseline: add previous LoRA adapters at inference/training time.
        # This is a SumLoRA-style baseline, not the default PESO path.
        if 'cumulative' in self.option and self.current_block > 0:
            for i in range(self.current_block):
                if i != self.current_block - 1 and 'latest' in self.option:
                    continue

                # Direction-only accumulation baseline.
                A_norm = self.A_matrices[i] / (torch.norm(self.A_matrices[i]) + 1e-8)
                B_norm = self.B_matrices[i] / (torch.norm(self.B_matrices[i]) + 1e-8)
                base += self.scaling * (dropped @ A_norm.t()) @ B_norm.t()


        # Current block adapter.
        base += self.scaling * (dropped @ self.A.t()) @ self.B.t()

        return base
        

    # In the single-adapter PESO path, final-checkpoint evaluation restores the
    # requested block adapter from the saved history.
    def set_saved_matrices(self, block_id: int, use_final_model: bool = False):
        """Set saved matrices for inference at a specific block."""
        self.current_block = block_id

        if use_final_model: # and block_id < len(self.A_matrices):
            self.A.data.copy_(self.A_matrices[block_id])
            self.B.data.copy_(self.B_matrices[block_id])

    def merge(self):
        """Merge current LoRA component into the original weight"""
        if not self.merged:
            deltaW = (self.B @ self.A) * self.scaling
            self.orig.weight.data += deltaW
            self.merged = True

    def unmerge(self):
        """Unmerge LoRA component from the original weight"""
        if self.merged:
            deltaW = (self.B @ self.A) * self.scaling
            self.orig.weight.data -= deltaW
            self.merged = False




class LearnableMagnitudeContinualLoRALinear(nn.Module):
    """
    Alternative continual adapter variant used by the `sdlora` family.

    This path keeps previous directions and learns magnitudes over time. It is
    retained for method comparisons and ablations rather than the default PESO path.
    """
    def __init__(self,
                 orig_linear: nn.Linear,
                 r: int,
                 alpha: float,
                 dropout: float = 0.0,
                 num_blocks: int = 5,
                 option: str = "lora"):
        super().__init__()
        self.orig = orig_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.num_blocks = num_blocks
        self.current_block = 0
        self.merged = False
        self.option = option

        # Get device and dtype from original linear layer
        in_f, out_f = orig_linear.in_features, orig_linear.out_features
        device, dtype = orig_linear.weight.device, orig_linear.weight.dtype

        # Active adapter for the current block plus an explicit magnitude parameter.
        self.A = nn.Parameter(torch.zeros(r, in_f, device=device, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(out_f, r, device=device, dtype=dtype))
        self.current_magnitude = nn.Parameter(torch.ones(1, device=device, dtype=dtype))

        self.magnitude_init_value = 1.0
        self.current_magnitude.data.fill_(self.magnitude_init_value)
        
        # Initialize A with kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        
        # Historical adapter snapshots and their magnitudes for later blocks/inference.
        self.A_matrices = nn.ParameterList([])
        self.B_matrices = nn.ParameterList([])
        self.magnitudes = nn.ParameterList([])
        
    
    def reinit_current_parameters(self):
        """Reset the current adapter and magnitude instead of inheriting the previous state."""
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        self.current_magnitude.data.fill_(1.0)


    def set_current_block(self, block_id: int):
        """Switch to a new block and prepare magnitude parameters for historical adapters."""
        assert 0 <= block_id < self.num_blocks, f"Block ID {block_id} out of range [0, {self.num_blocks})"
        self.current_block = block_id

        if 'noinherit' in self.option:
            self.reinit_current_parameters()
        
        # Reset the current block magnitude before training.
        self.current_magnitude.data.fill_(self.magnitude_init_value)
        
        # Create one learnable magnitude for each stored previous adapter.
        self.learnable_prev_magnitudes = nn.ParameterList([])
        if block_id > 0:
            # Clear existing magnitudes
            # Create new learnable magnitude for each previous direction pair
            for _ in range(self.current_block):
                new_mag = nn.Parameter(torch.ones(1, device=self.A.device, dtype=self.A.dtype))
                new_mag.data.fill_(self.magnitude_init_value)
                self.learnable_prev_magnitudes.append(new_mag)
        
        self._update_parameter_gradients()


    def _update_parameter_gradients(self):
        """Train the current adapter and magnitude parameters while freezing history."""
        self.A.requires_grad = True
        self.B.requires_grad = True
        self.current_magnitude.requires_grad = True
        
        # Historical magnitudes remain trainable in this variant.
        if self.current_block > 0:
            for mag in self.learnable_prev_magnitudes:
                mag.requires_grad = True

        # Historical adapter snapshots stay frozen.
        for i in range(self.current_block):
            # Set requires_grad on the original parameters, not on the moved tensors
            self.A_matrices[i].requires_grad = False
            self.B_matrices[i].requires_grad = False

    def save_current_matrices(self):
        """Store the current adapter snapshot and its magnitudes for later blocks."""
        self.A_matrices.append(nn.Parameter(self.A.data.clone()))
        self.B_matrices.append(nn.Parameter(self.B.data.clone()))

        combined_magnitudes = torch.cat(
            [mag.data.clone() for mag in self.learnable_prev_magnitudes] +
            [self.current_magnitude.data.clone()],
            dim=0
        )
        self.magnitudes.append(nn.Parameter(combined_magnitudes.clone()))


    def _apply_previous_directions(self, base, dropped):
        """Replay previous adapters using their current magnitudes."""

        previous = torch.zeros_like(base)
        for i in range(self.current_block):

            if 'latest' in self.option:
                if i < self.current_block - 1:
                    continue

            A_prev = self.A_matrices[i]
            B_prev = self.B_matrices[i]
            mag = self.learnable_prev_magnitudes[i]
            previous += mag * (dropped @ A_prev.t()/torch.norm(A_prev)) @ (B_prev.t()/torch.norm(B_prev))
            previous = self.scaling * previous

        return base + previous

    def forward(self, x):
        if self.merged:
            return self.orig(x)
        
        # Base projection
        base = self.orig(x)
        dropped = self.dropout(x)

        if self.current_block > 0:
            base = self._apply_previous_directions(base, dropped)
            
        # Current block update with optional direct block-0 pretraining behavior.
        if self.current_block == 0:
            delta = self.scaling * (dropped @ self.A.t()) @ self.B.t()

        else:
            mag = self.current_magnitude
            delta = mag * dropped @ self.A.t() @ self.B.t()
            delta = self.scaling * delta
            
        return base + delta
        

    # Restore historical magnitudes for the requested evaluation block.
    def set_saved_matrices(self, block_id: int, use_final_model: bool = False):
        self.current_block = block_id

        if use_final_model and block_id < len(self.A_matrices):
            self.A.data.copy_(self.A_matrices[block_id])
            self.B.data.copy_(self.B_matrices[block_id])
            if block_id < len(self.magnitudes):
                self.current_magnitude.data.copy_(self.magnitudes[block_id][-1])

        if block_id > 0:
            self.learnable_prev_magnitudes = nn.ParameterList([]) 

            if block_id < len(self.magnitudes):
                #print('magnitudes: ', self.magnitudes)
                for i in range(block_id):  
                    mag = nn.Parameter(self.magnitudes[block_id][i].clone())
                    self.learnable_prev_magnitudes.append(mag)
            else:
                # Fallback when the checkpoint contains fewer saved blocks than requested.
                for i in range(block_id):
                    mag = nn.Parameter(self.magnitudes[block_id-1][i].clone())
                    self.learnable_prev_magnitudes.append(mag)

            


def apply_continual_lora(model, target_modules, r, alpha, dropout, modules_to_save, 
                        num_blocks: int = 5, option: str = "lora"):
    """
    Inject the continual adapter module used by the chosen method variant.
    
    Args:
        model: The model to apply LoRA to
        target_modules: List of module names to apply LoRA to
        r: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        modules_to_save: List of module names to save (embed_tokens, lm_head)
        num_blocks: Number of data blocks for continual learning
        option: method/variant string used throughout the experimental codebase
    """
    # 1) Freeze the base model first; only adapters and selected saved modules train.
    for p in model.parameters():
        p.requires_grad = False

    # 2) Keep trainable copies of modules such as `embed_tokens` and `lm_head`.
    if modules_to_save is not None:
        for name, module in list(model.named_modules()):
            if any(name.endswith(ms) for ms in modules_to_save):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent_mod = model.get_submodule(parent_name)
                wrapped = SaveModuleWrapper(module, adapter_name="default")
                #print(f"Wrapped {name} with {wrapped}")
                setattr(parent_mod, child_name, wrapped)
                #print(f"Set {name} to {wrapped}")

    print(f'Continual LoRA option: {option}')

    if 'sdlora' in option:
        adapter_class = LearnableMagnitudeContinualLoRALinear
        print("using LearnableMagnitudeContinualLoRALinear")
    else:
        adapter_class = ContinualLoRALinear
        print("using continual normal lora")

    # 3) Replace target linear layers with the selected continual-adapter implementation.
    def _inject(mod, prefix=""):
        for nm, child in list(mod.named_children()):
            full = f"{prefix}.{nm}" if prefix else nm
            if modules_to_save is not None:
                if any(full.endswith(ms) for ms in modules_to_save):
                    continue

            if any(full.endswith(ms) for ms in ['embed_tokens', 'lm_head']):
                print(f"{full} is in modules_to_save")
                print(f"type of child: {type(child)}")

            if isinstance(child, nn.Linear) and any(full.endswith(tm) for tm in target_modules):
                if any(full.endswith(tm) for tm in ['embed_tokens', 'lm_head']):
                    print(f"{full} is in target_modules")
                    print(f"type of child: {type(child)}")
                setattr(mod, nm,
                        adapter_class(child, r=r, alpha=alpha, dropout=dropout, num_blocks=num_blocks, option=option))
            else:
                _inject(child, full)
    _inject(model)


    # 4) Unfreeze the trainable copies of modules-to-save.
    for n, p in model.named_parameters():
        if "modules_to_save.default" in n:
            p.requires_grad = True
                        


class SaveModuleWrapper(nn.Module):
    """
    Keep a frozen original module plus a trainable copy.

    This mirrors PEFT-style `modules_to_save` behavior for components such as
    token embeddings or the LM head.
    """
    def __init__(self, module: nn.Module, adapter_name="default"):
        super().__init__()
        self.original_module = module
        # Start the trainable copy from the same initialization as the frozen original.
        module_copy = copy.deepcopy(module)
        # for p in module_copy.parameters():
        #     p.requires_grad = True
        # Freeze the original module and optimize only the copied module.
        for p in self.original_module.parameters():
            p.requires_grad = False
        self.modules_to_save = nn.ModuleDict({adapter_name: module_copy})

    def forward(self, *args, **kwargs):
        # Route both training and inference through the trainable copy.
        return self.modules_to_save["default"](*args, **kwargs)


def print_trainable_parameters(model):
    """
    Prints the number of trainable vs total parameters in `model`.
    """
    trainable_params = 0
    total_params = 0
    print("--- Trainable Parameter Breakdown ---")
    for name, param in model.named_parameters():
        num = param.numel()
        total_params += num
        if param.requires_grad:
            trainable_params += num
            #print(f"  - {name:<70} | params: {num:<10,d} | requires_grad: {param.requires_grad}")
    
    print("-" * 100)
    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {total_params:,d} || "
        f"trainable%: {100 * trainable_params / total_params:.2f}%"
    ) 
