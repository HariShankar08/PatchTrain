import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

class PatchTrainModel(nn.Module):
    """
    A wrapper class that modifies a regular transformer model to implement patch training functionality.
    This class enables training models on patches of tokens rather than individual tokens, which can
    help with efficiency and potentially improve model performance.

    The patch training approach works by:
    1. Grouping input tokens into patches of size `patch_size`
    2. Computing a representation for each patch using the specified calculation method
    3. Processing these patch representations through the transformer layers
    4. Computing loss across all positions within each patch

    Example usage:
    ```python
    from transformers import AutoModelForCausalLM
    from patch_model import PatchTrainModel

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Create patch model wrapper
    patch_model = PatchTrainModel(
        base_model,
        patch_size=4,
        patch_calculation_method="paraMean"
    )
    
    # Use with Hugging Face Trainer
    from transformers import Trainer, TrainingArguments
    
    trainer = Trainer(
        model=patch_model,
        args=training_args,
        train_dataset=train_dataset
    )
    trainer.train()
    ```

    Args:
        base_model (nn.Module): The base transformer model to wrap. Should be compatible with
            Hugging Face's model interface.
        patch_size (int, optional): Size of each patch in tokens. Defaults to 4.
        patch_calculation_method (str, optional): Method to use for computing patch representations.
            Options are "mean" or "paraMean". Defaults to "mean".
            - "mean": Simple average of token embeddings in the patch
            - "paraMean": Learnable weighted average of token embeddings
    """
    def __init__(
        self, 
        base_model: nn.Module,
        patch_size: int = 4,
        patch_calculation_method: str = "mean"
    ):
        super().__init__()
        self.base_model = base_model
        self.patch_size = patch_size
        self.patch_calculation_method = patch_calculation_method
        
        # Initialize learnable parameters for paraMean calculation
        if patch_calculation_method == "paraMean":
            self.ParaMeans = nn.Parameter(torch.randn(patch_size))
    
    def compute_paraMean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted mean of patches using learnable parameters.
        
        This method applies a softmax-normalized weighted sum to the token embeddings
        within each patch. The weights are learned parameters that determine the
        importance of each position within the patch.
        
        Args:
            x: Input tensor of shape [batch_size, num_patches, patch_size, hidden_size]
               containing token embeddings for each patch
               
        Returns:
            Tensor of shape [batch_size, num_patches, hidden_size] containing
            the weighted patch representations
        """
        weights = torch.softmax(self.ParaMeans, dim=0)
        return torch.einsum('bnph,p->bnh', x, weights)
    
    def calculate_patch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate patch representation based on the configured method.
        
        This method applies the selected patch calculation method to convert
        token embeddings into patch representations.
        
        Args:
            x: Input tensor of shape [batch_size, num_patches, patch_size, hidden_size]
               containing token embeddings for each patch
               
        Returns:
            Tensor of shape [batch_size, num_patches, hidden_size] containing
            the computed patch representations
            
        Raises:
            ValueError: If an unknown patch calculation method is specified
        """
        if self.patch_calculation_method == "mean":
            return x.mean(2)
        elif self.patch_calculation_method == "paraMean":
            return self.compute_paraMean(x)
        else:
            raise ValueError(f"Unknown patch calculation method: {self.patch_calculation_method}")
    
    def prepare_patch_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for patch processing.
        
        This method handles the conversion of input tokens into patch representations
        and adjusts the attention mask and position IDs accordingly.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, sequence_length]
            attention_mask: Optional attention mask of shape [batch_size, sequence_length]
            position_ids: Optional position IDs of shape [batch_size, sequence_length]
            inputs_embeds: Optional pre-computed input embeddings
            
        Returns:
            Tuple containing:
            - Processed input embeddings of shape [batch_size, num_patches, hidden_size]
            - Adjusted position IDs of shape [batch_size, num_patches]
            - Adjusted attention mask of shape [batch_size, num_patches]
        """
        batch_size, seq_length = input_ids.shape
        
        # Calculate number of complete patches
        num_patches = seq_length // self.patch_size
        
        # Get input embeddings if not provided
        if inputs_embeds is None:
            if hasattr(self.base_model, "get_input_embeddings"):
                inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
            else:
                inputs_embeds = self.base_model.embed_tokens(input_ids)
        
        # Reshape for patch processing
        inputs_embeds = inputs_embeds.view(batch_size, num_patches, self.patch_size, -1)
        inputs_embeds = self.calculate_patch(inputs_embeds)
        
        # Adjust position IDs for patches
        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(num_patches, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = position_ids[:, :num_patches]
        
        # Adjust attention mask for patches
        if attention_mask is not None:
            attention_mask = attention_mask[:, :num_patches]
        
        return inputs_embeds, position_ids, attention_mask
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass with patch training support.
        
        This method implements the main forward pass of the model, handling:
        1. Conversion of inputs into patch representations
        2. Processing through the base model
        3. Patch-specific loss calculation during training
        
        Args:
            input_ids: Input token IDs of shape [batch_size, sequence_length]
            attention_mask: Optional attention mask of shape [batch_size, sequence_length]
            position_ids: Optional position IDs of shape [batch_size, sequence_length]
            past_key_values: Optional cached key/value pairs for faster decoding
            inputs_embeds: Optional pre-computed input embeddings
            labels: Optional target labels for loss calculation
            use_cache: Whether to use cached key/value pairs
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary instead of a tuple
            **kwargs: Additional arguments passed to the base model
            
        Returns:
            Either a tuple of outputs or a CausalLMOutputWithPast object containing:
            - loss: The computed loss (if labels are provided)
            - logits: The model's output logits
            - past_key_values: Cached key/value pairs (if use_cache is True)
            - hidden_states: All hidden states (if output_hidden_states is True)
            - attentions: All attention weights (if output_attentions is True)
        """
        # Prepare inputs for patch processing
        inputs_embeds, position_ids, attention_mask = self.prepare_patch_inputs(
            input_ids, attention_mask, position_ids, inputs_embeds
        )
        
        # Get base model outputs
        outputs = self.base_model(
            input_ids=None,  # We're using inputs_embeds instead
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        # Get logits from the base model output
        if hasattr(self.base_model, "lm_head"):
            logits = self.base_model.lm_head(outputs[0])
        else:
            logits = outputs[0]
        
        # Calculate patch-specific loss if labels are provided
        loss = None
        if labels is not None:
            # Reshape logits and labels for patch-wise loss calculation
            shift_logits = logits[..., :-1, :].reshape(-1, self.base_model.config.vocab_size)
            shift_labels = labels[..., self.patch_size:].reshape(-1, self.patch_size)
            
            # Calculate loss for each position in the patch
            loss = 0
            log_probs = F.log_softmax(shift_logits, dim=1)
            for i in range(self.patch_size):
                loss = loss + F.nll_loss(log_probs, shift_labels[:, i])
            loss = loss / self.patch_size
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )
    
    def generate(self, *args, **kwargs):
        """
        Generate text using the model.
        
        This method delegates to the base model's generate method, maintaining
        all the functionality of the original model for text generation.
        
        Args:
            *args: Positional arguments passed to the base model's generate method
            **kwargs: Keyword arguments passed to the base model's generate method
            
        Returns:
            Generated token IDs
        """
        return self.base_model.generate(*args, **kwargs)
    
    def __getattr__(self, name):
        """
        Delegate attribute access to the base model.
        
        This ensures that all attributes and methods of the base model
        remain accessible through the patch model wrapper.
        
        Args:
            name: Name of the attribute to access
            
        Returns:
            The requested attribute from the base model
        """
        return getattr(self.base_model, name) 