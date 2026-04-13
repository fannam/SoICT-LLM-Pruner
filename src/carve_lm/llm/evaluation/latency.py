from __future__ import annotations

import time

import torch
from transformers import AutoTokenizer

from ..auto_model import PrunedAutoModelForCausalLM


class LLMMeasurer:
    """
    Measure latency and throughput for regular or component-pruned causal LMs.
    """

    def __init__(self, model_name_or_path: str, device: str = None):
        """
        Initializes the LLMMetrics class.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model.
            device (str, optional): The device to run the model on ('cuda', 'cpu', etc.).
                                    Defaults to 'cuda' if available, otherwise 'cpu'.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        self.model = PrunedAutoModelForCausalLM.from_pretrained(model_name_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        print(f"Model '{model_name_or_path}' and tokenizer loaded successfully.")

    @torch.no_grad()
    def measure_latency(self, prompt: str, max_new_tokens: int = 50, num_runs: int = 5) -> tuple[float, int]:
        """
        Measures the average latency for a single generation.

        Args:
            prompt (str): The input text prompt for the model.
            max_new_tokens (int): The maximum number of new tokens to generate.
            num_runs (int): The number of times to run the generation for averaging.
                            The first run is a warm-up and is not included in the average.

        Returns:
            tuple[float, int]: A tuple containing:
                - average_latency_ms (float): The average latency in milliseconds.
                - generated_tokens (int): The number of tokens generated in the last run.
        """
        if num_runs <= 1:
            print("Warning: For meaningful latency, num_runs should be > 1 for warm-up.")

        total_time_ms = 0
        generated_tokens = 0

        if num_runs > 1:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            _ = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if self.device == "cuda":
                torch.cuda.synchronize()

        for _ in range(max(1, num_runs - 1)):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids_length = inputs.input_ids.shape[1]

            start_time = time.perf_counter()
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if self.device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            total_time_ms += (end_time - start_time) * 1000
            generated_tokens = outputs.shape[1] - input_ids_length

        avg_latency_ms = total_time_ms / max(1, num_runs - 1)
        return avg_latency_ms, generated_tokens

    @torch.no_grad()
    def measure_throughput(
        self,
        prompts: list[str],
        max_new_tokens: int = 50,
        batch_size: int = 1,
    ) -> tuple[float, float, int]:
        """
        Measures the throughput of the LLM (tokens per second).

        Args:
            prompts (list[str]): A list of input text prompts.
            max_new_tokens (int): The maximum number of new tokens to generate for each prompt.
            batch_size (int): The number of prompts to process in a single batch.
                              Note: Padded batching is more efficient.

        Returns:
            tuple[float, float, int]: A tuple containing:
                - throughput_tokens_per_sec (float): Tokens generated per second.
                - total_time_sec (float): Total time taken for all generations in seconds.
                - total_generated_tokens (int): Total number of new tokens generated.
        """
        total_generated_tokens = 0
        total_time_sec = 0
        num_prompts = len(prompts)

        self.tokenizer.padding_side = "left"

        for i in range(0, num_prompts, batch_size):
            batch_prompts = prompts[i:i + batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(
                self.device
            )
            input_ids_lengths = [len(ids) for ids in inputs.input_ids]

            start_time = time.perf_counter()
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if self.device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            total_time_sec += end_time - start_time
            for j in range(outputs.shape[0]):
                total_generated_tokens += outputs.shape[1] - input_ids_lengths[j]

        if total_time_sec == 0:
            throughput_tokens_per_sec = float("inf") if total_generated_tokens > 0 else 0.0
        else:
            throughput_tokens_per_sec = total_generated_tokens / total_time_sec

        return throughput_tokens_per_sec, total_time_sec, total_generated_tokens
