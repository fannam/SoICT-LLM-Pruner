from general_measure import LLMMeasurer


if __name__ == '__main__':
    # --- Configuration ---
    # Replace with a smaller model if you don't have a powerful GPU or want faster testing
    # e.g., "gpt2", "EleutherAI/pythia-160m-deduped"
    # For larger models like "meta-llama/Llama-2-7b-chat-hf", you'll need appropriate hardware and access.
    MODEL_NAME = "TheGardener/Qwen-0.8B-shortened-llama" # Example: Using a smaller, readily available model

    # --- Initialize Metrics ---
    try:
        llm_benchmarker = LLMMeasurer(model_name_or_path=MODEL_NAME)

        # --- Example Prompt ---
        PROMPT = "The weather today is"
        PROMPTS_LIST = [
            "Translate the following English text to French: 'Hello, how are you?'",
            "Write a short story about a robot who dreams of becoming a chef.",
            "What is the capital of Canada?",
            "Explain the theory of relativity in simple terms."
        ]

        # --- Measure Latency ---
        print(f"\n--- Measuring Latency for '{MODEL_NAME}' ---")
        avg_latency, generated_count = llm_benchmarker.measure_latency(
            prompt=PROMPT,
            max_new_tokens=20,
            num_runs=6 # 5 test runs + 1 warm-up
        )
        print(f"Average Latency (max_new_tokens=20): {avg_latency:.2f} ms")
        print(f"Tokens Generated in last latency run: {generated_count}")

        avg_latency_long, generated_count_long = llm_benchmarker.measure_latency(
            prompt=PROMPT,
            max_new_tokens=100,
            num_runs=6
        )
        print(f"Average Latency (max_new_tokens=100): {avg_latency_long:.2f} ms")
        print(f"Tokens Generated in last latency run: {generated_count_long}")

        # --- Measure Throughput ---
        print(f"\n--- Measuring Throughput for '{MODEL_NAME}' ---")
        throughput, total_time, total_tokens = llm_benchmarker.measure_throughput(
            prompts=PROMPTS_LIST,
            max_new_tokens=50,
            batch_size=2 # Try with batch_size=1 and a larger value
        )
        print(f"Throughput (max_new_tokens=50, batch_size=2): {throughput:.2f} tokens/sec")
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Total tokens generated: {total_tokens}")

        throughput_bs1, total_time_bs1, total_tokens_bs1 = llm_benchmarker.measure_throughput(
            prompts=PROMPTS_LIST,
            max_new_tokens=100,
            batch_size=4
        )
        print(f"Throughput (max_new_tokens=50, batch_size=1): {throughput_bs1:.2f} tokens/sec")
        print(f"Total time taken: {total_time_bs1:.2f} seconds")
        print(f"Total tokens generated: {total_tokens_bs1}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have the 'transformers' and 'torch' libraries installed.")
        print("If using a large model, make sure you have enough VRAM/RAM and have accepted any necessary licenses.")