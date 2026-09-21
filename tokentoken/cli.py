# tokentoken/cli.py
import functools
import typer
import os
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape
from .sdk import TokenToken
from .prompts import CompressionMode
from .utils import count_tokens

app = typer.Typer(help="Compile human prompts into LLM machine language.")
console = Console()


def handle_errors(func):
    """Print a clean one-line error instead of a traceback."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (typer.Exit, typer.Abort):
            raise
        except FileNotFoundError as e:
            console.print(f"[red]Error: file not found: {escape(e.filename or str(e))}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error: {escape(str(e) or type(e).__name__)}[/red]")
            raise typer.Exit(1)
    return wrapper

@app.command()
@handle_errors
def compress(
    file_path: str = typer.Argument(None, help="Path to your verbose text file (omit when using --text)"),
    text: str = typer.Option(None, "--text", "-t", help="Inline text to compress instead of a file"),
    output: str = typer.Option("dense.txt", "--out", "-o", help="Output file path"),
    provider: str = typer.Option("gemini", help="Provider: openai, gemini, anthropic, ollama"),
    base_url: str = typer.Option(None, "--base-url", help="OpenAI-compatible API base URL (e.g. http://localhost:1234/v1)"),
    model: str = typer.Option("gemini-3.5-flash", help="Model name"),
    mode: str = typer.Option("default", help="Compression mode from paper (default, bt_p1 to bt_p13)")
):
    """Compress text using TokenToken representation.
    
    Based on the research paper "Large Language Models Do Not Always Need Readable Language"
    """
    if file_path is not None and text is not None:
        console.print("[red]Provide either a file path or --text, not both.[/red]")
        raise typer.Exit(1)
    if text is not None:
        content = text
    elif file_path is not None:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        console.print("[red]No input. Provide a file path or --text '...'.[/red]")
        raise typer.Exit(1)
        
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    tt = TokenToken(provider=provider, model=model, api_key=api_key, base_url=base_url)

    try:
        compression_mode = CompressionMode(mode)
    except ValueError:
        console.print(f"[red]Invalid mode: {mode}[/red]")
        console.print("Available modes: " + ", ".join([m.value for m in CompressionMode]))
        return

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description=f"Compressing via {provider} ({model}) using {mode} mode...", total=None)
        result = tt.compress(content, mode=compression_mode)

    # Save Output
    with open(output, 'w', encoding='utf-8') as f:
        f.write(result.dense_text)

    # Print comprehensive results
    table = Table(title="TokenToken Compression Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Original Tokens", str(result.original_tokens))
    table.add_row("Compressed Tokens", str(result.compressed_tokens))
    table.add_row("Token Reduction", f"{result.savings_pct}%")
    table.add_row("Retention Ratio", f"{result.retention_ratio:.2%}")
    table.add_row("Output File", output)
    table.add_row("Compression Mode", mode)
    table.add_row("Provider", provider)
    table.add_row("Model", model)
    
    # Add readability metrics
    if result.readability_metrics:
        table.add_row("Dale-Chall Estimate", str(result.readability_metrics.get("dale_chall_estimate", "N/A")))
        table.add_row("Difficult Word Ratio", f"{result.readability_metrics.get('difficult_word_ratio', 0):.2%}")
    
    # Add efficiency metrics
    if result.efficiency_metrics:
        table.add_row("Tokens per Word", str(result.efficiency_metrics.get("tokens_per_word", "N/A")))
        table.add_row("Chars per Token", str(result.efficiency_metrics.get("chars_per_token", "N/A")))
    
    # Add validation warnings
    if result.validation and result.validation.get("cot_warning"):
        console.print(Panel(
            f"[yellow]{result.validation['cot_warning']}[/yellow]",
            title="Chain-of-Thought Tax Warning",
            border_style="yellow"
        ))
    
    console.print(table)

@app.command()
@handle_errors
def decompress(
    file_path: str = typer.Argument(..., help="Path to compressed file"),
    question: str = typer.Option(None, "--question", "-q", help="Question to answer based on compressed text"),
    provider: str = typer.Option("gemini", help="Provider: openai, gemini, anthropic, ollama"),
    base_url: str = typer.Option(None, "--base-url", help="OpenAI-compatible API base URL (e.g. http://localhost:1234/v1)"),
    model: str = typer.Option("gemini-3.5-flash", help="Model name")
):
    """Decompress and interpret compressed text."""
    with open(file_path, 'r', encoding='utf-8') as f:
        compressed_text = f.read()
    
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    tt = TokenToken(provider=provider, model=model, api_key=api_key, base_url=base_url)
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description=f"Interpreting compressed text via {provider} ({model})...", total=None)
        result = tt.decompress(compressed_text, question=question)
    
    console.print(Panel(
        result,
        title="TokenToken Decompression Result",
        border_style="green"
    ))

@app.command()
@handle_errors
def analyze(
    file_path: str = typer.Argument(..., help="Path to text file to analyze"),
    compressed_file: str = typer.Option(None, "--compressed", "-c", help="Path to compressed file for comparison")
):
    """Analyze text for compression potential and metrics."""
    with open(file_path, 'r', encoding='utf-8') as f:
        original_text = f.read()
    
    from .utils import calculate_token_efficiency, estimate_readability, calculate_compression_metrics
    
    original_tokens = count_tokens(original_text)
    efficiency = calculate_token_efficiency(original_text)
    readability = estimate_readability(original_text)
    
    table = Table(title="Text Analysis Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Original Tokens", str(original_tokens))
    table.add_row("Word Count", str(readability.get("word_count", 0)))
    table.add_row("Tokens per Word", str(efficiency.get("tokens_per_word", "N/A")))
    table.add_row("Chars per Token", str(efficiency.get("chars_per_token", "N/A")))
    table.add_row("Compression Potential", f"{efficiency.get('compression_potential', 0):.2%}")
    table.add_row("Dale-Chall Estimate", str(readability.get("dale_chall_estimate", "N/A")))
    table.add_row("Difficult Word Ratio", f"{readability.get('difficult_word_ratio', 0):.2%}")
    
    if compressed_file:
        with open(compressed_file, 'r', encoding='utf-8') as f:
            compressed_text = f.read()
        
        compressed_tokens = count_tokens(compressed_text)
        metrics = calculate_compression_metrics(original_tokens, compressed_tokens)
        
        table.add_row("Compressed Tokens", str(compressed_tokens))
        table.add_row("Token Reduction", f"{metrics['savings_pct']}%")
        table.add_row("Retention Ratio", f"{metrics['retention_ratio']:.2%}")
    
    console.print(table)

@app.command()
@handle_errors
def agent_memory(
    memories_dir: str = typer.Argument(..., help="Directory containing memory files"),
    output_dir: str = typer.Option("compressed_memories", "--out", "-o", help="Output directory"),
    provider: str = typer.Option("gemini", help="Provider: openai, gemini, anthropic, ollama"),
    base_url: str = typer.Option(None, "--base-url", help="OpenAI-compatible API base URL (e.g. http://localhost:1234/v1)"),
    model: str = typer.Option("gemini-3.5-flash", help="Model name")
):
    """Compress agent memory conversations (Section 4.5.2)."""
    import os
    from pathlib import Path
    
    memories_path = Path(memories_dir)
    if not memories_path.exists():
        console.print(f"[red]Directory not found: {memories_dir}[/red]")
        return
    
    # Read all memory files
    memory_files = list(memories_path.glob("*.txt")) + list(memories_path.glob("*.md"))
    if not memory_files:
        console.print("[red]No memory files found in directory[/red]")
        return
    
    memories = []
    for file in memory_files:
        with open(file, 'r', encoding='utf-8') as f:
            memories.append(f.read())
    
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    tt = TokenToken(provider=provider, model=model, api_key=api_key, base_url=base_url)
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description=f"Compressing {len(memories)} memory sessions...", total=None)
        result = tt.compress_for_agent_memory(memories)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save compressed memories
    for i, memory in enumerate(result.compressed_memories):
        output_file = Path(output_dir) / f"memory_{i}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(memory)
    
    # Print results
    table = Table(title="Agent Memory Compression Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Original Token Count", str(result.original_token_count))
    table.add_row("Compressed Token Count", str(result.compressed_token_count))
    table.add_row("Memory Count", str(result.memory_count))
    table.add_row("Average Compression Ratio", f"{result.avg_compression_ratio:.2%}")
    table.add_row("Output Directory", output_dir)
    
    console.print(table)

@app.command()
@handle_errors
def multi_agent(
    message_file: str = typer.Argument(..., help="Path to message file"),
    sender: str = typer.Option("agent_1", help="Sender agent name"),
    receiver: str = typer.Option("agent_2", help="Receiver agent name"),
    output: str = typer.Option("compressed_message.txt", "--out", "-o", help="Output file"),
    provider: str = typer.Option("gemini", help="Provider: openai, gemini, anthropic, ollama"),
    base_url: str = typer.Option(None, "--base-url", help="OpenAI-compatible API base URL (e.g. http://localhost:1234/v1)"),
    model: str = typer.Option("gemini-3.5-flash", help="Model name")
):
    """Compress message for multi-agent communication (Section 4.5.1)."""
    with open(message_file, 'r', encoding='utf-8') as f:
        message = f.read()
    
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    tt = TokenToken(provider=provider, model=model, api_key=api_key, base_url=base_url)
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description=f"Compressing message for {sender} -> {receiver}...", total=None)
        result = tt.compress_for_multi_agent(message, sender, receiver)
    
    # Save compressed message
    with open(output, 'w', encoding='utf-8') as f:
        f.write(result.content)
    
    # Print results
    table = Table(title="Multi-Agent Message Compression Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Sender", result.sender)
    table.add_row("Receiver", result.receiver)
    table.add_row("Original Tokens", str(result.original_tokens))
    table.add_row("Compressed Tokens", str(result.compressed_tokens))
    table.add_row("Token Reduction", f"{(1 - result.compressed_tokens/result.original_tokens)*100:.2f}%")
    table.add_row("Output File", output)
    
    console.print(table)

@app.command()
@handle_errors
def cross_model(
    compressed_file: str = typer.Argument(..., help="Path to compressed file"),
    reader_provider: str = typer.Option("openai", help="Provider for reader model"),
    reader_model: str = typer.Option("gpt-4", help="Model name for reader"),
    provider: str = typer.Option("gemini", help="Provider for compressor model"),
    base_url: str = typer.Option(None, "--base-url", help="OpenAI-compatible API base URL (e.g. http://localhost:1234/v1)"),
    model: str = typer.Option("gemini-3.5-flash", help="Model name for compressor")
):
    """Check cross-model compatibility (Section 4.4)."""
    with open(compressed_file, 'r', encoding='utf-8') as f:
        compressed_text = f.read()
    
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    tt = TokenToken(provider=provider, model=model, api_key=api_key, base_url=base_url)
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description=f"Testing cross-model compatibility...", total=None)
        result = tt.check_cross_model_compatibility(compressed_text, reader_provider, reader_model)
    
    # Print results
    table = Table(title="Cross-Model Compatibility Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Compressor Model", f"{result.compressor_provider}/{result.compressor_model}")
    table.add_row("Reader Model", f"{result.reader_provider}/{result.reader_model}")
    table.add_row("Compatibility Score", f"{result.compatibility_score:.2f}")
    table.add_row("Retention Ratio", f"{result.retention_ratio:.2%}")
    table.add_row("Notes", result.notes or "N/A")
    
    console.print(table)

@app.command()
@handle_errors
def extend_context(
    long_text_file: str = typer.Argument(..., help="Path to long text file"),
    output: str = typer.Option("compressed_long.txt", "--out", "-o", help="Output file"),
    target_tokens: int = typer.Option(200000, help="Target token count for context window"),
    provider: str = typer.Option("gemini", help="Provider: openai, gemini, anthropic, ollama"),
    base_url: str = typer.Option(None, "--base-url", help="OpenAI-compatible API base URL (e.g. http://localhost:1234/v1)"),
    model: str = typer.Option("gemini-3.5-flash", help="Model name")
):
    """Extend context window using TokenToken compression (Section 4.5.3)."""
    with open(long_text_file, 'r', encoding='utf-8') as f:
        long_text = f.read()
    
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    tt = TokenToken(provider=provider, model=model, api_key=api_key, base_url=base_url)
    
    original_tokens = count_tokens(long_text)
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description=f"Extending context window for {original_tokens} tokens...", total=None)
        result = tt.extend_context_window(long_text, target_tokens=target_tokens)
    
    # Save output
    with open(output, 'w', encoding='utf-8') as f:
        f.write(result)
    
    compressed_tokens = count_tokens(result)
    
    # Print results
    table = Table(title="Context Window Extension Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Original Tokens", str(original_tokens))
    table.add_row("Compressed Tokens", str(compressed_tokens))
    table.add_row("Token Reduction", f"{(1 - compressed_tokens/original_tokens)*100:.2f}%")
    table.add_row("Target Tokens", str(target_tokens))
    table.add_row("Output File", output)
    
    console.print(table)

@app.command()
@handle_errors
def estimate(file_path: str):
    """Estimate token count for a file."""
    with open(file_path, 'r') as f:
        text = f.read()
    tokens = count_tokens(text)
    console.print(f"[green]File contains ~{tokens} tokens.[/green]")
    console.print("[yellow]Expected reduction: ~70% (average)[/yellow]")

@app.command()
@handle_errors
def list_modes():
    """List all available compression modes from the paper."""
    table = Table(title="TokenToken Compression Modes (Paper Appendix C)")
    table.add_column("Mode", style="cyan")
    table.add_column("Paper Reference", style="magenta")
    table.add_column("Description", style="green")
    
    mode_descriptions = {
        CompressionMode.DEFAULT: ("C.1", "Default Compression Prompt"),
        CompressionMode.BT_P1: ("C.2.1", "Adaptive Symbolic Collapse"),
        CompressionMode.BT_P2: ("C.2.2", "Refined Zero-Overhead Compression"),
        CompressionMode.BT_P3: ("C.2.3", "Minimal Lossless Objective"),
        CompressionMode.BT_P4: ("C.2.4", "Structured Omnilingual Mapping"),
        CompressionMode.BT_P5: ("C.2.5", "Canonical Omnilingual-Symbolic"),
        CompressionMode.BT_P6: ("C.2.6", "Structured Mapping Control"),
        CompressionMode.BT_P7: ("C.2.7", "Canonical Compression Objective"),
        CompressionMode.BT_P8: ("C.2.8", "Fixed Symbolic Mapping Rules"),
        CompressionMode.BT_P9: ("C.2.9", "Structured Semantic Mapping"),
        CompressionMode.BT_P10: ("C.2.10", "LLM-Native Compressor"),
        CompressionMode.BT_P11: ("C.2.11", "Compact Symbolic Mapping"),
        CompressionMode.BT_P12: ("C.2.12", "Free-Emergence Attention Checklist"),
        CompressionMode.BT_P13: ("C.2.13", "ASCII Anchor Skeleton"),
    }
    
    for mode in CompressionMode:
        if mode in mode_descriptions:
            ref, desc = mode_descriptions[mode]
            table.add_row(mode.value, ref, desc)
    
    console.print(table)

if __name__ == "__main__":
    app()
