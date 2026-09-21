# tokentoken/sdk.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
from .prompts import CompressionMode, PROMPTS
from .utils import (
    count_tokens, 
    check_breakeven_threshold, 
    chunk_text, 
    calculate_compression_metrics,
    estimate_readability,
    calculate_token_efficiency,
    validate_compression_result
)
from .router import execute_compression

class CompressionResult(BaseModel):
    """Result of compression with comprehensive metrics."""
    dense_text: str
    original_tokens: int
    compressed_tokens: int
    savings_pct: float
    retention_ratio: float
    readability_metrics: Dict[str, Any]
    efficiency_metrics: Dict[str, Any]
    validation: Dict[str, Any]

class CrossModelCompatibility(BaseModel):
    """Cross-model compatibility matrix based on Section 4.4."""
    compressor_model: str
    compressor_provider: str
    reader_model: str
    reader_provider: str
    compatibility_score: float
    retention_ratio: float
    notes: Optional[str] = None

class AgentMemoryResult(BaseModel):
    """Result for agent memory compression based on Section 4.5.2."""
    compressed_memories: List[str]
    original_token_count: int
    compressed_token_count: int
    memory_count: int
    avg_compression_ratio: float

class MultiAgentMessage(BaseModel):
    """Message format for multi-agent communication based on Section 4.5.1."""
    sender: str
    receiver: str
    content: str
    compressed: bool = False
    original_tokens: int = 0
    compressed_tokens: int = 0

class TokenToken:
    def __init__(self, provider: str, model: str, api_key: str = None, host: str = None, base_url: str = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.host = host
        self.base_url = base_url

    def compress(self, text: str, mode: CompressionMode = CompressionMode.DEFAULT) -> CompressionResult:
        """Compress text into dense representation.
        
        Based on task-agnostic compression approach:
        - Uses specified compression mode (BT-P1 to BT-P13)
        - Handles ultra-long documents via chunking
        """
        original_tokens = count_tokens(text)
        check_breakeven_threshold(original_tokens)

        system_prompt = PROMPTS[mode]
        
        # Paper Appendix E.4.2: Handling ultra-long documents by chunking
        chunks = chunk_text(text, chunk_size_tokens=100000)
        dense_chunks = []
        
        for chunk in chunks:
            dense = execute_compression(
                provider=self.provider,
                model=self.model,
                system_prompt=system_prompt,
                user_text=chunk,
                api_key=self.api_key,
                host=self.host,
                base_url=self.base_url
            )
            dense_chunks.append(dense)
            
        final_dense_text = "\n\n".join(dense_chunks)
        compressed_tokens = count_tokens(final_dense_text)
        
        # Calculate comprehensive metrics
        metrics = calculate_compression_metrics(original_tokens, compressed_tokens)
        readability = estimate_readability(final_dense_text)
        efficiency = calculate_token_efficiency(final_dense_text)
        validation = validate_compression_result(text, final_dense_text, self.provider, self.model)
        
        return CompressionResult(
            dense_text=final_dense_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            savings_pct=metrics["savings_pct"],
            retention_ratio=metrics["retention_ratio"],
            readability_metrics=readability,
            efficiency_metrics=efficiency,
            validation=validation
        )

    def decompress(self, compressed_text: str, question: str = None) -> str:
        """Interpret compressed text using an LLM.
        
        LLMs can interpret compressed text without external codebooks.
        If a question is provided, answers it based on the compressed context.
        """
        if question:
            prompt = f"""You are an intelligent assistant that can understand dense compressed text.
The following text is compressed to sacrifice human readability for information density.
Your task is to answer the question based on this compressed text.

Compressed Text:
{compressed_text}

Question: {question}

Answer the question based only on the compressed text. If the information is not available, state that."""
        else:
            prompt = f"""You are an intelligent assistant that can understand dense compressed text.
The following text is compressed to sacrifice human readability for information density.
Please provide a clear explanation of what this compressed text represents.

Compressed Text:
{compressed_text}

Explain the meaning and content of this compressed text in natural language."""
        
        return execute_compression(
            provider=self.provider,
            model=self.model,
            system_prompt="You are an expert in interpreting compressed text.",
            user_text=prompt,
            api_key=self.api_key,
            host=self.host,
            base_url=self.base_url
        )

    def compress_for_agent_memory(self, memories: List[str], session_id: str = None) -> AgentMemoryResult:
        """Compress agent memory conversations.
        
        Each memory session is compressed independently to preserve individual context.
        """
        original_token_count = sum(count_tokens(memory) for memory in memories)
        compressed_memories = []
        
        for i, memory in enumerate(memories):
            # Use a specialized prompt for agent memory compression
            memory_prompt = f"""Compress the following agent memory conversation into dense format.
Preserve all key information, entities, actions, and temporal relationships.
Session ID: {session_id or f"session_{i}"}

Memory to compress:
{memory}"""
            
            compressed = execute_compression(
                provider=self.provider,
                model=self.model,
                system_prompt=PROMPTS[CompressionMode.BT_P2],  # Use zero-overhead for memory
                user_text=memory_prompt,
                api_key=self.api_key,
                host=self.host,
                base_url=self.base_url
            )
            compressed_memories.append(compressed)
        
        compressed_token_count = sum(count_tokens(memory) for memory in compressed_memories)
        avg_compression_ratio = compressed_token_count / original_token_count if original_token_count > 0 else 0
        
        return AgentMemoryResult(
            compressed_memories=compressed_memories,
            original_token_count=original_token_count,
            compressed_token_count=compressed_token_count,
            memory_count=len(memories),
            avg_compression_ratio=avg_compression_ratio
        )

    def compress_for_multi_agent(self, message: str, sender: str, receiver: str) -> MultiAgentMessage:
        """Compress message for multi-agent communication.
        
        Compresses inter-agent messages to reduce context overhead while preserving
        actionable information for task coordination.
        """
        original_tokens = count_tokens(message)
        
        # Use a compression mode optimized for agent communication
        multi_agent_prompt = f"""Compress the following inter-agent message for multi-agent communication.
Preserve all actionable information, commands, data, and coordination details.
Sender: {sender}
Receiver: {receiver}

Message to compress:
{message}"""
        
        compressed = execute_compression(
            provider=self.provider,
            model=self.model,
            system_prompt=PROMPTS[CompressionMode.BT_P5],  # Use canonical for agent communication
            user_text=multi_agent_prompt,
            api_key=self.api_key,
            host=self.host,
            base_url=self.base_url
        )
        
        compressed_tokens = count_tokens(compressed)
        
        return MultiAgentMessage(
            sender=sender,
            receiver=receiver,
            content=compressed,
            compressed=True,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens
        )

    def check_cross_model_compatibility(self, compressed_text: str, reader_provider: str, reader_model: str) -> CrossModelCompatibility:
        """Check compatibility of compressed text across models.
        
        Tests whether compressed text from this model can be understood by another model.
        """
        # Create a test question to verify comprehension
        test_prompt = f"""You are testing cross-model comprehension of compressed text.
The following text was compressed by a different model. Please answer this simple question:
What is the main topic of this compressed text?

Compressed Text:
{compressed_text}

Provide a brief answer."""
        
        response = execute_compression(
            provider=reader_provider,
            model=reader_model,
            system_prompt="You are testing cross-model comprehension.",
            user_text=test_prompt,
            api_key=self.api_key,
            host=self.host,
            base_url=self.base_url
        )
        
        # Simple heuristic to estimate compatibility based on response quality
        # In a real implementation, this would use more sophisticated evaluation
        response_tokens = count_tokens(response)
        original_tokens = count_tokens(compressed_text)
        
        # If the response is meaningful and not too short, assume good compatibility
        if response_tokens > 10 and "I don't understand" not in response:
            compatibility_score = 0.85  # Good compatibility
        else:
            compatibility_score = 0.6   # Moderate compatibility
        
        return CrossModelCompatibility(
            compressor_model=self.model,
            compressor_provider=self.provider,
            reader_model=reader_model,
            reader_provider=reader_provider,
            compatibility_score=compatibility_score,
            retention_ratio=original_tokens / original_tokens,  # 1.0 since we're not compressing further
            notes="Tested via simple comprehension check"
        )

    def extend_context_window(self, long_text: str, target_tokens: int = 200000) -> str:
        """Extend context window using compression.
        
        For texts exceeding the model's context window, compress chunks and concatenate.
        """
        # Split the long text into chunks that can be compressed
        chunks = chunk_text(long_text, chunk_size_tokens=200000)
        compressed_chunks = []
        
        for chunk in chunks:
            compressed = execute_compression(
                provider=self.provider,
                model=self.model,
                system_prompt=PROMPTS[CompressionMode.BT_P13],  # Use ASCII anchor for structured compression
                user_text=chunk,
                api_key=self.api_key,
                host=self.host,
                base_url=self.base_url
            )
            compressed_chunks.append(compressed)
        
        # Concatenate compressed chunks
        final_compressed = "\n\n".join(compressed_chunks)
        
        # Check if we've achieved the target compression
        final_tokens = count_tokens(final_compressed)
        if final_tokens > target_tokens:
            # If still too long, apply another round of compression
            final_compressed = execute_compression(
                provider=self.provider,
                model=self.model,
                system_prompt=PROMPTS[CompressionMode.BT_P2],  # Use zero-overhead for extreme compression
                user_text=final_compressed,
                api_key=self.api_key,
                host=self.host,
                base_url=self.base_url
            )
        
        return final_compressed

    def get_compression_analytics(self, original: str, compressed: str) -> Dict[str, Any]:
        """Get detailed analytics for compression results based on paper's evaluation metrics."""
        original_tokens = count_tokens(original)
        compressed_tokens = count_tokens(compressed)
        
        metrics = calculate_compression_metrics(original_tokens, compressed_tokens)
        readability = estimate_readability(compressed)
        efficiency = calculate_token_efficiency(compressed)
        
        # Chain-of-Thought Tax analysis for compression
        cot_tax_analysis = {
            "estimated_cot_multiplier": 1 + (1 - metrics["retention_ratio"]) * 0.3,
            "recommended_compression_range": "60-80% retention for optimal balance",
            "warning": "Extreme compression (<30% retention) may increase output tokens"
        }
        
        return {
            "compression_metrics": metrics,
            "readability_analysis": readability,
            "efficiency_analysis": efficiency,
            "cot_tax_analysis": cot_tax_analysis,
            "paper_reference": "Based on compression research"
        }
