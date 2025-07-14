# -*- coding: utf-8 -*-
"""
PersonaAgent Framework Implementation for Google Colab.

This script implements the PersonaAgent framework as described in the paper
"PersonaAgent: When Large Language Model Agents Meet Personalization at Test Time"
(arXiv:2506.06254v1). It is designed to be run in a Google Colab environment
using a quantized model from HuggingFace.
"""

# Step 1: Install necessary libraries (uncomment if running in Colab)
# !pip install transformers torch accelerate bitsandbytes sentence-transformers faiss-cpu -q

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import warnings
import os
from huggingface_hub import login

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Step 2: Load the HuggingFace Model and Tokenizer
# We use a quantized version of Mistral-7B-Instruct, which is powerful
# and can run on a free-tier Colab GPU.
model_id = "mistralai/Mistral-7B-Instruct-v0.1"

# Authenticate with Hugging Face if a token is provided. This is required for
# gated models such as Mistral-7B-Instruct.
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Configure quantization to load the model in 4-bit for efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    token=hf_token if hf_token else None,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token if hf_token else None)

# Load a sentence transformer model for creating embeddings for memory retrieval
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

print("✅ Models and libraries loaded successfully!")

# Step 3: Implement the PersonaAgent Framework Components

# 3.1. Episodic Memory (as described in Sec. 2.1, Eq. 1 & 2)
class EpisodicMemory:
    """Stores and retrieves fine-grained user interaction history."""
    def __init__(self, embedding_model):
        self.interactions = []
        self.embedding_model = embedding_model
        self.index = None

    def add_interaction(self, query, response):
        """Adds a new user interaction to the memory."""
        interaction_text = f"User asked: '{query}', and their response was: '{response}'"
        self.interactions.append({"text": interaction_text})
        self._update_index()

    def _update_index(self):
        """Re-creates the FAISS index for efficient similarity search."""
        if not self.interactions:
            return
        embeddings = self.embedding_model.encode([i['text'] for i in self.interactions])
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(np.array(embeddings, dtype=np.float32))

    def retrieve(self, query_text, k=2):
        """Retrieves the top-k most similar interactions."""
        if self.index is None or not self.interactions:
            return []
        query_embedding = self.embedding_model.encode([query_text])
        _, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        return [self.interactions[i] for i in indices[0]]

# 3.2. Semantic Memory (as described in Sec. 2.1, Eq. 3)
class SemanticMemory:
    """Generates an abstracted user profile from episodic memory."""
    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer

    def generate_profile(self, episodic_history):
        """Summarizes user preferences into a coherent profile."""
        if not episodic_history:
            return "No user history available."

        history_str = "\n".join([item['text'] for item in episodic_history])
        prompt = f"""
        Based on the following user interaction history, please create a concise summary of the user's likely preferences, interests, and style.

        Interaction History:
        {history_str}

        User Profile Summary:
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=150, pad_token_id=self.tokenizer.eos_token_id)
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean up the output to get just the summary part
        return summary.split("User Profile Summary:")[1].strip()

# 3.3. Test-Time User Preference Alignment (Sec. 2.2, Algorithm 1)
class TestTimeAligner:
    """Optimizes the persona prompt based on recent interactions."""
    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer

    def _get_feedback(self, query, agent_response, ground_truth):
        """Generates feedback on how to improve the persona (Loss Gradient/Feedback Prompt)."""
        prompt = f"""
        You are a meticulous evaluator of a personalized AI agent. Your task is to provide feedback on how to improve the agent's persona (its system prompt) to better align with the user's preferences.

        - User's Query: "{query}"
        - Agent's Actual Response: "{agent_response}"
        - User's Expected (Ground Truth) Response: "{ground_truth}"

        Based on the difference, provide concise feedback on how to adjust the persona. Focus on the user's style, preferences, and interests.

        Feedback:
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=100, pad_token_id=self.tokenizer.eos_token_id)
        feedback = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return feedback.split("Feedback:")[1].strip()

    def optimize_persona(self, current_persona, recent_interactions):
        """Updates the persona using aggregated feedback (Gradient Update Prompt)."""
        all_feedback = []
        for interaction in recent_interactions:
            # For this example, we'll use a placeholder for the agent's initial response
            placeholder_agent_response = "A generic, non-personalized response."
            feedback = self._get_feedback(
                interaction['query'],
                placeholder_agent_response,
                interaction['ground_truth']
            )
            all_feedback.append(feedback)

        aggregated_feedback = "\n- ".join(all_feedback)
        update_prompt = f"""
        You are a prompt engineering assistant. Your task is to refine an AI agent's system prompt (persona) based on user feedback to improve personalization.

        - Current System Prompt (Persona):
        "{current_persona}"

        - Aggregated Feedback from Recent Interactions:
        "{aggregated_feedback}"

        Based on the feedback, generate an updated and improved system prompt. The new prompt should better capture the user's unique preferences and style.

        New System Prompt:
        """
        inputs = self.tokenizer(update_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=250, pad_token_id=self.tokenizer.eos_token_id)
        new_persona = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return new_persona.split("New System Prompt:")[1].strip()

# 3.4. The Core PersonaAgent
class PersonaAgent:
    """The main agent that integrates memory, actions, and an optimized persona."""
    def __init__(self, llm, tokenizer, embedding_model, user_history):
        self.llm = llm
        self.tokenizer = tokenizer
        self.episodic_memory = EpisodicMemory(embedding_model)
        self.semantic_memory = SemanticMemory(llm, tokenizer)
        self.aligner = TestTimeAligner(llm, tokenizer)
        self.persona = ""

        # Populate memory with user history
        for item in user_history:
            self.episodic_memory.add_interaction(item['query'], item['response'])

    def initialize_persona(self):
        """Creates the initial persona from semantic memory."""
        profile = self.semantic_memory.generate_profile(self.episodic_memory.interactions)
        # This is the initial persona prompt from Appendix B
        self.persona = f"""You are a helpful personalized assistant.
User summary: {profile}
STRICT RULES:
1. Think step-by-step about what information you need.
2. Use information from the user's past interactions to tailor your response.
3. Provide clear, concise responses that match the user's style.
"""
        print("\n----- Initial Persona Created -----")
        print(self.persona)

    def align_at_test_time(self, recent_interactions_for_alignment):
        """Runs the test-time alignment to optimize the persona."""
        print("\n----- Starting Test-Time Persona Alignment -----")
        optimized_persona = self.aligner.optimize_persona(self.persona, recent_interactions_for_alignment)
        self.persona = optimized_persona
        print("\n----- Optimized Persona -----")
        print(self.persona)

    def execute(self, new_query):
        """Handles a new user query using the full framework."""
        print(f"\n----- Executing New Query: '{new_query}' -----")

        # 1. Retrieve relevant memories (Personalized Action)
        retrieved_mems = self.episodic_memory.retrieve(new_query, k=2)
        memory_context = "\n".join([mem['text'] for mem in retrieved_mems])
        print(f"\nStep 1: Retrieved relevant memories:\n- {memory_context}")

        # 2. Construct the final prompt with the persona and context
        final_prompt = f"""<s>[INST]
        {self.persona}

        Here is some relevant context from my past interactions:
        {memory_context}

        Now, please answer the following question:
        {new_query}
        [/INST]
        """
        print("\nStep 2: Constructed the final prompt for the LLM.")

        # 3. Generate the final response
        inputs = self.tokenizer(final_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=200, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[1].strip()

        print("\n----- Final Personalized Response -----")
        print(response)
        return response


# Step 4: Run a Demonstration

if __name__ == '__main__':
    # Sample data representing a user's history and recent interactions
    # This user is interested in sci-fi and prefers concise, factual answers.
    user_historical_data = [
        {"query": "What's a good sci-fi movie to watch?", "response": "Blade Runner 2049 has great visuals."},
        {"query": "Tell me about the book 'Dune'.", "response": "It's a foundational text of science fiction, exploring politics, religion, and ecology."},
        {"query": "Any thoughts on the 'Foundation' series by Asimov?", "response": "A classic. The concept of psychohistory is brilliant."}
    ]

    # A recent batch of interactions used to optimize the persona at test-time
    # Note: The 'ground_truth' reflects the user's specific style (e.g., direct, mentioning adaptations)
    recent_interactions_for_alignment = [
        {
            "query": "Should I watch the movie 'Ender's Game'?",
            "ground_truth": "Yes, but the book is far more detailed in its exploration of character psychology."
        },
        {
            "query": "Is 'The Expanse' TV show worth watching?",
            "ground_truth": "Absolutely. It's known for its realistic physics and complex political narrative. One of the best modern sci-fi adaptations."
        }
    ]

    # --- Initialization ---
    my_agent = PersonaAgent(model, tokenizer, embedding_model, user_historical_data)

    # --- Create Initial Persona ---
    # The agent first creates a general persona based on the user's entire history.
    my_agent.initialize_persona()

    # --- Test-Time Alignment ---
    # The agent then refines its persona using a few recent examples to better capture
    # the user's current preferences and style (e.g., comparing books to movies).
    my_agent.align_at_test_time(recent_interactions_for_alignment)

    # --- Execution ---
    # Finally, the agent uses its newly optimized persona to answer a new question.
    new_user_query = "What do you think about the 'Hyperion Cantos' series?"
    final_answer = my_agent.execute(new_user_query)
