"""
Knowledge distillation from a strong teacher LLM into 4 role-specialized
LoRA students.

Pipeline:
  1. generate_scenarios() — produce N diverse (circuit, GNN preds, KG context)
     scenarios drawn from real benchmark netlists + the deterministic pipeline
  2. teacher_distill.py — Groq Llama-3.3-70B reads each scenario through one
     role's system prompt and emits a structured expert response
  3. train_student.py — SFT a Qwen2.5-7B LoRA on each role's (scenario, teacher
     response) pairs
  4. vllm_serve.py — Modal vLLM with --enable-lora; per-request adapter routing
  5. autogen_backend wired to vLLM → real RoundRobinGroupChat with specialist
     student models per role

This is task-specific distillation, NOT vanilla fine-tuning. The supervision
signal is the teacher's high-quality reasoning over grounded inputs, not labeled
human ground truth (which doesn't exist for "good DRC advice"). The students
learn to mimic the teacher's *role-specific reasoning style* on EDA grounded
contexts.
"""

from .roles import ROLES, Role

__all__ = ["ROLES", "Role"]
