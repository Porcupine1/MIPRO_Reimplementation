from typing import List, Dict, Any, Optional, Tuple
from backend import LMBackend
from config import get_active_config


class GroundingHelper:
    """
    grounding: characterize task dynamics from dataset and program

    - characterize patterns in the raw dataset
    - summarize the program's control flow
    - bootstrapped program demonstrations
    - per-stage prompts with scores

    Generates dataset and program summaries during initialization.
    """

    def __init__(
        self,
        lm: LMBackend = None,
        train_examples: Optional[List[Dict[str, Any]]] = None,
        program: Optional[Any] = None,
    ):
        self.lm = lm or LMBackend()
        self.program = program  # Store program reference for later use

        # Generate summaries during initialization
        if train_examples is not None:
            self.dataset_summary = self.summarize_dataset(train_examples)
        else:
            self.dataset_summary = None

        if program is not None:
            self.program_summary = self.summarize_program(program)
        else:
            self.program_summary = None

    def summarize_dataset(
        self, examples: List[Dict[str, Any]], n_samples: int = None, batch_size: int = 5
    ) -> str:
        """
        Characterize patterns in the dataset using iterative batch process.

        Iterates over dataset in batches. If LM outputs "COMPLETE" 5 times consecutively,
        stops and summarizes accumulated observations.

        Args:
            examples: List of dataset examples
            n_samples: Number of samples to analyze (defaults to min(50, MAX_EXAMPLES))
            batch_size: Size of batches for processing

        Returns a summary of:
        - question types
        - answer types
        - common patterns
        - task nature
        """
        # Default n_samples to respect MAX_EXAMPLES from active tier config
        if n_samples is None:
            cfg = get_active_config()
            n_samples = min(50, cfg.max_examples)
        
        # Compute deterministic, LM-free dataset stats (answer lengths, question lengths, etc.)
        stats_block = self._build_dataset_stats_block(examples[: min(n_samples, len(examples))])

        observations = []
        complete_count = 0
        max_complete = 5

        # Process examples in batches
        num_batches = min(
            (n_samples + batch_size - 1) // batch_size, 20
        )  # Limit to 20 batches

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples, len(examples))

            if start_idx >= len(examples):
                break

            batch_examples = examples[start_idx:end_idx]

            # Format batch examples
            batch_text = []
            for i, ex in enumerate(batch_examples, 1):
                # Be robust to different example types:
                # - dict-like examples with "question"/"answer" keys (expected)
                # - raw strings (treat as the question text)
                if isinstance(ex, dict):
                    question = ex.get("question", "")
                    answer = ex.get("answer", "")
                else:
                    question = str(ex)
                    answer = ""
                batch_text.append(f"{i}. Question: {question} Answer: {answer}")

            # Build dataset descriptor prompt
            existing_obs = "\n".join(observations) if observations else "None yet."

            descriptor_prompt = f"""Given several examples from a dataset please write observations about trends that hold for most or all of the samples. I will also provide you with a few observations I have already made. Please add your own observations or if you feel the observations are comprehensive say 'COMPLETE'. Some areas you may consider in your observations: topics, content, syntax, conciseness, etc. It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative

Examples from dataset:
{chr(10).join(batch_text)}

Existing observations:
{existing_obs}

Your observations (or 'COMPLETE' if comprehensive):"""

            response = self.lm.generate(descriptor_prompt, temperature=0.7)
            response = response.strip()

            # Check if response is "COMPLETE"
            if response.upper() == "COMPLETE" or "COMPLETE" in response.upper():
                complete_count += 1
                if complete_count >= max_complete:
                    break
            else:
                complete_count = 0  # Reset counter if we get a non-COMPLETE response
                if response:  # Only add non-empty observations
                    observations.append(response)

        # If no observations collected, use a simple fallback
        if not observations:
            # Fallback: simple analysis of first few examples
            sample_text = []
            for i, ex in enumerate(examples[: min(5, len(examples))], 1):
                if isinstance(ex, dict):
                    question = ex.get("question", "")
                    answer = ex.get("answer", "")
                else:
                    question = str(ex)
                    answer = ""
                sample_text.append(f"{i}. Question: {question} Answer: {answer}")

            observations.append(f"Sample examples: {chr(10).join(sample_text)}")

        # Summarize accumulated observations
        observations_text = "\n".join(observations)

        summarizer_prompt = f"""Given a series of observations I have made about my dataset, please summarize them into a brief 2-3 sentence summary which highlights only the most important details.

Observations:
{observations_text}

Summary:"""

        summary = self.lm.generate(summarizer_prompt, temperature=0.3)
        summary = summary.strip()

        # Append deterministic stats so downstream proposers can enforce conciseness.
        # This avoids relying on the LM to "notice" length distribution.
        if stats_block:
            return f"{summary}\n\n{stats_block}".strip()
        return summary

    @staticmethod
    def _percentiles(values: List[int], percentiles: List[int]) -> Dict[int, int]:
        """
        Compute simple nearest-rank percentiles (no interpolation).
        Percentiles should be integers in [0, 100].
        """
        if not values:
            return {p: 0 for p in percentiles}
        xs = sorted(values)
        n = len(xs)
        out: Dict[int, int] = {}
        for p in percentiles:
            if p <= 0:
                out[p] = xs[0]
                continue
            if p >= 100:
                out[p] = xs[-1]
                continue
            # nearest-rank method
            k = int(round((p / 100) * (n - 1)))
            k = max(0, min(n - 1, k))
            out[p] = xs[k]
        return out

    @staticmethod
    def _safe_str_len(s: Any) -> int:
        try:
            return len(str(s))
        except Exception:
            return 0

    @staticmethod
    def _word_count(s: Any) -> int:
        try:
            return len(str(s).strip().split())
        except Exception:
            return 0

    def _answer_and_question_lengths(
        self, examples: List[Dict[str, Any]]
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Returns: answer_chars, answer_words, question_chars, question_words
        """
        a_chars: List[int] = []
        a_words: List[int] = []
        q_chars: List[int] = []
        q_words: List[int] = []

        for ex in examples:
            if not isinstance(ex, dict):
                continue
            a = ex.get("answer", "")
            q = ex.get("question", "")
            a_chars.append(self._safe_str_len(a))
            a_words.append(self._word_count(a))
            q_chars.append(self._safe_str_len(q))
            q_words.append(self._word_count(q))

        return a_chars, a_words, q_chars, q_words

    def _build_dataset_stats_block(self, examples: List[Dict[str, Any]]) -> str:
        """
        Build a compact, deterministic block describing typical output sizes.
        This is intended to guide instruction generation to match the dataset's
        expected answer length (and avoid verbose/refusal-style outputs).
        """
        a_chars, a_words, q_chars, q_words = self._answer_and_question_lengths(examples)
        if not a_chars and not q_chars:
            return ""

        ps = [10, 50, 90]
        a_chars_p = self._percentiles(a_chars, ps)
        a_words_p = self._percentiles(a_words, ps)
        q_chars_p = self._percentiles(q_chars, ps)
        q_words_p = self._percentiles(q_words, ps)

        # Heuristic: most HotpotQA answers are short entity strings. Use p90 to set guidance.
        # Keep it conservative: allow a bit of headroom over p90.
        answer_words_target_max = max(1, int(a_words_p[90] * 1.5))
        answer_chars_target_max = max(16, int(a_chars_p[90] * 1.5))

        # Query guidance is less grounded (there is no "gold query"), so keep it mild.
        query_words_target_max = max(6, int(q_words_p[50] * 0.4))  # ~short phrase from question
        query_words_target_max = min(query_words_target_max, 18)

        return (
            "DATASET SIZE / OUTPUT-LENGTH GUIDANCE (computed from ground-truth answers):\n"
            f"- Answer length (words) p10/p50/p90 = {a_words_p[10]}/{a_words_p[50]}/{a_words_p[90]}\n"
            f"- Answer length (chars) p10/p50/p90 = {a_chars_p[10]}/{a_chars_p[50]}/{a_chars_p[90]}\n"
            f"- Question length (words) p10/p50/p90 = {q_words_p[10]}/{q_words_p[50]}/{q_words_p[90]}\n"
            f"- Question length (chars) p10/p50/p90 = {q_chars_p[10]}/{q_chars_p[50]}/{q_chars_p[90]}\n"
            "\n"
            "When writing module instructions, enforce outputs that match the dataset:\n"
            f"- Answer module: output ONLY the final answer; target <= ~{answer_words_target_max} words "
            f"(~{answer_chars_target_max} chars), no explanations.\n"
            f"- Query module: output ONLY a short Wikipedia search query (target <= ~{query_words_target_max} words), "
            "no extra text.\n"
        ).strip()

    def summarize_program(self, program) -> str:
        """
        Summarize the program's control flow by analyzing actual program code.

        Uses the Program Summarizer Prompt to analyze what type of task
        the program solves and how it works.

        Returns description of what each module does and program flow.
        Program-agnostic: analyzes any program structure dynamically.
        """
        program_code = self.build_program_code(program)

        # Use Program Summarizer Prompt
        summarizer_prompt = f"""Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe what type of task this program appears to be designed to solve, and how it appears to work. Analyze the code structure, module names, and method calls to infer the program flow.

{program_code}

Description:"""

        summary = self.lm.generate(summarizer_prompt, temperature=0.3)
        return summary.strip()

    @staticmethod
    def build_program_code(program) -> str:
        """
        Build a pseudo-code representation of the program, including:
        - class definition
        - module composition
        - main entry point / forward method

        This helper is shared between GroundingHelper and other components
        (e.g., InstructionProposer) to avoid duplicate introspection logic.
        """
        import inspect

        # Get program source code
        try:
            program_source = inspect.getsource(program.__class__)
        except (OSError, TypeError):
            # Fallback: try to get module info
            program_source = f"Program class: {program.__class__.__name__}\n"
            if hasattr(program, "get_module_names"):
                program_source += f"Modules: {program.get_module_names()}\n"
            program_source += "Methods: " + ", ".join(
                [m for m in dir(program) if not m.startswith("_")]
            )

        # Get module information dynamically
        modules_info = []
        if hasattr(program, "get_module_names"):
            for module_name in program.get_module_names():
                if hasattr(program, "modules") and module_name in program.modules:
                    module = program.modules[module_name]
                    module_type = type(module).__name__
                    modules_info.append(f"- {module_name} ({module_type})")

        # Analyze the forward/main method to understand flow
        flow_description = "Not available"
        if hasattr(program, "forward"):
            try:
                forward_source = inspect.getsource(program.forward)
                # Extract a simplified view of the forward method
                flow_description = f"Forward method implementation:\n{forward_source}"
            except (OSError, TypeError):
                # Fallback: describe method signature
                forward_sig = inspect.signature(program.forward)
                flow_description = f"Forward method signature: forward{forward_sig}"

        # Build program pseudo-code representation (program-agnostic)
        program_code = f"""
Program Class: {program.__class__.__name__}

Program Structure:
{program_source}

Modules:
{chr(10).join(modules_info) if modules_info else "No modules found"}

Main Entry Point:
{flow_description}
"""

        return program_code
