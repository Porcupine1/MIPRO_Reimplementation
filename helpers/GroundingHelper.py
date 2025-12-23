from typing import List, Dict, Any, Optional
from backend import LMBackend


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
        self, examples: List[Dict[str, Any]], n_samples: int = 50, batch_size: int = 5
    ) -> str:
        """
        Characterize patterns in the dataset using iterative batch process.

        Iterates over dataset in batches. If LM outputs "COMPLETE" 5 times consecutively,
        stops and summarizes accumulated observations.

        Returns a summary of:
        - question types
        - answer types
        - common patterns
        - task nature
        """
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
                question = ex.get("question", "")  # Truncate for brevity
                answer = ex.get("answer", "")
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
                question = ex.get("question", "")
                answer = ex.get("answer", "")
                sample_text.append(f"{i}. Question: {question} Answer: {answer}")

            observations.append(f"Sample examples: {chr(10).join(sample_text)}")

        # Summarize accumulated observations
        observations_text = "\n".join(observations)

        summarizer_prompt = f"""Given a series of observations I have made about my dataset, please summarize them into a brief 2-3 sentence summary which highlights only the most important details.

Observations:
{observations_text}

Summary:"""

        summary = self.lm.generate(summarizer_prompt, temperature=0.3)
        return summary.strip()

    def summarize_program(self, program) -> str:
        """
        Summarize the program's control flow by analyzing actual program code.

        Uses the Program Summarizer Prompt to analyze what type of task
        the program solves and how it works.

        Returns description of what each module does and program flow.
        Program-agnostic: analyzes any program structure dynamically.
        """
        import inspect

        # Get program source code
        try:
            program_source = inspect.getsource(program.__class__)
        except (OSError, TypeError):
            # Fallback: try to get module info
            program_source = f"Program class: {program.__class__.__name__}\n"
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

        # Use Program Summarizer Prompt
        summarizer_prompt = f"""Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe what type of task this program appears to be designed to solve, and how it appears to work. Analyze the code structure, module names, and method calls to infer the program flow.

{program_code}

Description:"""

        summary = self.lm.generate(summarizer_prompt, temperature=0.3)
        return summary.strip()
