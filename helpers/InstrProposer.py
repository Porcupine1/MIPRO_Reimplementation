from typing import List, Dict, Any, Optional
import random
import logging
from backend import LMBackend
from .GroundingHelper import GroundingHelper
from config import N_INSTRUCTION_CANDIDATES
from programs import QAProgram


logger = logging.getLogger(__name__)


# Programming tips for instruction generation
PROMPTING_TIPS = {
    "none": "",
    "creative": "Don't be afraid to be creative!",
    "simple": "Keep the instruction clear and concise.",
    "description": "Make sure your instruction is very informative and descriptive.",
    "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
    "persona": "Provide the LM with a persona that is relevant to the task (ie. \"You are a ...\")"
}


class InstructionProposer:
    """
    Generates instruction candidates for modules using meta-prompting.

    - Uses bootstrapped demonstrations from successful traces
    - Grounds proposals in dataset/program characteristics
    - Meta-optimizes proposal strategy
    """
    
    def __init__(
        self,
        lm: LMBackend = None,
        train_examples: Optional[List[Dict[str, Any]]] = None,
        program: Optional[QAProgram] = None
    ):
        self.lm = lm or LMBackend()
        # Initialize grounding helper with training set and program to generate summaries
        self.grounding_helper = GroundingHelper(
            lm=self.lm,
            train_examples=train_examples,
            program=program
        )
    
    def _get_default_instruction(self, module_name: str) -> str:
        """Get default instruction for a module"""
        defaults = {
            "query": "Generate a search query to answer the question.",
            "answer": "Answer the question based on the context."
        }
        return defaults.get(module_name, "Complete the task.")

    def _format_demo_for_prompt(self, demo: Dict[str, Any], module_name: str) -> str:
        """
        Format a demo example. Uses field prefixes.
        """
        lines = []
        
        # Handle both trace format and flat format
        if "input" in demo:
            demo_input = demo["input"]
            demo_output = demo.get("output", "")
        else:
            demo_input = demo
            demo_output = demo.get("output") or demo.get(module_name, "")
        
        # Format inputs with prefixes
        for key, value in demo_input.items():
            if key not in ["output", module_name]:  # Skip output fields
                prefix = key.upper().replace("_", " ")
                lines.append(f"{prefix}: {value}")
        
        # Format output
        if demo_output:
            output_prefix = module_name.upper().replace("_", " ")
            lines.append(f"{output_prefix}: {demo_output}")
        
        return "\n".join(lines)

    def _get_module_task_description(self, module_name: str, overall_task: str) -> str:
        """Get task description specific to a module."""
        module_tasks = {
            "query": (
                "Given a question, generate an effective search query that will "
                f"retrieve relevant context. Overall task: {overall_task}"
            ),
            "answer": (
                "Given a question and retrieved context, generate an accurate answer. "
                f"Overall task: {overall_task}"
            ),
        }
        return module_tasks.get(module_name, overall_task)

    def _get_module_role_description(
        self,
        module_name: str,
        task_desc: str,
        program_summ: Optional[str] = None,
    ) -> str:
        """
        Get description of the specific module's role in the program.

        Ground this description directly in the overall task description,
        optionally enriched with a short note that this is one stage in
        a multi-step pipeline.
        """
        base_desc = self._get_module_task_description(module_name, task_desc)

        if program_summ:
            return (
                f"{base_desc} This module is a distinct stage in a multi-step pipeline "
                f"described above, and should focus on its part of the task only."
            )

        return base_desc

    def _get_program_code(self, program: Optional[QAProgram] = None) -> str:
        """
        Get program code representation for grounding.
        Delegates to GroundingHelper.build_program_code to avoid duplicating
        introspection logic.
        """
        if program is None:
            return ""

        return GroundingHelper.build_program_code(program)

    def _build_grounded_meta_prompt(
        self,
        module_name: str,
        use_task_demos: bool = True,
        bootstrapped_demos: Optional[List[Dict[str, Any]]] = None,
        dataset_summ: Optional[str] = None,
        program_summ: Optional[str] = None,
        program_code: Optional[str] = None,
        module_description: Optional[str] = None,
        basic_instruction: Optional[str] = None,
        tip: Optional[str] = None,
        use_dataset_summary: bool = True,
        use_program_aware: bool = True,
        use_tip: bool = True,
    ) -> str:
        """
        Build meta-prompt.
        """
    
        # Main instruction
        parts = [
            "Use the information below to learn about a task that we are trying to solve using calls to an LM, "
            "then generate a new instruction that will be used to prompt a Language Model to better solve the task."
        ]
        parts.append("")
        
        # DATASET SUMMARY (if enabled)
        if use_dataset_summary and dataset_summ:
            parts.append("DATASET SUMMARY:")
            parts.append(dataset_summ)
            parts.append("")
        
        # PROGRAM CODE (if program-aware)
        if use_program_aware and program_code:
            parts.append("PROGRAM CODE:")
            parts.append(program_code)
            parts.append("")
        
        # PROGRAM DESCRIPTION (if program-aware)
        if use_program_aware and program_summ:
            parts.append("PROGRAM DESCRIPTION:")
            parts.append(program_summ)
            parts.append("")
        
        # MODULE DESCRIPTION (if program-aware)
        if use_program_aware and module_description:
            parts.append("MODULE DESCRIPTION:")
            parts.append(module_description)
            parts.append("")
        
        # TASK DEMO(S) (if enabled)
        if use_task_demos and bootstrapped_demos:
            parts.append("TASK DEMO(S):")
            # Format examples similar to DSPy's create_example_string
            for i, demo in enumerate(bootstrapped_demos[:3], 1):  # Limit to 3 like DSPy
                demo_str = self._format_demo_for_prompt(demo, module_name)
                parts.append(demo_str)
                if i < len(bootstrapped_demos[:3]):
                    parts.append("")
            parts.append("")
        
        # BASIC INSTRUCTION (always included - this is the original instruction)
        if basic_instruction:
            parts.append("BASIC INSTRUCTION:")
            parts.append(basic_instruction)
            parts.append("")
        
        # TIP (if enabled)
        if use_tip and tip and tip in PROMPTING_TIPS:
            tip_text = PROMPTING_TIPS[tip]
            if tip_text:  # Only add if tip is not "none"
                parts.append("TIP:")
                parts.append(tip_text)
                parts.append("")
        
        # Output field prefix
        parts.append("PROPOSED INSTRUCTION:")
        
        return "\n".join(parts)

    def propose_instruction(
        self,
        module_name: str,
        task_desc: str,
        bootstrapped_demos: Optional[List[Dict[str, Any]]] = None,
        dataset_summ: Optional[str] = None,
        program_summ: Optional[str] = None,
        program_code: Optional[str] = None,
        module_description: Optional[str] = None,
        basic_instruction: Optional[str] = None,
        tip: Optional[str] = None,
        program_aware: bool = True,
        use_dataset_summary: bool = True,
        use_task_demos: bool = True,
        use_tip: bool = True,
    ) -> str:
        """
        Generate a single instruction candidate.
        """
        
        # Use summaries from grounding_helper if not provided
        if dataset_summ is None:
            dataset_summ = self.grounding_helper.dataset_summary
        if program_summ is None:
            program_summ = self.grounding_helper.program_summary
        
        # Get program code if program-aware
        if program_aware and program_code is None:
            # Try to get program from grounding_helper if available
            if hasattr(self.grounding_helper, 'program') and self.grounding_helper.program:
                program_code = self._get_program_code(self.grounding_helper.program)
            else:
                program_code = ""
        
        # Get module description if program-aware: use task description
        if program_aware and module_description is None and program_summ:
            module_description = self._get_module_role_description(
                module_name, task_desc, program_summ
            )
        
        # Get basic instruction (original instruction)
        if basic_instruction is None:
            basic_instruction = self._get_default_instruction(module_name)
        
        # Randomly select a tip if not provided
        if tip is None and use_tip:
            tip = random.choice(list(PROMPTING_TIPS.keys()))
        
        # Build meta-prompt
        meta_prompt = self._build_grounded_meta_prompt(
            module_name=module_name,
            bootstrapped_demos=bootstrapped_demos,
            dataset_summ=dataset_summ,
            program_summ=program_summ,
            program_code=program_code,
            module_description=module_description,
            basic_instruction=basic_instruction,
            tip=tip,
            use_dataset_summary=use_dataset_summary,
            use_program_aware=program_aware,
            use_task_demos=use_task_demos,
            use_tip=use_tip,
        )
        
        # Generate instruction with retries
        logger.info(
            "Proposing instruction for module '%s' (tip=%s, use_task_demos=%s)",
            module_name,
            tip,
            bool(bootstrapped_demos),
        )
        MAX_TRIES = 3
        for _ in range(MAX_TRIES):
            instruction = self.lm.generate(
                meta_prompt, 
                temperature=0.7,  # Match todo.txt specification
            )
            instruction = self._parse_instruction(instruction)
            if instruction:
                logger.debug("Proposed instruction for '%s': %s", module_name, instruction)
                return instruction
        
        # Fallback to default if generation fails
        fallback = basic_instruction or self._get_default_instruction(module_name)
        logger.warning(
            "Falling back to basic instruction for module '%s' after %d failed generations",
            module_name,
            MAX_TRIES,
        )
        return fallback

    def _parse_instruction(self, instruction: str) -> str:
        """
        Parse instruction from LLM response
        Only keep everything after and including the first occurrence of "instruction:" (case-insensitive).
        """
        import re

        # Case-insensitive search for the first occurrence of "instruction:"
        pattern = re.compile(r"instruction\s*:\s*", re.IGNORECASE)
        match = pattern.search(instruction)
        if match:
            instruction = instruction[match.end():]

        # Remove code block markers if present
        instruction = instruction.strip()
        if instruction.startswith("```"):
            lines = instruction.split("\n")
            if len(lines) > 1:
                instruction = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else instruction

        # Remove quotes
        instruction = instruction.strip('"').strip("'")

        return instruction.strip()

    def propose_for_all_modules(
        self,
        program: QAProgram,
        task_desc: str,
        bootstrapped_demos: Dict[str, List[Dict[str, Any]]],
        dataset_summ: Optional[str] = None,
        program_summ: Optional[str] = None,
        n_candidates: int = N_INSTRUCTION_CANDIDATES,
        tip: Optional[str] = None,
        program_aware: bool = True
    ) -> Dict[int, List[str]]:
        """
        Generate instruction candidates for all modules in a program.
        
        For each predictor in the program:
        - Determines how many instructions to generate (n_candidates)
        - Gets the original instruction from the module (at index 0)
        - For each instruction candidate:
          - Randomly selects a prompting tip (if tip not provided)
          - Calls propose_instruction
        
        In propose_instruction:
        - Gathers task demos from few-shot example candidates (bootstrapped_demos)
        - If program-aware: describes the overall program and specific module's role
        - Generates instruction using the prompt model
        
        Args:
            program: QAProgram instance
            task_desc: Overall task description
            bootstrapped_demos: Dict mapping module names to lists of successful traces (few-shot example candidates)
            dataset_summ: Dataset summary (optional, uses grounding_helper.dataset_summary if None)
            program_summ: Program summary (optional, uses grounding_helper.program_summary if None)
            n_candidates: Number of proposed instruction candidates to generate per module (original at index 0, then n_candidates proposed)
            tip: Programming tip to guide instruction generation. If None, randomly selects for each candidate.
            program_aware: If True, includes program summary and module role descriptions
            
        Returns:
            Dict mapping predictor indices (0, 1, 2...) to lists of instruction candidates.
            Each list starts with the original instruction at index 0, followed by proposed candidates.
        """
        # Use summaries from grounding_helper if not provided
        if dataset_summ is None:
            dataset_summ = self.grounding_helper.dataset_summary
        if program_summ is None:
            program_summ = self.grounding_helper.program_summary
        
        # Get program code if program-aware
        program_code = None
        if program_aware:
            program_code = self._get_program_code(program)
        
        all_candidates = {}
        module_names = program.get_module_names()

        logger.info(
            "Proposing %d instruction candidates per module for %d modules "
            "(program_aware=%s, fixed_tip=%s)",
            n_candidates,
            len(module_names),
            program_aware,
            tip,
        )

        # For each predictor in the program (by index)
        for predictor_idx, module_name in enumerate(module_names):
            candidates = []
            
            # Get the original instruction from the module (index 0)
            if module_name in program.modules:
                original_instruction = program.modules[module_name].instruction
                if not original_instruction:
                    # Fallback to default if no instruction set
                    original_instruction = self._get_default_instruction(module_name)
            else:
                original_instruction = self._get_default_instruction(module_name)
            
            candidates.append(original_instruction)
            
            # Generate proposed instruction candidates (indices 1, 2, 3, ...)
            for _ in range(n_candidates):
                # Call propose_instruction (it will randomly select a tip if tip=None)
                instruction = self.propose_instruction(
                    module_name=module_name,
                    task_desc=task_desc,
                    bootstrapped_demos=bootstrapped_demos.get(module_name, []),
                    dataset_summ=dataset_summ,
                    program_summ=program_summ,
                    program_code=program_code,
                    tip=tip,  # Pass through - propose_instruction handles None by random selection
                    program_aware=program_aware
                )
                candidates.append(instruction)

            logger.info(
                "Module '%s' (predictor %d): %d total instruction options "
                "(1 original + %d proposed)",
                module_name,
                predictor_idx,
                len(candidates),
                len(candidates) - 1,
            )

            all_candidates[predictor_idx] = candidates

        return all_candidates
