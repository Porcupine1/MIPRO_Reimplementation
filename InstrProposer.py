from typing import List, Dict, Any, Optional
from LMBackend import LMBackend
from GroundingHelper import GroundingHelper
from config import N_INSTRUCTION_CANDIDATES
from QAProgram import QAProgram


class InstructionProposer:
    """
    generates instruction candidates for modules using meta-prompting

    - Uses bootstrapped demonstrations from successful traces
    - grounds proposals in dataset/program characteristics
    - meta-optimizes proposal strategy
    """
    
    def __init__(self, lm: LMBackend = None):
        self.lm = lm or LMBackend()
        self.grounding_helper = GroundingHelper(lm=self.lm)

    
    def _parse_instruction(self, raw_output: str) -> str:
        """parse and clean instruction from LM output"""
        instruction = raw_output.strip()
        
        # remove common prefixes
        prefixes = ["Instruction:", "instruction:", "Task:", "task:"]
        for prefix in prefixes:
            if instruction.lower().startswith(prefix.lower()):
                instruction = instruction[len(prefix):].strip()
        
        # take first paragraph if multiple
        if "\n\n" in instruction:
            instruction = instruction.split("\n\n")[0]
        
        return instruction
    
    def _get_default_instruction(self, module_name: str) -> str:
        """get default instruction for a module"""
        defaults = {
            "query": "Generate a search query to answer the question.",
            "answer": "Answer the question based on the context."
        }
        return defaults.get(module_name, "Complete the task.")

    def _build_grounded_meta_prompt(
        self,
        module_name: str,
        task_desc: str,
        bootstrapped_demos: Optional[List[Dict[str, Any]]] = None,
        dataset_summ: Optional[str] = None,
        program_summ: Optional[str] = None
    ) -> str:
        """
        build meta-prompt with grounding
        
        includes:
        - dataset characterization
        - program summary
        - bootstrapped demonstrations from successful traces
        """
        parts = ["You are an expert at writing instructions for language models.\n"]
        
        # task description
        parts.append(f"Task: {task_desc}")
        parts.append(f"Module: {module_name}\n")
        
        # dataset grounding
        if dataset_summ:
            parts.append("Dataset Summary:")
            parts.append(dataset_summ)
            parts.append("")
        
        # program grounding
        if program_summ:
            parts.append("Program Structure:")
            parts.append(program_summ)
            parts.append("")
        
        # bootstrapped demonstrations (successful traces)
        if bootstrapped_demos:
            parts.append("Examples of successful module outputs from high-scoring traces:\n")
            for i, demo in enumerate(bootstrapped_demos, 1):
                parts.append(f"{i}. (score: {demo.get('score', 0):.2f}):")
                
                if module_name == "query":
                    parts.append(f"  Question: {demo['question']}")
                    parts.append(f"  Query: {demo['query']}")
                
                elif module_name == "answer":
                    parts.append(f"  Question: {demo['question']}")
                    parts.append(f"  Context: {demo['context']}...")
                    parts.append(f"  Answer: {demo['answer']}")
                
                parts.append("")
        
        # instruction request
        parts.append(f"Based on this information, generate a clear, specific instruction that will help a language model perform the {module_name} task effectively.\n Instruction:")
        
        return "\n".join(parts)
    
    def _get_module_task_description(self, module_name: str, overall_task: str) -> str:
        """Get task description specific to a module."""
        module_tasks = {
            "query": f"Given a question, generate an effective search query that will retrieve relevant context. Overall task: {overall_task}",
            "answer": f"Given a question and retrieved context, generate an accurate answer. Overall task: {overall_task}"
        }
        return module_tasks.get(module_name, overall_task)
    
    def propose_instructions(
        self,
        module_name: str,
        task_desc: str,
        bootstrapped_demos: Optional[List[Dict[str, Any]]] = None,
        dataset_summ: Optional[str] = None,
        program_summ: Optional[str] = None,
        n_candidates: int = N_INSTRUCTION_CANDIDATES
    ) -> List[str]:
        """
        generate instruction candidates for a module
        
        args:
            module_name: name of module
            task_description: Overall task description
            data_sample: sample of raw data
            bootstrapped_demos: successful traces (from DemoBootstrapper)
            dataset_summary: characterization of dataset patterns
            program_summary: description of program flow
            n_candidates: number of candidates to generate
            
        returns:
            list of instruction candidates
        """
        
        # build meta-prompt with grounding
        meta_prompt = self._build_grounded_meta_prompt(
            module_name=module_name,
            task_desc=task_desc,
            bootstrapped_demos=bootstrapped_demos,
            dataset_summ=dataset_summ,
            program_summ=program_summ
        )
        
        candidates = []
        MAX_TRIES = n_candidates * 3  # Allow enough tries to get n_candidates
        tries = 0
        while len(candidates) < n_candidates and tries < MAX_TRIES:
            instruction = self.lm.generate(meta_prompt, temperature=0.9)
            instruction = self._parse_instruction(instruction)
            if instruction and instruction not in candidates:
                candidates.append(instruction)
            tries += 1
        
        if not candidates:
            candidates.append(self._get_default_instruction(module_name))
        
        return candidates

    def propose_for_all_modules(
        self,
        program: QAProgram,
        task_desc: str,
        bootstrapped_demos: Dict[str, List[Dict[str, Any]]],
        dataset_summ: Optional[str] = None,
        program_summ: Optional[str] = None,
        n_candidates: int = N_INSTRUCTION_CANDIDATES
    ) -> Dict[str, List[str]]:
        """generate instruction candidates for all modules in a program"""
        all_candidates = {}

        for module_name in program.get_module_names():
            module_task = self._get_module_task_description(module_name, task_desc)
            candidates = self.propose_instructions(
                module_name=module_name,
                task_desc=module_task,
                bootstrapped_demos=bootstrapped_demos.get(module_name, []),
                dataset_summ=dataset_summ,
                program_summ=program_summ,
                n_candidates=n_candidates
            )
            all_candidates[module_name] = candidates

        return all_candidates
