
def build_elements_extract_prompt(inputs, prompt_template):
    input_prompt = prompt_template.format(dialogue_history_text=inputs)
    return input_prompt


def build_elements_aware_summary_prompt(inputs, inputs_elements,prompt_template):
    input_prompt = prompt_template.format(dialogue_history_text=inputs, dialogue_elements_text=inputs_elements)
    return input_prompt


