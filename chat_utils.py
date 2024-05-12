import re


def replace_newlines(md_text):
    code_blocks = re.findall(r"```.*?```", md_text, flags=re.DOTALL)
    non_code_blocks = re.split(r"```.*?```", md_text, flags=re.DOTALL)

    transformed_text = non_code_blocks[0].replace("\n", "<br>")
    for i in range(1, len(non_code_blocks)):
        transformed_part = non_code_blocks[i].replace("\n", "<br>")
        if i - 1 < len(code_blocks):
            transformed_text += code_blocks[i - 1]
        transformed_text += transformed_part

    return transformed_text
