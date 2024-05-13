import re


def convert_code_blocks_to_tags(text):
    pattern = r"```([A-Za-z0-9_+\-]+)?\n?([\s\S]*?)\n?```"

    def replace(match):
        language = match.group(1)
        code = match.group(2)
        if language:
            return f"<pre><code class=\"language-{language}\">{code}</code></pre>"
        else:
            return f"<pre><code>{code}</code></pre>"
    converted_text = re.sub(pattern, replace, text, flags=re.DOTALL)
    return converted_text


def convert_code_tags_to_md(html_text):
    pattern = r"<pre><code(?: class=\"language-([A-Za-z0-9_+\-]+)\")?>([\s\S]*?)</code></pre>"

    def replace(match):
        language = match.group(1)
        code = match.group(2)
        if language:
            return f"```{language}\n{code}\n```"
        else:
            return f"```\n{code}\n```"
    converted_text = re.sub(pattern, replace, html_text, flags=re.DOTALL)
    return converted_text


def replace_newlines_code_blocks(md_text):
    code_blocks = re.findall(r"```[\s\S]*?```", md_text, flags=re.DOTALL)
    non_code_blocks = re.split(r"```[\s\S]*?```", md_text, flags=re.DOTALL)

    transformed_text = non_code_blocks[0].replace("\n", "<br>")
    for i in range(1, len(non_code_blocks)):
        transformed_part = non_code_blocks[i].replace("\n", "<br>")
        if i - 1 < len(code_blocks):
            transformed_text += code_blocks[i - 1]
        transformed_text += transformed_part

    transformed_text = convert_code_blocks_to_tags(transformed_text)

    return transformed_text
