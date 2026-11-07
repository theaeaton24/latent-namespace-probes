"""No metadata, answer keys, family names, or reference bridges enter prompts."""
QUESTIONS = {
    "recognition": "Has a playful imaginative reframe actually been offered in this scene?",
    "resolution": "Which explanation best describes what the final statement does in this scene?",
    "meaning_match": "Does the final line reinterpret this scene's specific unusual detail, or a different detail?",
    "incongruity": "Does the actual result match the stated norm?",
    "expectation": "Does the actual result fulfill or violate this viewer's prediction?",
    "intention": "What communicative intention is supported by the context?",
    "benign_context": "Does the response respect the owner's expressed wishes in this context?",
}
SYSTEM = "Read the scene carefully and follow the requested response format."


def rotations(options):
    return [options[i:] + options[:i] for i in range(len(options))]


def mc_prompt(row, ordered_options):
    choices = "\n".join(f'{chr(65+i)}. {o["text"]}' for i, o in enumerate(ordered_options))
    return f'{row["text"]}\n\n{QUESTIONS[row["task"]]}\n{choices}\nReply with only the option letter.'


def open_prompt(row):
    if row["task"] == "explain":
        question = "In at most two sentences, explain the unusual detail and how the final line connects to it."
    elif row["task"] == "production" and row["variant"] == "playful":
        question = "Write one kind, playful sentence that treats the unusual detail as an intentional feature. Keep the detail true."
    elif row["task"] == "production":
        question = "Write one kind, literal sentence describing the unusual detail without a joke or imaginative reinterpretation."
    else:
        raise ValueError("Not an open-response item")
    return row["text"] + "\n\n" + question
