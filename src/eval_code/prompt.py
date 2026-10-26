sft_prompt = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request."
    "\n\n### Instruction:\n{instruction}\n\n### Response:{response}"
)


# Public release keeps a single canonical prompt for continual sequential recommendation.
all_prompt = {
    "seqrec": [
        {
            "instruction": (
                "The user has interacted with items {inters} in chronological order. "
                "Can you predict the next possible item that the user may expect?"
            ),
            "response": "{item}",
        }
    ]
}
