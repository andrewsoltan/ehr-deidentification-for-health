def reconstruct_and_append_text(original_text, classified_tokens):
    """
    Reconstructs the text based on BERT output, merging entities and appending reconstructed sections at the token level,
    while keeping track of merged tokens.

    Args:
        original_text (str): The original input text.
        classified_tokens (list): List of token classification results with start/end positions.

    Returns:
        str: The reconstructed text with entity labels.
    """
    reconstructed_output = ""
    current_position = 0
    merged_results = []
    current_entity = None
    removed_tokens_count = 0  # To keep track of merged (removed) tokens

    # Merge entities within a +/- 2 character tolerance
    for result in classified_tokens:
        result['entity'] = clean_label(result['entity'])

        # If the current entity exists and is the same as the new one, merge them
        if current_entity and (current_entity['entity'] == result['entity']) and (current_entity['end'] - 2 <= result['start'] <= current_entity['end'] + 2):
            # Merge the token into the current entity and extend the end position
            current_entity['word'] += result['word'].replace("##", "")
            current_entity['end'] = result['end']
            removed_tokens_count += 1  # Count the token as merged (removed)
        else:
            # Add the previous entity (if exists) and reset the current entity
            if current_entity:
                merged_results.append(current_entity)
            current_entity = result
            removed_tokens_count = 0  # Reset removed tokens count for the new entity

    # Append the last entity if any
    if current_entity:
        merged_results.append(current_entity)

    # Reconstruct the text
    for result in merged_results:
        start = result["start"]
        end = result["end"]
        label = result["entity"]

        # Add the part of the original text before the entity
        reconstructed_output += original_text[current_position:start]
        # Add the entity with its label
        reconstructed_output += f"[{label}]"
        
        # Update the current position after appending the entity
        current_position = end

    # After all entities are processed, adjust for the number of removed tokens
    reconstructed_output += original_text[current_position:]
    
    return reconstructed_output


def clean_label(label):
    """Remove BILOU prefix before the hyphen."""
    return label.split('-')[-1]
