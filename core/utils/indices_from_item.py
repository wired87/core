def get_indices_after_item(input_list, target_item):
    """
    Finds the index of the target item and returns a list of indices
    for all elements that come after it.
    """

    # Check if the list is empty
    if not input_list:
        print("Input list is empty.")
        return []

    # Get the index of the target item
    try:
        # Get the index of the first occurrence of the target item
        target_index = input_list.index(target_item)
    except ValueError:
        # Handle the case where the item is not found
        print(f"Item '{target_item}' not found in the list.")
        return []

    # Calculate the starting index for the slice (the element *after* the target)
    start_index = target_index + 1

    # Check if the target item is the last element
    if start_index >= len(input_list):
        print(f"Item '{target_item}' is the last element. No entries after it.")
        return []

    # Generate the list of indices from the element after the target to the end
    # Use range to create the sequence of indices
    indices_after = list(range(start_index, len(input_list)))

    return indices_after