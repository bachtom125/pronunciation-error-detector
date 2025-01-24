import numpy as np
def get_nested_position(nested_list, flat_index):
    """
    Finds the nested list and the index within it for a given flat index.

    Args:
        nested_list (list of lists): The list of lists.
        flat_index (int): The flattened index.

    Returns:
        tuple: (nested_list_index, element_index_in_nested_list)
    """
    cumulative_index = 0

    for list_index, sublist in enumerate(nested_list):
        # Check if the flat index falls within the current sublist
        if cumulative_index + len(sublist) > flat_index:
            # Calculate the index within the sublist
            element_index = flat_index - cumulative_index
            return list_index, element_index
        # Update cumulative index
        cumulative_index += len(sublist)
   
    raise IndexError("Index out of range for the flattened list.")

def label_specific_elements_in_reference(reference, start_word_idx, start_element_idx, end_word_idx, end_element_idx, label):
    """
    Labels elements in a nested list between specified start and end indices (inclusive).

    Args:
        reference (list of lists): The original list of lists.
        start_word_idx (int): Index of the starting nested list.
        start_element_idx (int): Index of the starting element in the start list.
        end_word_idx (int): Index of the ending nested list.
        end_element_idx (int): Index of the ending element in the end list.
        label: The label to attach to the elements.

    Returns:
        list of lists: A new list of lists with labels attached where applicable.
    """
    labeled_reference = []
    for word_idx, sublist in enumerate(reference):
        labeled_sublist = []

        for element_idx, element in enumerate(sublist):
            if start_word_idx < end_word_idx:
                # Case 1: start_word_idx < end_word_idx
                if (
                    (word_idx > start_word_idx and word_idx < end_word_idx) or
                    (word_idx == start_word_idx and element_idx >= start_element_idx) or
                    (word_idx == end_word_idx and element_idx <= end_element_idx)
                ):
                    # Attach the label to elements within the inclusive range
                    if isinstance(element, tuple):
                        print(f"There is already a label at index ({word_idx}, {element_idx})") 
                    labeled_sublist.append((element, label))
                else:
                    # Keep elements outside the range unchanged
                    labeled_sublist.append(element)
            elif start_word_idx == end_word_idx:
                # Case 2: start_word_idx == end_word_idx
                if word_idx == start_word_idx and start_element_idx <= element_idx <= end_element_idx:
                    # Attach the label to elements within the inclusive range
                    if isinstance(element, tuple):
                        print(f"There is already a label at index ({word_idx}, {element_idx})") 
                    labeled_sublist.append((element, label))
                else:
                    # Keep elements outside the range unchanged
                    labeled_sublist.append(element)

        labeled_reference.append(labeled_sublist)
    
    return labeled_reference
