"""Utility functions for creating interactive command-line menus."""


def choose_from_list(prompt_text, choices, default_index=0):
    if len(choices) == 1:
        print(f"{prompt_text}: {choices[0]}")
        return choices[0]

    print(f"\n{prompt_text}")
    for i, choice in enumerate(choices, start=1):
        print(f"  {i}. {choice}")
    print(f" [default: {default_index + 1}]")

    while True:
        answer = input("Select a number: ").strip()
        if not answer:
            return choices[default_index]
        try:
            idx = int(answer) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        except ValueError:
            pass
        print("Invalid choice. Please enter a valid number.")


def choose_multiple(prompt_text, choices, default_choices=None):
    default_choices = default_choices or choices
    print(f"\n{prompt_text}")
    for i, choice in enumerate(choices, start=1):
        marker = "x" if choice in default_choices else " "
        print(f"  {i}. [{marker}] {choice}")

    print("Enter numbers separated by commas, or press Enter for default \n(x marker indicates default selection)")
    while True:
        answer = input("Selection: ").strip()
        if not answer:
            return list(default_choices)

        try:
            selected = []
            for part in answer.split(","):
                value = int(part.strip())
                if 1 <= value <= len(choices):
                    selected.append(choices[value - 1])
            if selected:
                return selected
        except ValueError:
            pass
        print("Invalid selection. Example: 1,3,5")


def ask_int(prompt_text, default_value, minimum=1):
    while True:
        answer = input(f"\nEnter {prompt_text} or press Enter for default \n(default: [{default_value}]): ").strip()
        if not answer:
            return default_value
        try:
            value = int(answer)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"Please enter an integer >= {minimum}.")


def ask_string(prompt_text, default_value):
    answer = input(f"\nEnter {prompt_text} or press Enter for default \n(default: [{default_value}]): ").strip()
    return answer if answer else default_value