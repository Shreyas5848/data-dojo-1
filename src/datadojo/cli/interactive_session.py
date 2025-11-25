from prompt_toolkit import prompt

def start_interactive_session():
    """Starts the interactive Dojo session (REPL)."""
    print("Welcome to the DataDojo Interactive Session!")
    print("Type 'exit' to quit.")

    while True:
        try:
            user_input = prompt("(dojo) > ")
            if user_input.lower() == 'exit':
                break
            
            # For now, just echo the input
            print(f"You entered: {user_input}")

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            break
        except EOFError:
            # Handle Ctrl+D gracefully
            break

    print("Exiting DataDojo session. Goodbye!")
