import os
from pathlib import Path
from typing import Optional

def load_env_var(var_name: str, prompt_if_missing: bool = True) -> Optional[str]:
    """
    Load environment variable from .env file or prompt user for input.
    
    Args:
        var_name: Name of the environment variable to load
        prompt_if_missing: Whether to prompt user for input if variable is not found
        
    Returns:
        The environment variable value if found or provided by user, else None
        
    Raises:
        OSError: If there are issues reading or writing to the .env file
    """
    # Try to get from environment first
    value = os.getenv(var_name)
    
    if not value:
        # Check for variable in .env file
        env_path = Path('.env')
        try:
            if env_path.exists():
                with open(env_path, encoding='utf-8') as f:
                    for line in f:
                        if line.strip() and line.startswith(f'{var_name}='):
                            value = line.split('=', 1)[1].strip()
                            break
        except OSError as e:
            print(f"Error reading .env file: {e}")
            # Continue execution to allow for user input if prompting is enabled
    
    # Prompt user if still no value and prompting is enabled
    if not value and prompt_if_missing:
        try:
            value = input(f"Please enter your {var_name}: ").strip()
            if value:
                save = input(f"Would you like to save this {var_name} to .env file? (y/N): ").lower()
                if save == 'y':
                    try:
                        # Ensure newline before new entry if file exists and doesn't end with newline
                        newline = '\n' if env_path.exists() and env_path.read_text().strip() else ''
                        with open(env_path, 'a', encoding='utf-8') as f:
                            f.write(f"{newline}{var_name}={value}\n")
                        print(f"{var_name} saved to .env file")
                    except OSError as e:
                        print(f"Error saving to .env file: {e}")
                        # Continue execution since we still have the value in memory
        except (EOFError, KeyboardInterrupt):
            print("\nInput cancelled")
            return None
    
    return value
